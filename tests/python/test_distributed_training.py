from __future__ import annotations

import os
import socket
import struct
import zlib
from pathlib import Path

import pytest
import torch

from chessmoe.training.checkpoint import load_training_checkpoint
from chessmoe.training.config import TrainingConfig
from chessmoe.training.distributed import DistributedContext, should_log
from chessmoe.training.train import run_training, _append_metrics
from replay.index import index_replay_file


HEADER = struct.Struct("<8sHHIIIIIQQI12s")
HEADER_SIZE = 64
MAGIC = b"CMREPLAY"


def _move(from_square: int, to_square: int, promotion: int = 0) -> bytes:
    encoded = from_square | (to_square << 6) | (promotion << 12)
    return struct.pack("<H", encoded)


def _sample_bytes(game_id: int, ply_index: int, move_from: int, move_to: int) -> bytes:
    board = bytearray(64)
    board[4] = 6
    board[12] = 1
    board[60] = 12
    board[52] = 7

    body = bytearray()
    body.extend(board)
    body.extend(
        struct.pack(
            "<BBBHHBfIQIHH",
            0,
            0b1111,
            64,
            0,
            1,
            2,
            0.8,
            16,
            game_id,
            ply_index,
            1,
            1,
        )
    )
    body.extend(_move(move_from, move_to))
    body.extend(_move(move_from, move_to))
    body.extend(struct.pack("<If", 16, 1.0))
    return struct.pack("<I", len(body)) + body


def _chunk_bytes(samples: list[bytes]) -> bytes:
    payload = b"".join(samples)
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    header = HEADER.pack(
        MAGIC,
        1,
        HEADER_SIZE,
        0,
        len(samples),
        0,
        1,
        1,
        1_715_000_000_000,
        len(payload),
        checksum,
        b"\0" * 12,
    )
    return header + payload


def _write_indexed_chunk(tmp_path: Path, sample_count: int = 8) -> tuple[Path, Path]:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    chunk_path = replay_dir / "tiny.cmrep"
    samples = [
        _sample_bytes(game_id=7, ply_index=ply, move_from=12, move_to=28)
        for ply in range(sample_count)
    ]
    chunk_path.write_bytes(_chunk_bytes(samples))
    db_path = tmp_path / "replay.sqlite"
    index_replay_file(db_path, chunk_path)
    return db_path, chunk_path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _set_dist_env(world_size: int, rank: int) -> dict[str, str | None]:
    backup = {key: os.environ.get(key) for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]}
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    return backup


def _restore_env(backup: dict[str, str | None]) -> None:
    for key, value in backup.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_ddp_single_process_matches_non_distributed(tmp_path: Path):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")

    db_path, _ = _write_indexed_chunk(tmp_path, sample_count=8)

    base = TrainingConfig(
        replay_index=db_path,
        checkpoint_path=tmp_path / "single.pt",
        metrics_path=tmp_path / "single.jsonl",
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        weight_decay=0.0,
        train_fraction=1.0,
        validation_fraction=0.0,
        seed=7,
        device="cpu",
        compile_model=False,
        amp=False,
        deterministic=True,
        model_channels=8,
        model_hidden=32,
    )

    run_training(base)

    backup = _set_dist_env(world_size=1, rank=0)
    try:
        ddp_config = TrainingConfig(
            **{
                **base.__dict__,
                "checkpoint_path": tmp_path / "ddp.pt",
                "metrics_path": tmp_path / "ddp.jsonl",
                "distributed": True,
                "distributed_backend": "gloo",
                "distributed_init_method": "env://",
            }
        )
        run_training(ddp_config)
    finally:
        _restore_env(backup)

    single = load_training_checkpoint(
        base.checkpoint_path,
        model_channels=8,
        model_hidden=32,
        map_location="cpu",
    ).model
    ddp = load_training_checkpoint(
        tmp_path / "ddp.pt",
        model_channels=8,
        model_hidden=32,
        map_location="cpu",
    ).model

    for p_single, p_ddp in zip(single.parameters(), ddp.parameters(), strict=True):
        torch.testing.assert_close(p_single, p_ddp)


def test_ddp_resume_checkpoint(tmp_path: Path):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")

    db_path, _ = _write_indexed_chunk(tmp_path, sample_count=6)
    checkpoint_path = tmp_path / "resume.pt"
    metrics_path = tmp_path / "metrics.jsonl"

    backup = _set_dist_env(world_size=1, rank=0)
    try:
        base = TrainingConfig(
            replay_index=db_path,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            epochs=1,
            batch_size=3,
            learning_rate=0.005,
            train_fraction=1.0,
            validation_fraction=0.0,
            seed=321,
            device="cpu",
            compile_model=False,
            amp=False,
            distributed=True,
            distributed_backend="gloo",
            distributed_init_method="env://",
            model_channels=8,
            model_hidden=32,
        )
        first = run_training(base)
        resumed = run_training(
            TrainingConfig(
                **{
                    **base.__dict__,
                    "epochs": 2,
                    "resume_checkpoint": checkpoint_path,
                }
            )
        )
    finally:
        _restore_env(backup)

    checkpoint = load_training_checkpoint(
        checkpoint_path,
        model_channels=8,
        model_hidden=32,
        map_location="cpu",
    )

    assert first.start_epoch == 0
    assert resumed.start_epoch == 1
    assert checkpoint.epoch == 2


def test_rank_safe_metrics_writes(tmp_path: Path):
    context = DistributedContext(
        enabled=True,
        backend="gloo",
        rank=1,
        world_size=2,
        local_rank=1,
        device=torch.device("cpu"),
    )
    metrics_path = tmp_path / "metrics.jsonl"
    if should_log(context, log_all_ranks=False):
        _append_metrics(metrics_path, {"epoch": 1, "train": {"loss": 1.0}})
    assert not metrics_path.exists()
