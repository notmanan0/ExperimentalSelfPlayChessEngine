import sqlite3
import struct
import zlib
from pathlib import Path

import torch

from chessmoe.models.encoding import BOARD_SHAPE, NUM_MOVE_BUCKETS, move_to_index
from chessmoe.training.checkpoint import load_training_checkpoint
from chessmoe.training.data import ReplayDataset, TrainingBatch, collate_replay_samples
from chessmoe.training.train import run_training
from chessmoe.training.config import TrainingConfig
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


def test_replay_dataset_loads_samples_from_sqlite_index(tmp_path: Path):
    db_path, _ = _write_indexed_chunk(tmp_path, sample_count=3)

    dataset = ReplayDataset.from_index(db_path)

    assert len(dataset) == 3
    item = dataset[0]
    assert item.features.shape == BOARD_SHAPE
    assert item.policy.shape == (NUM_MOVE_BUCKETS,)
    assert item.policy[move_to_index("e2e4")].item() == 1.0
    assert item.wdl.item() == 0
    assert item.moves_left.item() == 3.0


def test_collate_replay_samples_returns_batch_shapes(tmp_path: Path):
    db_path, _ = _write_indexed_chunk(tmp_path, sample_count=2)
    dataset = ReplayDataset.from_index(db_path)

    batch = collate_replay_samples([dataset[0], dataset[1]])

    assert isinstance(batch, TrainingBatch)
    assert batch.features.shape == (2, *BOARD_SHAPE)
    assert batch.policy.shape == (2, NUM_MOVE_BUCKETS)
    assert batch.wdl.shape == (2,)
    assert batch.moves_left.shape == (2,)


def test_training_loss_decreases_on_tiny_replay_dataset(tmp_path: Path):
    db_path, _ = _write_indexed_chunk(tmp_path, sample_count=8)
    checkpoint_path = tmp_path / "checkpoints" / "tiny.pt"
    metrics_path = tmp_path / "metrics.jsonl"

    result = run_training(
        TrainingConfig(
            replay_index=db_path,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            epochs=6,
            batch_size=4,
            learning_rate=0.01,
            weight_decay=0.0,
            train_fraction=1.0,
            validation_fraction=0.0,
            seed=123,
            device="cpu",
            compile_model=False,
            amp=False,
            model_channels=8,
            model_hidden=32,
        )
    )

    assert result.train_losses[-1] < result.train_losses[0]
    assert checkpoint_path.exists()
    assert metrics_path.exists()


def test_training_checkpoint_resume_continues_epoch_count(tmp_path: Path):
    db_path, _ = _write_indexed_chunk(tmp_path, sample_count=6)
    checkpoint_path = tmp_path / "checkpoints" / "resume.pt"
    metrics_path = tmp_path / "metrics.jsonl"

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

    checkpoint = load_training_checkpoint(
        checkpoint_path,
        model_channels=8,
        model_hidden=32,
        map_location="cpu",
    )

    assert first.start_epoch == 0
    assert resumed.start_epoch == 1
    assert checkpoint.epoch == 2
