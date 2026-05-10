import sqlite3
import struct
import zlib
from pathlib import Path

from chessmoe.training.checkpoint import load_training_checkpoint
from chessmoe.training.config import TrainingConfig, load_training_config
from chessmoe.training.train import run_training
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


def _write_indexed_chunk(tmp_path: Path, sample_count: int = 6) -> tuple[Path, Path]:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    chunk_path = replay_dir / "tiny.cmrep"
    samples = [
        _sample_bytes(game_id=9, ply_index=ply, move_from=12, move_to=28)
        for ply in range(sample_count)
    ]
    chunk_path.write_bytes(_chunk_bytes(samples))
    db_path = tmp_path / "replay.sqlite"
    index_replay_file(db_path, chunk_path)
    return db_path, chunk_path


def test_training_config_loads_dense_transformer_fields(tmp_path: Path):
    config_path = tmp_path / "dense_transformer.json"
    config_path.write_text(
        """{
          "replay_index": "replay.sqlite",
          "checkpoint_path": "checkpoints/model.pt",
          "metrics_path": "metrics.jsonl",
          "model_kind": "dense_transformer",
          "transformer_d_model": 32,
          "transformer_layers": 1,
          "transformer_heads": 4,
          "transformer_ffn_dim": 64,
          "transformer_dropout": 0.0
        }""",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.model_kind == "dense_transformer"
    assert config.transformer_d_model == 32
    assert config.transformer_layers == 1
    assert config.transformer_heads == 4
    assert config.transformer_ffn_dim == 64


def test_dense_transformer_training_loop_saves_resumeable_checkpoint(tmp_path: Path):
    db_path, _ = _write_indexed_chunk(tmp_path, sample_count=6)
    checkpoint_path = tmp_path / "checkpoints" / "dense_transformer.pt"
    metrics_path = tmp_path / "metrics.jsonl"

    result = run_training(
        TrainingConfig(
            replay_index=db_path,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            epochs=1,
            batch_size=3,
            learning_rate=0.001,
            train_fraction=1.0,
            validation_fraction=0.0,
            seed=77,
            device="cpu",
            compile_model=False,
            amp=False,
            model_kind="dense_transformer",
            transformer_d_model=32,
            transformer_layers=1,
            transformer_heads=4,
            transformer_ffn_dim=64,
            transformer_dropout=0.0,
        )
    )

    checkpoint = load_training_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )

    assert result.epochs_completed == 1
    assert checkpoint.epoch == 1
    assert checkpoint.metadata["config"]["model_kind"] == "dense_transformer"
