import struct
import zlib
from pathlib import Path

from chessmoe.models.tiny_model import TinyChessNet
from chessmoe.training.checkpoint import load_training_checkpoint, save_checkpoint
from chessmoe.training.distill import run_distillation
from chessmoe.training.distill_config import DistillationConfig
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


def _write_indexed_chunk(tmp_path: Path, sample_count: int = 4) -> tuple[Path, Path]:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    chunk_path = replay_dir / "tiny.cmrep"
    samples = [
        _sample_bytes(game_id=5, ply_index=ply, move_from=12, move_to=28)
        for ply in range(sample_count)
    ]
    chunk_path.write_bytes(_chunk_bytes(samples))
    db_path = tmp_path / "replay.sqlite"
    index_replay_file(db_path, chunk_path)
    return db_path, chunk_path


def test_distillation_checkpoint_saves_student(tmp_path: Path):
    db_path, _ = _write_indexed_chunk(tmp_path, sample_count=4)
    teacher_checkpoint = tmp_path / "teacher.pt"
    teacher = TinyChessNet(channels=4, hidden=16)
    save_checkpoint(teacher, teacher_checkpoint)

    checkpoint_path = tmp_path / "checkpoints" / "student.pt"
    metrics_path = tmp_path / "metrics.jsonl"

    run_distillation(
        DistillationConfig(
            replay_index=db_path,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            teacher_checkpoint=teacher_checkpoint,
            epochs=1,
            batch_size=2,
            learning_rate=0.01,
            train_fraction=1.0,
            validation_fraction=0.0,
            seed=42,
            device="cpu",
            amp=False,
            student_kind="student_hybrid",
            student_hybrid_conv_channels=8,
            student_hybrid_d_model=16,
            student_hybrid_layers=1,
            student_hybrid_heads=2,
            student_hybrid_ffn_dim=32,
            student_hybrid_dropout=0.0,
            temperature=1.0,
            policy_kl_weight=1.0,
            wdl_kl_weight=1.0,
            value_weight=1.0,
            moves_left_weight=1.0,
            hard_target_weight=0.0,
            hard_value_weight=0.0,
            target_policy="original",
        )
    )

    checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")

    assert checkpoint.epoch == 1
    assert checkpoint.metadata["config"]["teacher_checkpoint"].endswith("teacher.pt")
    assert checkpoint_path.exists()
    assert metrics_path.exists()
