import json
import sqlite3
import struct
import zlib
from pathlib import Path

import torch
import pytest

from chessmoe.models.encoding import move_to_index
from chessmoe.training.data import ReplayDataset
from replay.index import index_replay_file
from replay.reanalysis import (
    ReanalysisConfig,
    ReanalysisTarget,
    ReanalysisTargetPolicy,
    SimpleModelAnalyzer,
    load_latest_reanalysis_target,
    reanalyze_index,
    select_replay_chunks,
    write_reanalysis_target,
)


HEADER = struct.Struct("<8sHHIIIIIQQI12s")
HEADER_SIZE = 64
MAGIC = b"CMREPLAY"


def _move(from_square: int, to_square: int, promotion: int = 0) -> bytes:
    encoded = from_square | (to_square << 6) | (promotion << 12)
    return struct.pack("<H", encoded)


def _sample_bytes(game_id: int, ply_index: int, root_value: float = 0.25) -> bytes:
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
            1,
            root_value,
            16,
            game_id,
            ply_index,
            2,
            2,
        )
    )
    body.extend(_move(12, 28))
    body.extend(_move(6, 21))
    body.extend(_move(12, 28))
    body.extend(struct.pack("<If", 12, 0.75))
    body.extend(_move(6, 21))
    body.extend(struct.pack("<If", 4, 0.25))
    return struct.pack("<I", len(body)) + body


def _chunk_bytes(
    samples: list[bytes],
    *,
    model_version: int,
    timestamp_ms: int,
) -> bytes:
    payload = b"".join(samples)
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    header = HEADER.pack(
        MAGIC,
        1,
        HEADER_SIZE,
        0,
        len(samples),
        0,
        model_version,
        1,
        timestamp_ms,
        len(payload),
        checksum,
        b"\0" * 12,
    )
    return header + payload


def _indexed_chunk(
    tmp_path: Path,
    *,
    model_version: int = 3,
    timestamp_ms: int = 1_715_000_000_000,
    root_value: float = 0.25,
) -> tuple[Path, Path]:
    chunk_path = tmp_path / f"chunk-{model_version}.cmrep"
    db_path = tmp_path / "replay.sqlite"
    chunk_path.write_bytes(
        _chunk_bytes(
            [_sample_bytes(game_id=44, ply_index=0, root_value=root_value)],
            model_version=model_version,
            timestamp_ms=timestamp_ms,
        )
    )
    index_replay_file(db_path, chunk_path)
    return db_path, chunk_path


def test_select_replay_chunks_filters_by_model_age_and_priority(tmp_path: Path):
    db_path, old_chunk = _indexed_chunk(
        tmp_path,
        model_version=2,
        timestamp_ms=1_000,
    )
    _, new_chunk = _indexed_chunk(
        tmp_path,
        model_version=5,
        timestamp_ms=9_000,
    )

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "insert into chunk_priorities(path, sampling_priority) values (?, ?)",
            (str(old_chunk.resolve()), 0.9),
        )
        connection.execute(
            "insert into chunk_priorities(path, sampling_priority) values (?, ?)",
            (str(new_chunk.resolve()), 0.1),
        )

    selected = select_replay_chunks(
        db_path,
        model_versions={2},
        older_than_timestamp_ms=5_000,
        minimum_sampling_priority=0.5,
    )

    assert selected == [old_chunk.resolve()]


def test_reanalysis_preserves_original_chunk_and_indexes_targets(tmp_path: Path):
    db_path, chunk_path = _indexed_chunk(tmp_path, model_version=1)
    original_bytes = chunk_path.read_bytes()

    summary = reanalyze_index(
        ReanalysisConfig(
            replay_index=db_path,
            output_index=db_path,
            current_model_version=9,
            search_budget=32,
            reanalysis_timestamp_ms=2_000,
            max_chunks=1,
        ),
        analyzer=SimpleModelAnalyzer(policy_bias={"g1f3": 3.0}, root_value=0.6),
    )

    assert chunk_path.read_bytes() == original_bytes
    assert summary.samples_reanalyzed == 1

    target = load_latest_reanalysis_target(db_path, chunk_path, game_id=44, ply_index=0)
    assert target is not None
    assert target.model_version == 9
    assert target.search_budget == 32
    assert target.reanalysis_timestamp_ms == 2_000
    assert target.root_value == 0.6
    assert target.policy[0]["move"] == "g1f3"


def test_target_replacement_policy_prefers_latest_reanalysis(tmp_path: Path):
    db_path, chunk_path = _indexed_chunk(tmp_path, model_version=1)
    write_reanalysis_target(
        db_path,
        ReanalysisTarget(
            chunk_path=chunk_path.resolve(),
            game_id=44,
            ply_index=0,
            source_model_version=1,
            model_version=10,
            search_budget=8,
            reanalysis_timestamp_ms=100,
            root_value=-0.2,
            policy=[
                {"move": "e2e4", "visit_count": 1, "probability": 0.1},
                {"move": "g1f3", "visit_count": 9, "probability": 0.9},
            ],
        ),
    )

    original = ReplayDataset.from_index(
        db_path,
        target_policy=ReanalysisTargetPolicy.ORIGINAL,
    )[0]
    latest = ReplayDataset.from_index(
        db_path,
        target_policy=ReanalysisTargetPolicy.LATEST_REANALYSIS,
    )[0]

    assert original.policy[move_to_index("e2e4")].item() == 0.75
    assert latest.policy[move_to_index("g1f3")].item() == pytest.approx(0.9)


def test_dataset_can_mix_original_and_reanalyzed_targets(tmp_path: Path):
    db_path, chunk_path = _indexed_chunk(tmp_path, model_version=1)
    write_reanalysis_target(
        db_path,
        ReanalysisTarget(
            chunk_path=chunk_path.resolve(),
            game_id=44,
            ply_index=0,
            source_model_version=1,
            model_version=11,
            search_budget=16,
            reanalysis_timestamp_ms=200,
            root_value=0.1,
            policy=[
                {"move": "e2e4", "visit_count": 0, "probability": 0.2},
                {"move": "g1f3", "visit_count": 0, "probability": 0.8},
            ],
        ),
    )

    reanalyzed = ReplayDataset.from_index(
        db_path,
        target_policy=ReanalysisTargetPolicy.MIX,
        reanalysis_fraction=1.0,
        reanalysis_seed=1,
    )[0]
    original = ReplayDataset.from_index(
        db_path,
        target_policy=ReanalysisTargetPolicy.MIX,
        reanalysis_fraction=0.0,
        reanalysis_seed=1,
    )[0]

    assert reanalyzed.policy[move_to_index("g1f3")].item() == pytest.approx(0.8)
    assert original.policy[move_to_index("e2e4")].item() == 0.75
    assert torch.equal(
        reanalyzed.features,
        original.features,
    )


def test_reanalysis_target_json_schema_is_stable(tmp_path: Path):
    db_path, chunk_path = _indexed_chunk(tmp_path, model_version=4)
    target = ReanalysisTarget(
        chunk_path=chunk_path.resolve(),
        game_id=44,
        ply_index=0,
        source_model_version=4,
        model_version=12,
        search_budget=64,
        reanalysis_timestamp_ms=300,
        root_value=0.7,
        policy=[{"move": "e2e4", "visit_count": 64, "probability": 1.0}],
    )

    write_reanalysis_target(db_path, target)

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "select policy_json from reanalysis_targets where model_version = 12"
        ).fetchone()

    assert json.loads(row[0]) == target.policy
