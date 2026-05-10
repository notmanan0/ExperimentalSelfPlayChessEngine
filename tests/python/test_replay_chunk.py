import sqlite3
import struct
import zlib
from pathlib import Path

import pytest

from replay.index import index_replay_file
from replay.reader import (
    ReplayChecksumError,
    ReplayReader,
    UnsupportedReplayVersion,
)


HEADER = struct.Struct("<8sHHIIIIIQQI12s")
CURRENT_VERSION = 1
HEADER_SIZE = 64
MAGIC = b"CMREPLAY"


def _move(from_square: int, to_square: int, promotion: int = 0) -> bytes:
    encoded = from_square | (to_square << 6) | (promotion << 12)
    return struct.pack("<H", encoded)


def _sample_bytes() -> bytes:
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
            0.25,
            99,
            12,
            2,
            2,
            2,
        )
    )
    body.extend(_move(12, 28))
    body.extend(_move(6, 21))
    body.extend(_move(12, 28))
    body.extend(struct.pack("<If", 32, 0.8))
    body.extend(_move(6, 21))
    body.extend(struct.pack("<If", 8, 0.2))
    return struct.pack("<I", len(body)) + body


def make_chunk(*, version: int = CURRENT_VERSION, mutate_payload=None) -> bytes:
    payload = bytearray(_sample_bytes())
    if mutate_payload is not None:
        mutate_payload(payload)
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    header = HEADER.pack(
        MAGIC,
        version,
        HEADER_SIZE,
        0,
        1,
        0,
        17,
        3,
        1_715_000_000_000,
        len(payload),
        checksum,
        b"\0" * 12,
    )
    return header + payload


def test_reader_decodes_replay_chunk_sample():
    chunk = ReplayReader.read_bytes(make_chunk())

    assert chunk.header.sample_count == 1
    assert chunk.header.model_version == 17
    assert chunk.header.generator_version == 3
    assert chunk.header.creation_timestamp_ms == 1_715_000_000_000
    assert chunk.samples[0].side_to_move == "white"
    assert chunk.samples[0].castling_rights == 0b1111
    assert chunk.samples[0].en_passant_square is None
    assert chunk.samples[0].halfmove_clock == 0
    assert chunk.samples[0].fullmove_number == 1
    assert chunk.samples[0].legal_moves == ["e2e4", "g1f3"]
    assert chunk.samples[0].policy[0].move == "e2e4"
    assert chunk.samples[0].policy[0].visit_count == 32
    assert chunk.samples[0].final_wdl == "draw"
    assert chunk.samples[0].root_value == pytest.approx(0.25)
    assert chunk.samples[0].search_budget == 99
    assert chunk.samples[0].game_id == 12
    assert chunk.samples[0].ply_index == 2
    assert chunk.samples[0].board[4] == "K"
    assert chunk.samples[0].board[60] == "k"


def test_reader_rejects_payload_checksum_mismatch():
    raw = bytearray(make_chunk())
    raw[-1] ^= 0xFF

    with pytest.raises(ReplayChecksumError):
        ReplayReader.read_bytes(bytes(raw))


def test_reader_rejects_newer_version():
    with pytest.raises(UnsupportedReplayVersion):
        ReplayReader.read_bytes(make_chunk(version=CURRENT_VERSION + 1))


def test_index_records_chunk_metadata():
    output_dir = Path("python-test-output/replay-tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = output_dir / "sample.cmrep"
    db_path = output_dir / "replay.sqlite"
    if db_path.exists():
        db_path.unlink()
    chunk_path.write_bytes(make_chunk())

    index_replay_file(db_path, chunk_path)

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "select sample_count, model_version, generator_version, checksum from chunks"
        ).fetchone()

    assert row == (1, 17, 3, zlib.crc32(_sample_bytes()) & 0xFFFFFFFF)
