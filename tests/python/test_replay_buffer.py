from __future__ import annotations

import json
from pathlib import Path

from chessmoe.analysis.replay_buffer import (
    RollingReplayBuffer,
    deduplicate_replay_index,
    detect_duplicate_positions,
    compute_replay_statistics,
    split_dataset_by_game,
)


def test_deduplicate_replay_index_no_duplicates(tmp_path: Path):
    import sqlite3
    db = tmp_path / "test.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("""
            CREATE TABLE chunks (
                path TEXT PRIMARY KEY,
                magic TEXT, version INTEGER, compression_flags INTEGER,
                sample_count INTEGER, model_version INTEGER,
                generator_version INTEGER, creation_timestamp_ms INTEGER,
                payload_size INTEGER, checksum INTEGER, indexed_at_ms INTEGER
            )
        """)
        conn.execute("INSERT INTO chunks VALUES ('a.cmrep','CM',1,0,10,0,14,1000,100,0,1000)")
        conn.execute("INSERT INTO chunks VALUES ('b.cmrep','CM',1,0,10,0,14,1001,100,0,1001)")
    removed = deduplicate_replay_index(db)
    assert removed == 0


def test_split_dataset_by_game(tmp_path: Path):
    import sqlite3
    db = tmp_path / "test.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("""
            CREATE TABLE chunks (
                path TEXT PRIMARY KEY,
                magic TEXT, version INTEGER, compression_flags INTEGER,
                sample_count INTEGER, model_version INTEGER,
                generator_version INTEGER, creation_timestamp_ms INTEGER,
                payload_size INTEGER, checksum INTEGER, indexed_at_ms INTEGER
            )
        """)
        for i in range(10):
            conn.execute(
                "INSERT INTO chunks VALUES (?,'CM',1,0,10,0,14,?,100,0,?)",
                (f"game_{i}.cmrep", i * 1000, i * 1000),
            )
    train, val = split_dataset_by_game(db, train_fraction=0.8, seed=42)
    assert len(train) == 8
    assert len(val) == 2
    assert len(set(train) & set(val)) == 0


def test_rolling_buffer_prune(tmp_path: Path):
    import sqlite3
    db = tmp_path / "test.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE chunks (
                path TEXT PRIMARY KEY, magic TEXT, version INTEGER,
                compression_flags INTEGER, sample_count INTEGER,
                model_version INTEGER, generator_version INTEGER,
                creation_timestamp_ms INTEGER, payload_size INTEGER,
                checksum INTEGER, indexed_at_ms INTEGER
            );
            CREATE TABLE chunk_priorities (
                path TEXT PRIMARY KEY, sampling_priority REAL,
                updated_at_ms INTEGER DEFAULT 0
            );
        """)
        for i in range(20):
            conn.execute(
                "INSERT INTO chunks VALUES (?,'CM',1,0,10,0,14,?,100,0,?)",
                (f"g{i}.cmrep", i * 1000, i * 1000),
            )
    buffer = RollingReplayBuffer(db, max_chunks=10)
    pruned = buffer.prune_old_chunks()
    assert pruned == 10
    with sqlite3.connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == 10


def test_rolling_buffer_maintain(tmp_path: Path):
    import sqlite3
    db = tmp_path / "test.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE chunks (
                path TEXT PRIMARY KEY, magic TEXT, version INTEGER,
                compression_flags INTEGER, sample_count INTEGER,
                model_version INTEGER, generator_version INTEGER,
                creation_timestamp_ms INTEGER, payload_size INTEGER,
                checksum INTEGER, indexed_at_ms INTEGER
            );
            CREATE TABLE chunk_priorities (
                path TEXT PRIMARY KEY, sampling_priority REAL,
                updated_at_ms INTEGER DEFAULT 0
            );
        """)
        for i in range(5):
            conn.execute(
                "INSERT INTO chunks VALUES (?,'CM',1,0,10,0,14,?,100,0,?)",
                (f"g{i}.cmrep", i * 1000, i * 1000),
            )
    buffer = RollingReplayBuffer(db, max_chunks=10, decay_rate=0.9)
    result = buffer.maintain()
    assert "pruned" in result
    assert "priorities_updated" in result
