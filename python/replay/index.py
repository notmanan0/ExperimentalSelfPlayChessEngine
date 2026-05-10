from __future__ import annotations

from pathlib import Path
import sqlite3
import time

from replay.reader import MAGIC, ReplayReader


SCHEMA_SQL = """
create table if not exists chunks (
  path text primary key,
  magic text not null,
  version integer not null,
  compression_flags integer not null,
  sample_count integer not null,
  model_version integer not null,
  generator_version integer not null,
  creation_timestamp_ms integer not null,
  payload_size integer not null,
  checksum integer not null,
  indexed_at_ms integer not null
);

create index if not exists idx_chunks_model_version
  on chunks(model_version);

create index if not exists idx_chunks_creation_timestamp
  on chunks(creation_timestamp_ms);
"""


def init_db(db_path: str | Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA_SQL)


def index_replay_file(db_path: str | Path, chunk_path: str | Path) -> None:
    chunk_path = Path(chunk_path)
    chunk = ReplayReader.read_file(chunk_path)
    header = chunk.header
    indexed_at_ms = int(time.time() * 1000)

    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.execute(
            """
            insert into chunks (
              path,
              magic,
              version,
              compression_flags,
              sample_count,
              model_version,
              generator_version,
              creation_timestamp_ms,
              payload_size,
              checksum,
              indexed_at_ms
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(path) do update set
              magic=excluded.magic,
              version=excluded.version,
              compression_flags=excluded.compression_flags,
              sample_count=excluded.sample_count,
              model_version=excluded.model_version,
              generator_version=excluded.generator_version,
              creation_timestamp_ms=excluded.creation_timestamp_ms,
              payload_size=excluded.payload_size,
              checksum=excluded.checksum,
              indexed_at_ms=excluded.indexed_at_ms
            """,
            (
                str(chunk_path.resolve()),
                MAGIC.decode("ascii"),
                header.version,
                header.compression_flags,
                header.sample_count,
                header.model_version,
                header.generator_version,
                header.creation_timestamp_ms,
                header.payload_size,
                header.checksum,
                indexed_at_ms,
            ),
        )
