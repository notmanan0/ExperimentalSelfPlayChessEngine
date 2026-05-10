# Phase 7 Replay Chunk Format

## Objective

Phase 7 stores self-play output as compact binary replay chunks with fixed metadata and checksummed payloads. PGN is not used as the main training data format.

## File Extension

Use `.cmrep` for replay chunks.

## Byte Order

All integer and floating-point fields are little-endian. Floating-point values are IEEE-754 `float32`.

## Chunk Header

The fixed header is 64 bytes.

| Offset | Size | Field | Type | Notes |
| - | -: | - | - | - |
| 0 | 8 | magic | bytes | ASCII `CMREPLAY` |
| 8 | 2 | version | uint16 | Current version is `1` |
| 10 | 2 | header_size | uint16 | Current size is `64` |
| 12 | 4 | compression_flags | uint32 | `0` means uncompressed; other values are reserved |
| 16 | 4 | sample_count | uint32 | Number of samples in payload |
| 20 | 4 | reserved | uint32 | Must be zero in version `1` |
| 24 | 4 | model_version | uint32 | Numeric model identifier |
| 28 | 4 | generator_version | uint32 | Numeric self-play generator identifier |
| 32 | 8 | creation_timestamp_ms | uint64 | Unix epoch timestamp in milliseconds |
| 40 | 8 | payload_size | uint64 | Bytes after the header |
| 48 | 4 | checksum | uint32 | CRC32 of the payload bytes |
| 52 | 12 | reserved | bytes | Must be zero in version `1` |

## Sample Payload

Each sample starts with `sample_size: uint32`, followed by exactly that many bytes. Readers may skip trailing bytes inside a sample when future versions append fields.

| Field | Type | Notes |
| - | - | - |
| board | `uint8[64]` | A1 through H8. `0` empty, `1..6` white PNBRQK, `7..12` black pnbrqk |
| side_to_move | uint8 | `0` white, `1` black |
| castling_rights | uint8 | Bitmask: white king, white queen, black king, black queen |
| en_passant_square | uint8 | `0..63` or `64` for none |
| halfmove_clock | uint16 | FEN halfmove clock |
| fullmove_number | uint16 | FEN fullmove number |
| final_wdl | uint8 | `0` black win, `1` draw, `2` white win, `3` unknown |
| root_value | float32 | Search root value |
| search_budget | uint32 | Visits or equivalent search budget |
| game_id | uint64 | Self-play game identifier |
| ply_index | uint32 | Ply index in game |
| legal_count | uint16 | Number of legal moves |
| policy_count | uint16 | Number of policy entries |
| legal_moves | `uint16[legal_count]` | Packed moves |
| policy | repeated | `move:uint16`, `visit_count:uint32`, `probability:float32` |

## Packed Move

Packed moves use 16 bits:

- bits `0..5`: from-square.
- bits `6..11`: to-square.
- bits `12..14`: promotion, where `0` none, `1` knight, `2` bishop, `3` rook, `4` queen.
- bit `15`: reserved and must be zero in version `1`.

## Metadata Index

The SQLite index stores one row per chunk in `chunks`:

```sql
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
```

## Forward Compatibility

Readers reject chunks with a version newer than they support. Existing sample fields stay in the same order. Future versions can append fields inside each sample because `sample_size` lets older compatible tooling skip unknown trailing bytes.
