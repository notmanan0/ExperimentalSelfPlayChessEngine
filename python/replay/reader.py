from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import zlib


MAGIC = b"CMREPLAY"
CURRENT_VERSION = 1
HEADER_SIZE = 64
HEADER = struct.Struct("<8sHHIIIIIQQI12s")
FIXED_SAMPLE_PREFIX = struct.Struct("<BBBHHBfIQIHH")


class ReplayFormatError(ValueError):
    pass


class ReplayChecksumError(ReplayFormatError):
    pass


class UnsupportedReplayVersion(ReplayFormatError):
    pass


@dataclass(frozen=True)
class ReplayHeader:
    version: int
    header_size: int
    compression_flags: int
    sample_count: int
    model_version: int
    generator_version: int
    creation_timestamp_ms: int
    payload_size: int
    checksum: int


@dataclass(frozen=True)
class ReplayPolicyEntry:
    move: str
    visit_count: int
    probability: float


@dataclass(frozen=True)
class ReplaySample:
    board: tuple[str | None, ...]
    side_to_move: str
    castling_rights: int
    en_passant_square: str | None
    halfmove_clock: int
    fullmove_number: int
    legal_moves: list[str]
    policy: list[ReplayPolicyEntry]
    final_wdl: str
    root_value: float
    search_budget: int
    game_id: int
    ply_index: int


@dataclass(frozen=True)
class ReplayChunk:
    header: ReplayHeader
    samples: list[ReplaySample]


class ReplayReader:
    @classmethod
    def read_file(cls, path: str | Path) -> ReplayChunk:
        return cls.read_bytes(Path(path).read_bytes())

    @classmethod
    def read_bytes(cls, raw: bytes) -> ReplayChunk:
        if len(raw) < HEADER_SIZE:
            raise ReplayFormatError("replay chunk is shorter than the fixed header")

        unpacked = HEADER.unpack_from(raw, 0)
        magic = unpacked[0]
        if magic != MAGIC:
            raise ReplayFormatError("invalid replay chunk magic")

        header = ReplayHeader(
            version=unpacked[1],
            header_size=unpacked[2],
            compression_flags=unpacked[3],
            sample_count=unpacked[4],
            model_version=unpacked[6],
            generator_version=unpacked[7],
            creation_timestamp_ms=unpacked[8],
            payload_size=unpacked[9],
            checksum=unpacked[10],
        )

        if header.version > CURRENT_VERSION:
            raise UnsupportedReplayVersion(
                f"replay version {header.version} is newer than supported version {CURRENT_VERSION}"
            )
        if header.header_size < HEADER_SIZE:
            raise ReplayFormatError("replay header size is smaller than required")
        if header.compression_flags != 0:
            raise ReplayFormatError("compressed replay chunks are reserved for a future version")
        if len(raw) < header.header_size:
            raise ReplayFormatError("replay chunk is shorter than its declared header")

        payload_start = header.header_size
        payload_end = payload_start + header.payload_size
        if payload_end != len(raw):
            raise ReplayFormatError("declared payload size does not match file size")

        payload = raw[payload_start:payload_end]
        actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_checksum != header.checksum:
            raise ReplayChecksumError("replay payload checksum mismatch")

        samples = _decode_samples(payload, header.sample_count)
        return ReplayChunk(header=header, samples=samples)


def _decode_samples(payload: bytes, sample_count: int) -> list[ReplaySample]:
    offset = 0
    samples: list[ReplaySample] = []
    for _ in range(sample_count):
        if offset + 4 > len(payload):
            raise ReplayFormatError("payload ended before sample size")
        sample_size = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        end = offset + sample_size
        if end > len(payload):
            raise ReplayFormatError("sample size exceeds payload")
        samples.append(_decode_sample(payload[offset:end]))
        offset = end

    if offset != len(payload):
        raise ReplayFormatError("payload contains trailing bytes after declared samples")
    return samples


def _decode_sample(sample: bytes) -> ReplaySample:
    required = 64 + FIXED_SAMPLE_PREFIX.size
    if len(sample) < required:
        raise ReplayFormatError("sample is shorter than required fields")

    board = tuple(_decode_piece(code) for code in sample[:64])
    offset = 64
    (
        side,
        castling_rights,
        ep_square,
        halfmove_clock,
        fullmove_number,
        wdl,
        root_value,
        search_budget,
        game_id,
        ply_index,
        legal_count,
        policy_count,
    ) = FIXED_SAMPLE_PREFIX.unpack_from(sample, offset)
    offset += FIXED_SAMPLE_PREFIX.size

    legal_moves = []
    for _ in range(legal_count):
        _require_bytes(sample, offset, 2, "legal move")
        legal_moves.append(_decode_move(struct.unpack_from("<H", sample, offset)[0]))
        offset += 2

    policy = []
    for _ in range(policy_count):
        _require_bytes(sample, offset, 10, "policy entry")
        move = _decode_move(struct.unpack_from("<H", sample, offset)[0])
        visit_count, probability = struct.unpack_from("<If", sample, offset + 2)
        offset += 10
        policy.append(
            ReplayPolicyEntry(
                move=move,
                visit_count=visit_count,
                probability=probability,
            )
        )

    return ReplaySample(
        board=board,
        side_to_move=_decode_side(side),
        castling_rights=castling_rights,
        en_passant_square=None if ep_square == 64 else _square_name(ep_square),
        halfmove_clock=halfmove_clock,
        fullmove_number=fullmove_number,
        legal_moves=legal_moves,
        policy=policy,
        final_wdl=_decode_wdl(wdl),
        root_value=root_value,
        search_budget=search_budget,
        game_id=game_id,
        ply_index=ply_index,
    )


def _require_bytes(raw: bytes, offset: int, size: int, label: str) -> None:
    if offset + size > len(raw):
        raise ReplayFormatError(f"sample ended inside {label}")


def _decode_side(value: int) -> str:
    if value == 0:
        return "white"
    if value == 1:
        return "black"
    raise ReplayFormatError("invalid side-to-move code")


def _decode_wdl(value: int) -> str:
    if value == 0:
        return "black_win"
    if value == 1:
        return "draw"
    if value == 2:
        return "white_win"
    if value == 3:
        return "unknown"
    raise ReplayFormatError("invalid WDL code")


def _decode_piece(value: int) -> str | None:
    pieces = {
        0: None,
        1: "P",
        2: "N",
        3: "B",
        4: "R",
        5: "Q",
        6: "K",
        7: "p",
        8: "n",
        9: "b",
        10: "r",
        11: "q",
        12: "k",
    }
    try:
        return pieces[value]
    except KeyError as exc:
        raise ReplayFormatError("invalid board piece code") from exc


def _decode_move(value: int) -> str:
    from_square = value & 0x3F
    to_square = (value >> 6) & 0x3F
    promotion = (value >> 12) & 0x7
    move = _square_name(from_square) + _square_name(to_square)
    if promotion:
        promotions = {1: "n", 2: "b", 3: "r", 4: "q"}
        try:
            move += promotions[promotion]
        except KeyError as exc:
            raise ReplayFormatError("invalid promotion code") from exc
    return move


def _square_name(index: int) -> str:
    if index < 0 or index >= 64:
        raise ReplayFormatError("invalid square index")
    file = chr(ord("a") + (index & 7))
    rank = str((index >> 3) + 1)
    return file + rank
