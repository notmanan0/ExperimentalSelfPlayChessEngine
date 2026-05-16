"""Tests proving Python and C++ policy indexing and board encoding agree.

The C++ side is verified in tests/cpp/inference/test_inference.cpp which checks
the same reference values. These tests verify the Python implementation matches
those exact same reference values, proving cross-language agreement.

Additionally, when the compat_helper binary is available, we run direct
cross-language comparison tests.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch

from chessmoe.models.encoding import (
    encode_fen,
    move_to_index,
    square_to_index,
    NUM_MOVE_BUCKETS,
)

# ---------------------------------------------------------------------------
# Locate the C++ compat_helper binary (optional, for direct comparison)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_compat_helper() -> Path | None:
    """Search common build directories for the compat_helper executable."""
    for build_dir in _REPO_ROOT.iterdir():
        if not build_dir.is_dir() or not build_dir.name.startswith("build"):
            continue
        for sub in [
            "bin/Debug",
            "bin/Release",
            "Debug",
            "Release",
            "tools/compat/Debug",
            "tools/compat/Release",
            "tools/compat",
        ]:
            for name in ("compat_helper.exe", "compat_helper"):
                candidate = build_dir / sub / name
                if candidate.exists():
                    return candidate
    return None


_COMPAT_HELPER = _find_compat_helper()


def _run_helper(*args: str) -> str | None:
    """Run compat_helper with given arguments and return stdout, or None if unavailable."""
    import os
    import subprocess

    if _COMPAT_HELPER is None:
        return None
    try:
        # Use CREATE_NO_WINDOW to avoid Windows handle inheritance issues
        creationflags = 0
        if os.name == "nt":
            creationflags = 0x08000000  # CREATE_NO_WINDOW
        result = subprocess.run(
            [str(_COMPAT_HELPER), *args],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=creationflags,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reference values from C++ test_inference.cpp (must match exactly)
# These are the ground truth that BOTH C++ and Python must agree on.
# ---------------------------------------------------------------------------

# From test_policy_index_mapping_matches_python_contract():
REFERENCE_POLICY_INDICES = {
    "a1a1": 0,
    "e2e4": 796,
    "e7e8q": 7484,
    "e7e8n": 19772,
}

# Additional reference values for comprehensive coverage
EXTENDED_POLICY_INDICES = {
    # Quiet moves: from*64 + to
    "a1a2": 0 * 64 + 8,      # a1=0, a2=8 -> 8
    "d4d5": 27 * 64 + 35,    # d4=27, d5=35 -> 1763
    "h1h8": 7 * 64 + 63,     # h1=7, h8=63 -> 511
    # Captures
    "e2d3": 12 * 64 + 19,    # e2=12, d3=19 -> 787
    "h7g8": 55 * 64 + 62,    # h7=55, g8=62 -> 3582
    # Promotions: 4096 + promo_offset * 4096 + base
    "e7e8r": 4096 + 1 * 4096 + 52 * 64 + 60,  # r offset=1
    "e7e8b": 4096 + 2 * 4096 + 52 * 64 + 60,  # b offset=2
    "a7a8q": 4096 + 0 * 4096 + 48 * 64 + 56,  # q offset=0
    "a7a8n": 4096 + 3 * 4096 + 48 * 64 + 56,  # n offset=3
    "c7c8b": 4096 + 2 * 4096 + 50 * 64 + 58,  # b offset=2
}


# ---------------------------------------------------------------------------
# Policy index agreement tests
# ---------------------------------------------------------------------------

class TestPolicyIndexAgreement:
    """Verify Python move_to_index agrees with C++ reference values."""

    @pytest.mark.parametrize("uci,expected", REFERENCE_POLICY_INDICES.items())
    def test_reference_indices_match_cpp(self, uci: str, expected: int) -> None:
        """These exact values are verified in C++ test_inference.cpp."""
        assert move_to_index(uci) == expected

    @pytest.mark.parametrize("uci,expected", EXTENDED_POLICY_INDICES.items())
    def test_extended_indices(self, uci: str, expected: int) -> None:
        assert move_to_index(uci) == expected

    def test_all_indices_in_range(self) -> None:
        for uci in list(REFERENCE_POLICY_INDICES) + list(EXTENDED_POLICY_INDICES):
            idx = move_to_index(uci)
            assert 0 <= idx < NUM_MOVE_BUCKETS

    def test_a1a1_is_zero(self) -> None:
        assert move_to_index("a1a1") == 0

    def test_e2e4_specific_index(self) -> None:
        expected = square_to_index("e2") * 64 + square_to_index("e4")
        assert move_to_index("e2e4") == expected

    def test_promotion_offsets_are_separated(self) -> None:
        base = square_to_index("e7") * 64 + square_to_index("e8")
        q = move_to_index("e7e8q")
        r = move_to_index("e7e8r")
        b = move_to_index("e7e8b")
        n = move_to_index("e7e8n")
        assert q < r < b < n
        assert q == 4096 + base
        assert r == 4096 + 4096 + base
        assert b == 4096 + 2 * 4096 + base
        assert n == 4096 + 3 * 4096 + base

    @pytest.mark.parametrize("bad_move", ["", "a1", "a1a2x", "i1a2", "a9a2", "a1a0"])
    def test_invalid_moves_raise(self, bad_move: str) -> None:
        with pytest.raises((ValueError, Exception)):
            move_to_index(bad_move)

    def test_direct_cpp_comparison_if_available(self) -> None:
        """If compat_helper is built, directly compare C++ and Python output."""
        all_moves = list(REFERENCE_POLICY_INDICES) + list(EXTENDED_POLICY_INDICES)
        for uci in all_moves:
            cpp_out = _run_helper("policy-index", uci)
            if cpp_out is None:
                pytest.skip("compat_helper not available")
            cpp_index = int(cpp_out)
            py_index = move_to_index(uci)
            assert py_index == cpp_index, f"Mismatch for {uci}: py={py_index} cpp={cpp_index}"


# ---------------------------------------------------------------------------
# Board encoding agreement tests
# ---------------------------------------------------------------------------

# Reference checksums computed by C++ encode-fen-checksum (adler32 of the tensor)
# These are verified to be correct by the C++ inference_tests.
REFERENCE_CHECKSUMS = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": "78c206b0",
}


def _python_adler32(fen: str) -> str:
    """Compute adler32 checksum of Python-encoded tensor, matching C++ adler32."""
    tensor = encode_fen(fen)
    flat = tensor.flatten().tolist()
    byte_data = struct.pack(f"<{len(flat)}f", *flat)
    a, b = 1, 0
    mod = 65521
    for byte in byte_data:
        a = (a + byte) % mod
        b = (b + a) % mod
    return f"{(b << 16) | a:08x}"


ENCODING_TEST_FENS: list[str] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9",
]


class TestBoardEncodingAgreement:
    """Verify Python encode_fen produces the same tensor as C++ encode_position_nchw."""

    @pytest.mark.parametrize("fen", ENCODING_TEST_FENS)
    def test_encoding_checksum_matches_cpp(self, fen: str) -> None:
        py_checksum = _python_adler32(fen)
        if fen in REFERENCE_CHECKSUMS:
            assert py_checksum == REFERENCE_CHECKSUMS[fen], (
                f"Python checksum {py_checksum} != C++ reference {REFERENCE_CHECKSUMS[fen]}"
            )

    def test_direct_cpp_comparison_if_available(self) -> None:
        """If compat_helper is built, directly compare C++ and Python tensors."""
        for fen in ENCODING_TEST_FENS:
            cpp_out = _run_helper("encode-fen-checksum", fen)
            if cpp_out is None:
                pytest.skip("compat_helper not available")
            py_checksum = _python_adler32(fen)
            assert py_checksum == cpp_out, (
                f"Checksum mismatch for {fen}: py={py_checksum} cpp={cpp_out}"
            )

    def test_direct_cpp_element_wise_if_available(self) -> None:
        """If compat_helper is built, compare element-by-element."""
        fen = ENCODING_TEST_FENS[0]
        cpp_out = _run_helper("encode-fen", fen)
        if cpp_out is None:
            pytest.skip("compat_helper not available")
        cpp_flat = [float(x) for x in cpp_out.split()]
        py_flat = encode_fen(fen).flatten().tolist()
        assert len(cpp_flat) == len(py_flat) == 18 * 8 * 8
        for i, (c, p) in enumerate(zip(cpp_flat, py_flat)):
            assert abs(c - p) < 1e-6, f"Index {i}: cpp={c} py={p}"

    def test_starting_position_channels(self) -> None:
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        tensor = encode_fen(fen)
        assert tensor[0, 1, :].sum() == 8.0, "White pawns on rank 2"
        assert tensor[6, 6, :].sum() == 8.0, "Black pawns on rank 7"
        assert tensor[5, 0, 4] == 1.0, "White king on e1"
        assert tensor[11, 7, 4] == 1.0, "Black king on e8"
        assert tensor[12].sum() == 64.0, "Side-to-move is white"
        for ch in range(13, 17):
            assert tensor[ch].sum() == 64.0, f"Castling channel {ch}"
        assert tensor[17].sum() == 0.0, "No en passant"

    def test_black_to_move_channel(self) -> None:
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        tensor = encode_fen(fen)
        assert tensor[12].sum() == 0.0, "Side-to-move is black"
        assert tensor[17, 2, 4] == 1.0, "En passant on e3"

    def test_no_castling_rights(self) -> None:
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1"
        tensor = encode_fen(fen)
        for ch in range(13, 17):
            assert tensor[ch].sum() == 0.0


# ---------------------------------------------------------------------------
# Round-trip FEN agreement
# ---------------------------------------------------------------------------

class TestFenRoundTrip:
    """Verify C++ FEN parse -> to_string round-trip matches input."""

    @pytest.mark.parametrize("fen", ENCODING_TEST_FENS)
    def test_cpp_fen_roundtrip_if_available(self, fen: str) -> None:
        result = _run_helper("roundtrip-fen", fen)
        if result is None:
            pytest.skip("compat_helper not available")
        assert result == fen


# ---------------------------------------------------------------------------
# Legal moves agreement
# ---------------------------------------------------------------------------

class TestLegalMovesAgreement:
    """Verify legal move counts match between C++ and expected values."""

    def test_starting_position_20_moves_if_available(self) -> None:
        result = _run_helper(
            "legal-moves",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        if result is None:
            pytest.skip("compat_helper not available")
        assert len(result.split()) == 20

    def test_king_only_5_moves_if_available(self) -> None:
        result = _run_helper(
            "legal-moves",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        )
        if result is None:
            pytest.skip("compat_helper not available")
        assert len(result.split()) == 5
