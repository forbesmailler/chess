"""Tests for scripts/build_opening_book.py."""

import struct
from pathlib import Path

import chess
import chess.polyglot

from scripts.build_opening_book import (
    _extract_san_moves,
    _parse_game_bytes,
    build_book,
    find_optimal_depth,
    write_book,
)


def _make_pgn(games_text: str, tmp_path: Path) -> Path:
    """Write PGN text to a temp file and return the path."""
    p = tmp_path / "test.pgn"
    p.write_text(games_text, encoding="utf-8")
    return p


SAMPLE_PGN = """\
[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 1-0

[Event "Test"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nf6 0-1

[Event "Test"]
[Result "1/2-1/2"]

1. d4 d5 2. c4 e6 1/2-1/2

[Event "Test"]
[Result "1-0"]

1. e4 c5 1-0

"""


class TestParseGameBytes:
    def test_elo_parsing(self):
        data = b'[WhiteElo "1500"]\n[BlackElo "1600"]\n\n1. e4 e5 1-0'
        we, be, mt = _parse_game_bytes(data)
        assert we == 1500
        assert be == 1600
        assert "e4" in mt

    def test_missing_elo(self):
        data = b'[Event "Test"]\n\n1. e4 1-0'
        we, be, mt = _parse_game_bytes(data)
        assert we == 0
        assert be == 0

    def test_no_movetext(self):
        we, be, mt = _parse_game_bytes(b'[Event "Test"]')
        assert mt == ""


class TestExtractSanMoves:
    def test_basic(self):
        moves = _extract_san_moves("1. e4 e5 2. Nf3 Nc6 1-0", 10)
        assert moves == ["e4", "e5", "Nf3", "Nc6"]

    def test_max_depth(self):
        moves = _extract_san_moves("1. e4 e5 2. Nf3 Nc6 1-0", 2)
        assert moves == ["e4", "e5"]

    def test_with_comments(self):
        moves = _extract_san_moves(
            "1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 1-0", 10
        )
        assert moves == ["e4", "e5"]

    def test_result_only(self):
        moves = _extract_san_moves("*", 10)
        assert moves == []

    def test_with_nags(self):
        moves = _extract_san_moves("1. e4 $1 e5 $2 1-0", 10)
        assert moves == ["e4", "e5"]


class TestFindOptimalDepth:
    def test_basic(self, tmp_path):
        pgn_path = _make_pgn(SAMPLE_PGN, tmp_path)
        depth, total = find_optimal_depth([str(pgn_path)], 250000, 0.5, 30, 0, 1)
        assert depth >= 1
        assert total == 4

    def test_max_depth_limit(self, tmp_path):
        pgn_path = _make_pgn(SAMPLE_PGN, tmp_path)
        depth, total = find_optimal_depth([str(pgn_path)], 250000, 0.5, 2, 0, 1)
        assert depth <= 2


class TestBuildBook:
    def test_basic(self, tmp_path):
        pgn_path = _make_pgn(SAMPLE_PGN, tmp_path)
        move_counts = build_book([str(pgn_path)], 2, 0, 1)

        # Starting position should have entries
        start_hash = chess.polyglot.zobrist_hash(chess.Board())
        start_moves = {k: v for k, v in move_counts.items() if k[0] == start_hash}
        assert len(start_moves) > 0

        # e4 appears in 3 games, d4 in 1
        e4_key = (start_hash, chess.E2, chess.E4, 0)
        d4_key = (start_hash, chess.D2, chess.D4, 0)
        assert move_counts[e4_key] == 3
        assert move_counts[d4_key] == 1

    def test_depth_1(self, tmp_path):
        pgn_path = _make_pgn(SAMPLE_PGN, tmp_path)
        move_counts = build_book([str(pgn_path)], 1, 0, 1)
        # Only first ply: e4 (3 games) and d4 (1 game)
        assert len(move_counts) == 2


class TestWriteBook:
    def test_roundtrip(self, tmp_path):
        start_hash = chess.polyglot.zobrist_hash(chess.Board())
        move_counts = {
            (start_hash, chess.E2, chess.E4, 0): 100,
            (start_hash, chess.D2, chess.D4, 0): 50,
        }

        output = tmp_path / "test.bin"
        write_book(move_counts, output)

        with open(output, "rb") as f:
            magic = f.read(4)
            assert magic == b"BOOK"

            version, num_pos, num_moves = struct.unpack("<III", f.read(12))
            assert version == 1
            assert num_pos == 1
            assert num_moves == 2

            # Read position
            pos_hash, offset, count = struct.unpack("<QIH", f.read(14))
            _reserved = f.read(2)
            assert pos_hash == start_hash
            assert offset == 0
            assert count == 2

            # Read moves
            moves = []
            for _ in range(num_moves):
                from_sq, to_sq, promo, weight = struct.unpack("<BBBB", f.read(4))
                moves.append((from_sq, to_sq, promo, weight))

            # e4 should have weight 255 (higher count), d4 lower
            e4_move = next(m for m in moves if m[0] == chess.E2 and m[1] == chess.E4)
            d4_move = next(m for m in moves if m[0] == chess.D2 and m[1] == chess.D4)
            assert e4_move[3] == 255
            assert d4_move[3] == 128  # round(50/100 * 255) = 128

    def test_multiple_positions(self, tmp_path):
        board1 = chess.Board()
        h1 = chess.polyglot.zobrist_hash(board1)

        board2 = chess.Board()
        board2.push_uci("e2e4")
        h2 = chess.polyglot.zobrist_hash(board2)

        move_counts = {
            (h1, chess.E2, chess.E4, 0): 100,
            (h2, chess.E7, chess.E5, 0): 80,
        }

        output = tmp_path / "test.bin"
        write_book(move_counts, output)

        with open(output, "rb") as f:
            f.read(4)  # magic
            _, num_pos, num_moves = struct.unpack("<III", f.read(12))
            assert num_pos == 2
            assert num_moves == 2

            # Positions should be sorted by hash
            hashes = []
            for _ in range(num_pos):
                pos_hash = struct.unpack("<Q", f.read(8))[0]
                f.read(8)  # offset + count + reserved
                hashes.append(pos_hash)
            assert hashes == sorted(hashes)
