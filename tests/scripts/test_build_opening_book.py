"""Tests for scripts/build_opening_book.py."""

import struct
from pathlib import Path

import chess
import chess.polyglot

from scripts.build_opening_book import (
    _check_elo,
    _check_time_control,
    _extract_san_moves,
    _parse_game_bytes,
    build_book,
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


class TestCheckElo:
    def test_both_above(self):
        data = b'[WhiteElo "2300"]\n[BlackElo "2400"]\n\n1. e4 1-0'
        assert _check_elo(data, 2200) is True

    def test_white_below(self):
        data = b'[WhiteElo "2100"]\n[BlackElo "2400"]\n\n1. e4 1-0'
        assert _check_elo(data, 2200) is False

    def test_black_below(self):
        data = b'[WhiteElo "2300"]\n[BlackElo "2100"]\n\n1. e4 1-0'
        assert _check_elo(data, 2200) is False

    def test_missing_elo(self):
        data = b'[Event "Test"]\n\n1. e4 1-0'
        assert _check_elo(data, 2200) is False


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


class TestCheckTimeControl:
    def test_rapid(self):
        data = b'[TimeControl "600+0"]\n\n1. e4 1-0'
        assert _check_time_control(data, 180) is True

    def test_bullet(self):
        data = b'[TimeControl "60+0"]\n\n1. e4 1-0'
        assert _check_time_control(data, 180) is False

    def test_blitz_boundary(self):
        data = b'[TimeControl "180+0"]\n\n1. e4 1-0'
        assert _check_time_control(data, 180) is True

    def test_no_increment(self):
        data = b'[TimeControl "300"]\n\n1. e4 1-0'
        assert _check_time_control(data, 180) is True

    def test_missing(self):
        data = b'[Event "Test"]\n\n1. e4 1-0'
        assert _check_time_control(data, 180) is False

    def test_zero_min(self):
        data = b'[TimeControl "60+0"]\n\n1. e4 1-0'
        assert _check_time_control(data, 0) is True


class TestBuildBook:
    def test_basic(self, tmp_path):
        pgn_path = _make_pgn(SAMPLE_PGN, tmp_path)
        move_counts = build_book([str(pgn_path)], 0, 0, 1, 1)

        # Starting position should have entries
        start_hash = chess.polyglot.zobrist_hash(chess.Board())
        start_moves = {k: v for k, v in move_counts.items() if k[0] == start_hash}
        assert len(start_moves) > 0

        # e4 appears in 3 games (2× "e4 e5" + 1× "e4 c5"), d4 in 1
        e4_key = (start_hash, chess.E2, chess.E4, 0)
        d4_key = (start_hash, chess.D2, chess.D4, 0)
        assert move_counts[e4_key] == 3
        assert move_counts[d4_key] == 1

    def test_all_plies_recorded(self, tmp_path):
        """All plies from all games are recorded (depth-agnostic)."""
        pgn_path = _make_pgn(SAMPLE_PGN, tmp_path)
        move_counts = build_book([str(pgn_path)], 0, 0, 1, 1)

        start_hash = chess.polyglot.zobrist_hash(chess.Board())
        e4_key = (start_hash, chess.E2, chess.E4, 0)
        # All 3 e4 games contribute (even "e4 c5" which is only 2 plies)
        assert move_counts[e4_key] == 3

        # Bb5 from game 1 (depth 5) should also be recorded
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6"]:
            board.push_uci(uci)
        h4 = chess.polyglot.zobrist_hash(board)
        bb5_key = (h4, chess.F1, chess.B5, 0)
        assert move_counts[bb5_key] == 1

    def test_non_unique_final_position_filtered(self, tmp_path):
        """Any game whose final position appears in another game is removed."""
        # Two identical games (same final position) + one unique game
        pgn = """\
[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Test"]
[Result "0-1"]

1. d4 d5 0-1

"""
        pgn_path = _make_pgn(pgn, tmp_path)
        move_counts = build_book([str(pgn_path)], 0, 0, 1, 1)

        # The two identical e4 games share a final position -> both removed
        # Only d4 d5 game survives
        start_hash = chess.polyglot.zobrist_hash(chess.Board())
        d4_key = (start_hash, chess.D2, chess.D4, 0)
        assert d4_key in move_counts

        e4_key = (start_hash, chess.E2, chess.E4, 0)
        assert e4_key not in move_counts


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

    def test_unreachable_positions_pruned(self, tmp_path):
        """Positions not reachable from starting position are pruned."""
        start_hash = chess.polyglot.zobrist_hash(chess.Board())

        # Create a position reachable from start (after e4)
        board_e4 = chess.Board()
        board_e4.push_uci("e2e4")
        h_e4 = chess.polyglot.zobrist_hash(board_e4)

        # Create an unreachable position (random hash)
        h_orphan = 0xDEADBEEF

        move_counts = {
            (start_hash, chess.E2, chess.E4, 0): 100,
            (h_e4, chess.E7, chess.E5, 0): 80,
            (h_orphan, chess.A2, chess.A3, 0): 50,
        }

        output = tmp_path / "test.bin"
        write_book(move_counts, output)

        with open(output, "rb") as f:
            f.read(4)  # magic
            _, num_pos, num_moves = struct.unpack("<III", f.read(12))
            # Orphan position should be pruned
            assert num_pos == 2
            assert num_moves == 2

    def test_min_count_filters(self, tmp_path):
        """Moves below min_count are filtered out before weight computation."""
        start_hash = chess.polyglot.zobrist_hash(chess.Board())
        move_counts = {
            (start_hash, chess.E2, chess.E4, 0): 100,
            (start_hash, chess.D2, chess.D4, 0): 5,  # below min_count=10
        }

        output = tmp_path / "test.bin"
        write_book(move_counts, output, min_count=10)

        with open(output, "rb") as f:
            f.read(4)  # magic
            _, num_pos, num_moves = struct.unpack("<III", f.read(12))
            assert num_pos == 1
            assert num_moves == 1
