"""Tests for scripts/build_opening_book.py."""

import struct
from pathlib import Path

import chess
import chess.polyglot
import numpy as np
import pytest

from scripts.build_opening_book import (
    _REPLAY_DTYPE,
    _SEQ_HEADER,
    _check_elo,
    _check_time_control,
    _count_lines_worker,
    _extract_san_moves,
    _find_boundaries,
    _get_movetext,
    _iter_replay_batches,
    _numpy_aggregate,
    _parse_game_bytes,
    _read_chunk,
    _replay_worker,
    _seq_hash,
    _split_chunk,
    build_book,
    main,
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestFindBoundaries:
    def test_empty_file_returns_empty_list(self, tmp_path):
        p = tmp_path / "empty.pgn"
        p.write_bytes(b"")
        result = _find_boundaries(str(p), 4)
        assert result == []

    def test_single_chunk_covers_full_file(self, tmp_path):
        data = b'[Event "Test"]\n\n1. e4 1-0\n'
        p = tmp_path / "test.pgn"
        p.write_bytes(data)
        result = _find_boundaries(str(p), 1)
        assert result == [(0, len(data))]


class TestReadChunk:
    def test_full_read(self, tmp_path):
        data = b"hello world test data"
        p = tmp_path / "test.bin"
        p.write_bytes(data)
        assert _read_chunk(str(p), 0, len(data)) == data

    def test_partial_read(self, tmp_path):
        data = b"hello world"
        p = tmp_path / "test.bin"
        p.write_bytes(data)
        assert _read_chunk(str(p), 6, 11) == b"world"


class TestSplitChunk:
    def test_single_game(self):
        chunk = b'[Event "Test"]\n\n1. e4 1-0'
        result = _split_chunk(chunk)
        assert len(result) == 1
        assert result[0] == chunk

    def test_multiple_games(self):
        # Separator is b"\n[Event " — the leading \n is consumed, so result[0] has no trailing \n.
        chunk = b'[Event "A"]\n\n1. e4 1-0\n[Event "B"]\n\n1. d4 1-0'
        result = _split_chunk(chunk)
        assert len(result) == 2
        assert result[0] == b'[Event "A"]\n\n1. e4 1-0'
        assert result[1] == b'[Event "B"]\n\n1. d4 1-0'


class TestSeqHash:
    def test_deterministic(self):
        assert _seq_hash(0, "e4") == _seq_hash(0, "e4")

    def test_different_moves_differ(self):
        assert _seq_hash(0, "e4") != _seq_hash(0, "d4")

    def test_chaining_order_matters(self):
        h = _seq_hash(0, "e4")
        assert _seq_hash(h, "e5") != _seq_hash(h, "c5")


class TestCheckEloEdgeCases:
    def test_white_elo_missing_closing_quote(self):
        data = b'[WhiteElo "2000'  # no closing quote
        assert _check_elo(data, 1800) is False

    def test_white_elo_invalid_value(self):
        data = b'[WhiteElo "abc"]\n[BlackElo "2000"]\n\n1. e4 1-0'
        assert _check_elo(data, 1800) is False

    def test_black_elo_missing_header(self):
        data = b'[WhiteElo "2000"]\n\n1. e4 1-0'
        assert _check_elo(data, 1800) is False

    def test_black_elo_missing_closing_quote(self):
        data = b'[WhiteElo "2000"]\n[BlackElo "2000'  # no closing quote for black
        assert _check_elo(data, 1800) is False

    def test_black_elo_invalid_value(self):
        data = b'[WhiteElo "2000"]\n[BlackElo "abc"]\n\n1. e4 1-0'
        assert _check_elo(data, 1800) is False


class TestCheckTimeControlEdgeCases:
    def test_missing_closing_quote(self):
        data = b'[TimeControl "600'  # no closing quote
        assert _check_time_control(data, 180) is False

    def test_invalid_base_with_increment(self):
        data = b'[TimeControl "abc+0"]\n\n1. e4 1-0'
        assert _check_time_control(data, 180) is False

    def test_invalid_base_no_increment(self):
        data = b'[TimeControl "abc"]\n\n1. e4 1-0'
        assert _check_time_control(data, 180) is False


class TestGetMovetext:
    def test_no_double_newline_returns_empty(self):
        data = b'[Event "Test"][Result "1-0"]'
        assert _get_movetext(data) == ""

    def test_with_movetext(self):
        data = b'[Event "Test"]\n\n1. e4 e5 1-0'
        assert _get_movetext(data) == "1. e4 e5 1-0"


class TestParseGameBytesEdgeCases:
    def test_white_elo_invalid_value(self):
        data = b'[WhiteElo "abc"]\n[BlackElo "1600"]\n\n1. e4 1-0'
        we, be, _ = _parse_game_bytes(data)
        assert we == 0
        assert be == 1600

    def test_black_elo_invalid_value(self):
        data = b'[WhiteElo "1500"]\n[BlackElo "xyz"]\n\n1. e4 1-0'
        we, be, _ = _parse_game_bytes(data)
        assert we == 1500
        assert be == 0


class TestExtractSanMovesEdgeCases:
    def test_parenthesized_variation(self):
        moves = _extract_san_moves("1. e4 (1. d4 d5) e5 1-0", 10)
        assert moves == ["e4", "e5"]

    def test_nested_parentheses(self):
        moves = _extract_san_moves("1. e4 (1. d4 (1. c4 c5)) e5 1-0", 10)
        assert moves == ["e4", "e5"]

    def test_spurious_close_brace_produces_empty_token(self):
        # A stray "}" is not caught by the "{" or "(" handlers;
        # the token-extraction loop exits immediately → empty token → skipped.
        moves = _extract_san_moves("1. e4 } e5 1-0", 10)
        assert moves == ["e4", "e5"]

    def test_spurious_close_paren_produces_empty_token(self):
        moves = _extract_san_moves("1. e4 ) e5 1-0", 10)
        assert moves == ["e4", "e5"]


class TestNumpyAggregate:
    def test_empty_array_returned_unchanged(self):
        arr = np.empty(0, dtype=_REPLAY_DTYPE)
        result = _numpy_aggregate(arr)
        assert len(result) == 0

    def test_single_entry_preserved(self):
        arr = np.zeros(1, dtype=_REPLAY_DTYPE)
        arr[0]["pos_hash"] = 12345
        arr[0]["from_sq"] = 8
        arr[0]["to_sq"] = 16
        arr[0]["promo"] = 0
        arr[0]["count"] = 5
        result = _numpy_aggregate(arr)
        assert len(result) == 1
        assert int(result[0]["count"]) == 5

    def test_duplicate_entries_summed(self):
        arr = np.zeros(2, dtype=_REPLAY_DTYPE)
        for i in range(2):
            arr[i]["pos_hash"] = 100
            arr[i]["from_sq"] = 8
            arr[i]["to_sq"] = 16
            arr[i]["promo"] = 0
            arr[i]["count"] = 3
        result = _numpy_aggregate(arr)
        assert len(result) == 1
        assert int(result[0]["count"]) == 6


class TestIterReplayBatches:
    def _write_seq_file(self, path, entries):
        """Write (seq_hash, san_str) pairs to a seq temp file."""
        data = b""
        for h, san in entries:
            san_bytes = san.encode()
            data += _SEQ_HEADER.pack(h, len(san_bytes)) + san_bytes
        Path(path).write_bytes(data)

    def test_batch_fills_and_flushes(self, tmp_path):
        seq_file = tmp_path / "test.seq"
        survivors = {1: 10, 2: 20, 3: 30}
        self._write_seq_file(seq_file, [(1, "e4 e5"), (2, "d4 d5"), (3, "c4 c5")])
        batches = list(_iter_replay_batches(str(seq_file), survivors, batch_size=2))
        # 3 survivors, batch_size=2 → first batch of 2, then batch of 1
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

    def test_skips_non_survivors(self, tmp_path):
        seq_file = tmp_path / "test.seq"
        self._write_seq_file(seq_file, [(1, "e4"), (2, "d4"), (3, "c4")])
        survivors = {2: 5}  # only seq 2 survives
        batches = list(_iter_replay_batches(str(seq_file), survivors))
        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0][0] == 2


class TestReplayWorker:
    def test_basic_two_moves(self):
        result = _replay_worker([(0, 1, "e4 e5")])
        # 2 distinct position-move pairs → 2 entries × 16 bytes
        assert len(result) == 2 * 16

    def test_illegal_move_skipped(self):
        result = _replay_worker([(0, 1, "invalid_move_xyz")])
        assert len(result) == 0

    def test_weight_accumulation(self):
        # Same board, same move from two separate items → counts add
        result = _replay_worker([(0, 3, "e4"), (1, 3, "e4")])
        # Both start from chess.Board() → same pos_hash, same move → 1 unique entry
        assert len(result) == 1 * 16


class TestCountLinesWorker:
    def test_basic_game_counted(self, tmp_path):
        pgn = (
            b'[Event "A"]\n[WhiteElo "2000"]\n'
            b'[BlackElo "2100"]\n\n1. e4 e5 2. Nf3 1-0\n'
        )
        p = tmp_path / "test.pgn"
        p.write_bytes(pgn)
        data, matched, scanned, sequences, fh_data = _count_lines_worker(
            (str(p), 0, len(pgn), 0, 0)
        )
        assert scanned == 1
        assert matched == 1
        assert len(sequences) == 1

    def test_elo_filter_rejects_low_elo(self, tmp_path):
        pgn = b'[Event "A"]\n[WhiteElo "1500"]\n[BlackElo "1500"]\n\n1. e4 1-0\n'
        p = tmp_path / "test.pgn"
        p.write_bytes(pgn)
        _, matched, scanned, _, _ = _count_lines_worker((str(p), 0, len(pgn), 2000, 0))
        assert scanned == 1
        assert matched == 0

    def test_short_game_data_skipped(self, tmp_path):
        pgn = b'[Event "X"]\n'  # < 20 bytes → skipped before scanned increment
        p = tmp_path / "test.pgn"
        p.write_bytes(pgn)
        _, matched, scanned, _, _ = _count_lines_worker((str(p), 0, len(pgn), 0, 0))
        assert scanned == 0
        assert matched == 0


class TestWriteBookPromotion:
    def test_promotion_uci_constructed(self, tmp_path):
        """Illegal promotion move hits UCI construction (line 641) then exception handler."""
        start_hash = chess.polyglot.zobrist_hash(chess.Board())
        move_counts = {
            (start_hash, chess.E2, chess.E4, 0): 100,
            (start_hash, chess.E2, chess.E8, 5): 200,  # illegal queen promo from start
        }
        output = tmp_path / "book.bin"
        write_book(move_counts, output)
        with open(output, "rb") as f:
            assert f.read(4) == b"BOOK"
            _, num_pos, num_moves = struct.unpack("<III", f.read(12))
        assert num_pos == 1
        assert num_moves == 2  # both moves kept in pos_table despite illegal dest

    def test_most_common_line_exits_book(self, tmp_path):
        """When best move dest is not in dest_map, the line-tracing loop breaks (line 697)."""
        start_hash = chess.polyglot.zobrist_hash(chess.Board())
        # Illegal move has higher count → higher weight → picked as "best" in line trace
        move_counts = {
            (start_hash, chess.E2, chess.E8, 5): 200,  # illegal, weight=255
            (start_hash, chess.E2, chess.E4, 0): 100,  # legal, weight=128
        }
        output = tmp_path / "book.bin"
        write_book(move_counts, output)  # must not raise
        assert output.exists()


class TestBuildBookEdgeCases:
    def test_empty_pgn_file_returns_empty_book(self, tmp_path):
        p = tmp_path / "empty.pgn"
        p.write_bytes(b"")
        move_counts = build_book([str(p)], 0, 0, 1, 1)
        assert move_counts == {}

    def test_all_duplicate_games_filtered(self, tmp_path):
        """All games share identical sequence → same final pos → all filtered out."""
        pgn = """\
[Event "Test"]
[Result "1-0"]

1. e4 e5 1-0

[Event "Test"]
[Result "1-0"]

1. e4 e5 1-0

"""
        p = _make_pgn(pgn, tmp_path)
        move_counts = build_book([str(p)], 0, 0, 1, 1)
        assert move_counts == {}


class TestMain:
    def test_main_creates_book_file(self, tmp_path, monkeypatch):
        pgn_path = _make_pgn(SAMPLE_PGN, tmp_path)
        out_path = tmp_path / "book.bin"
        monkeypatch.setattr(
            "sys.argv",
            ["build_opening_book.py", str(pgn_path), "-o", str(out_path)],
        )
        main()
        assert out_path.exists()
        assert out_path.read_bytes()[:4] == b"BOOK"

    def test_main_missing_file_exits_with_error(self, tmp_path, monkeypatch):
        missing = tmp_path / "nonexistent.pgn"
        out_path = tmp_path / "book.bin"
        monkeypatch.setattr(
            "sys.argv",
            ["build_opening_book.py", str(missing), "-o", str(out_path)],
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
