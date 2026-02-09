"""Tests for scripts/fix_training_data.py."""

import struct

from scripts.fix_training_data import (
    EVAL_OFFSET,
    POSITION_SIZE,
    SIDE_TO_MOVE_OFFSET,
    fix,
)


def _make_position(side_to_move, search_eval):
    """Build a minimal 42-byte position with given STM and eval."""
    buf = bytearray(POSITION_SIZE)
    buf[SIDE_TO_MOVE_OFFSET] = side_to_move
    struct.pack_into("<f", buf, EVAL_OFFSET, search_eval)
    return bytes(buf)


def _read_eval(data, index):
    """Read search eval float from position at given index."""
    offset = index * POSITION_SIZE + EVAL_OFFSET
    return struct.unpack_from("<f", data, offset)[0]


def test_fix_negates_black_positions(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(_make_position(1, 3.0))
    fix(str(path))
    assert _read_eval(path.read_bytes(), 0) == -3.0


def test_fix_preserves_white_positions(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(_make_position(0, 5.0))
    fix(str(path))
    assert _read_eval(path.read_bytes(), 0) == 5.0


def test_fix_mixed_positions(tmp_path):
    path = tmp_path / "data.bin"
    data = _make_position(0, 2.0) + _make_position(1, 4.0) + _make_position(0, -1.0)
    path.write_bytes(data)
    fix(str(path))
    result = path.read_bytes()
    assert _read_eval(result, 0) == 2.0
    assert _read_eval(result, 1) == -4.0
    assert _read_eval(result, 2) == -1.0


def test_fix_empty_file(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"")
    fix(str(path))
    assert path.read_bytes() == b""


def test_fix_zero_eval_black(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(_make_position(1, 0.0))
    fix(str(path))
    assert _read_eval(path.read_bytes(), 0) == 0.0


def test_fix_negative_eval_black(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(_make_position(1, -7.5))
    fix(str(path))
    assert _read_eval(path.read_bytes(), 0) == 7.5


def test_fix_prints_summary(tmp_path, capsys):
    path = tmp_path / "data.bin"
    data = _make_position(0, 1.0) + _make_position(1, 2.0) + _make_position(1, 3.0)
    path.write_bytes(data)
    fix(str(path))
    output = capsys.readouterr().out
    assert "2/3" in output
    assert "66.7%" in output


def test_fix_all_white_prints_zero(tmp_path, capsys):
    path = tmp_path / "data.bin"
    data = _make_position(0, 1.0) + _make_position(0, 2.0)
    path.write_bytes(data)
    fix(str(path))
    output = capsys.readouterr().out
    assert "0/2" in output


def test_fix_file_size_unchanged(tmp_path):
    path = tmp_path / "data.bin"
    data = _make_position(0, 1.0) + _make_position(1, 2.0)
    path.write_bytes(data)
    original_size = path.stat().st_size
    fix(str(path))
    assert path.stat().st_size == original_size


def test_fix_non_stm_bytes_preserved(tmp_path):
    """Fix should only modify eval bytes, not piece placement or other fields."""
    buf = bytearray(POSITION_SIZE)
    buf[0] = 0xAB  # piece placement byte
    buf[SIDE_TO_MOVE_OFFSET] = 1
    buf[33] = 0x0F  # castling
    buf[34] = 4  # en passant file
    struct.pack_into("<f", buf, EVAL_OFFSET, 5.0)
    buf[39] = 2  # game result
    struct.pack_into("<H", buf, 40, 99)  # ply

    path = tmp_path / "data.bin"
    path.write_bytes(bytes(buf))
    fix(str(path))
    result = path.read_bytes()

    assert result[0] == 0xAB
    assert result[33] == 0x0F
    assert result[34] == 4
    assert _read_eval(result, 0) == -5.0
    assert result[39] == 2
    assert struct.unpack_from("<H", result, 40)[0] == 99
