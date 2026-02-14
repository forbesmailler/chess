"""Tests for scripts/gen_config_header.py."""

import re

import pytest

import scripts.gen_config_header as gen_config_header
from scripts.gen_config_header import (
    flatten_pst,
    fmt_int_array,
    fmt_pst,
    generate,
    load,
)


class TestFlattenPst:
    def test_basic(self):
        rows = [[1, 2], [3, 4]]
        assert flatten_pst(rows) == [1, 2, 3, 4]

    def test_8x8(self):
        rows = [list(range(i * 8, (i + 1) * 8)) for i in range(8)]
        result = flatten_pst(rows)
        assert len(result) == 64
        assert result == list(range(64))

    def test_empty(self):
        assert flatten_pst([]) == []

    def test_single_row(self):
        assert flatten_pst([[10, 20, 30]]) == [10, 20, 30]


class TestFmtIntArray:
    def test_basic(self):
        result = fmt_int_array([1, 2, 3], "FOO", "    ")
        assert result == "    static constexpr int FOO[] = {1, 2, 3};\n"

    def test_negative_values(self):
        result = fmt_int_array([-10, 0, 10], "BAR", "")
        assert result == "static constexpr int BAR[] = {-10, 0, 10};\n"

    def test_single_element(self):
        result = fmt_int_array([42], "X", "  ")
        assert result == "  static constexpr int X[] = {42};\n"


class TestFmtPst:
    def test_64_elements(self):
        values = list(range(64))
        result = fmt_pst(values, "TEST_PST", "    ")
        assert "TEST_PST[64]" in result
        assert result.startswith("    static constexpr int TEST_PST[64] = {")
        assert result.endswith("};\n")
        # 8 rows of values
        lines = result.strip().split("\n")
        assert len(lines) == 10  # header + 8 rows + closing brace

    def test_values_present(self):
        values = [0] * 63 + [99]
        result = fmt_pst(values, "PST", "")
        assert "  99" in result

    def test_comma_separation(self):
        values = list(range(64))
        result = fmt_pst(values, "PST", "")
        # First 7 rows should end with comma, last should not
        lines = result.strip().split("\n")
        for line in lines[1:8]:  # rows 0-6
            assert line.rstrip().endswith(",")
        assert not lines[8].rstrip().endswith(",")  # row 7


class TestGenerate:
    def test_returns_string(self):
        result = generate()
        assert isinstance(result, str)

    def test_header_guard(self):
        result = generate()
        assert "#pragma once" in result

    def test_auto_generated_comment(self):
        result = generate()
        assert "Auto-generated" in result
        assert "DO NOT EDIT" in result

    def test_includes(self):
        result = generate()
        assert "#include <cstddef>" in result
        assert "#include <string_view>" in result

    def test_namespace_config(self):
        result = generate()
        assert "namespace config {" in result
        assert "}  // namespace config" in result

    def test_mate_value(self):
        result = generate()
        assert "MATE_VALUE = 10000.0f" in result

    def test_nnue_constants(self):
        result = generate()
        assert "INPUT_SIZE = 773" in result
        assert "HIDDEN1_SIZE = 256" in result
        assert "HIDDEN2_SIZE = 32" in result
        assert "MAX_HIDDEN2_SIZE = 64" in result
        assert "OUTPUT_SIZE = 1" in result

    def test_search_constants(self):
        result = generate()
        assert "MAX_DEPTH = 50" in result
        assert "QUIESCENCE_MAX_DEPTH = 8" in result

    def test_null_move_namespace(self):
        result = generate()
        assert "namespace null_move {" in result
        assert "MIN_DEPTH = 3" in result
        assert "MATE_MARGIN = 1000" in result

    def test_lmr_namespace(self):
        result = generate()
        assert "namespace lmr {" in result
        assert "MIN_MOVES_SEARCHED = 2" in result

    def test_mcts_constants(self):
        result = generate()
        assert "EXPLORATION_CONSTANT = 1.4f" in result
        assert "MAX_SIMULATION_DEPTH = 100" in result

    def test_bot_constants(self):
        result = generate()
        assert "MAX_RETRIES = 3" in result
        assert 'USER_AGENT = "Lichess-Bot-CPP/1.0"' in result
        assert 'LICHESS_BASE_URL = "https://lichess.org/api"' in result

    def test_curl_constants(self):
        result = generate()
        assert "REQUEST_TIMEOUT = 30L" in result
        assert "MAX_REDIRECTS = 5L" in result

    def test_eval_material(self):
        result = generate()
        assert "MATERIAL_MG[] = {100, 320, 330, 500, 900, 0}" in result
        assert "MATERIAL_EG[] = {110, 310, 330, 520, 950, 0}" in result

    def test_eval_total_phase(self):
        result = generate()
        # 4*1 + 4*1 + 4*2 + 2*4 = 24
        assert "TOTAL_PHASE = 24" in result

    def test_eval_pst_arrays(self):
        result = generate()
        assert "PST_PAWN_MG[64]" in result
        assert "PST_PAWN_EG[64]" in result
        assert "PST_KNIGHT_MG[64]" in result
        assert "PST_KING_EG[64]" in result

    def test_eval_pst_lookup(self):
        result = generate()
        assert "PST_MG[]" in result
        assert "PST_EG[]" in result

    def test_pawn_structure(self):
        result = generate()
        assert "namespace pawn_structure {" in result
        assert "PASSED_BASE = 10" in result
        assert "ISOLATED_MG = 15" in result
        assert "DOUBLED_EG = 15" in result

    def test_rook_file(self):
        result = generate()
        assert "namespace rook_file {" in result
        assert "OPEN_MG = 15" in result

    def test_bishop_pair(self):
        result = generate()
        assert "namespace bishop_pair {" in result
        assert "BONUS_MG = 30" in result
        assert "BONUS_EG = 50" in result

    def test_king_safety(self):
        result = generate()
        assert "namespace king_safety {" in result
        assert "SHIELD_BONUS_MG = 10" in result

    def test_self_play_constants(self):
        result = generate()
        assert "namespace self_play {" in result
        assert "NUM_GAMES = 20000" in result
        assert "SEARCH_DEPTH = 6" in result
        assert "NUM_THREADS = 16" in result
        assert "SEARCH_TIME_MS = 500" in result
        assert "RANDOM_PLIES = 4" in result
        assert "SOFTMAX_PLIES = 20" in result
        assert "SOFTMAX_TEMPERATURE = 300.0f" in result

    def test_no_resign(self):
        result = generate()
        assert "RESIGN" not in result

    def test_valid_cpp_syntax(self):
        result = generate()
        # Every opened brace should have a closing one
        assert result.count("{") == result.count("}")
        # Should end with newline
        assert result.endswith("\n")

    def test_all_namespaces_closed(self):
        result = generate()
        for ns in [
            "nnue",
            "search",
            "null_move",
            "lmr",
            "mcts",
            "bot",
            "curl",
            "eval",
            "pawn_structure",
            "rook_file",
            "bishop_pair",
            "king_safety",
            "self_play",
            "config",
        ]:
            assert f"// namespace {ns}" in result

    def test_constexpr_float_has_f_suffix(self):
        result = generate()
        # All float constexpr values should end with 'f'
        for match in re.finditer(r"constexpr float \w+ = ([\d.]+)(f?);", result):
            assert match.group(2) == "f", (
                f"Float constant missing 'f' suffix: {match.group(0)}"
            )

    def test_constexpr_long_has_l_suffix(self):
        result = generate()
        for match in re.finditer(r"constexpr long \w+ = (\d+)(L?);", result):
            assert match.group(2) == "L", (
                f"Long constant missing 'L' suffix: {match.group(0)}"
            )


class TestLoad:
    def test_load_engine(self):
        cfg = load("engine")
        assert "nnue" in cfg
        assert "search" in cfg

    def test_load_eval(self):
        cfg = load("eval")
        assert "material_mg" in cfg

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load("nonexistent_config_xyz")


class TestMain:
    def test_main_writes_file(self, tmp_path, monkeypatch):
        output = tmp_path / "generated_config.h"
        monkeypatch.setattr(gen_config_header, "OUTPUT", output)
        gen_config_header.main()
        assert output.exists()
        content = output.read_text()
        assert "#pragma once" in content
        assert "namespace config" in content

    def test_main_prints_path(self, tmp_path, monkeypatch, capsys):
        output = tmp_path / "generated_config.h"
        monkeypatch.setattr(gen_config_header, "OUTPUT", output)
        gen_config_header.main()
        captured = capsys.readouterr().out
        assert "Generated" in captured
