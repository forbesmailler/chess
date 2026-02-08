"""Tests for config/load_config.py."""

import pytest

from config.load_config import deploy, engine, eval_, load, training


def test_load_engine_config():
    cfg = engine()
    assert "nnue" in cfg
    assert "search" in cfg
    assert "mcts" in cfg
    assert "bot" in cfg
    assert cfg["nnue"]["input_size"] == 773


def test_load_training_config():
    cfg = training()
    assert "self_play" in cfg
    assert "training" in cfg
    assert "invoke" in cfg
    assert cfg["training"]["epochs"] == 100


def test_load_eval_config():
    cfg = eval_()
    assert len(cfg["material_mg"]) == 6
    assert len(cfg["material_eg"]) == 6
    assert cfg["material_mg"][0] == 100  # pawn


def test_load_deploy_config():
    cfg = deploy()
    assert "paths" in cfg
    assert "vps" in cfg
    assert "repo_dir" in cfg["vps"]


def test_load_nonexistent():
    with pytest.raises(FileNotFoundError):
        load("nonexistent_config_file")


def test_load_returns_same_object_on_cache_hit():
    cfg1 = load("engine")
    cfg2 = load("engine")
    assert cfg1 is cfg2


def test_load_different_names_return_different_objects():
    cfg_engine = load("engine")
    cfg_eval = load("eval")
    assert cfg_engine is not cfg_eval


def test_engine_eval_material_values_count():
    cfg = eval_()
    assert len(cfg["material_mg"]) == 6
    assert len(cfg["material_eg"]) == 6
    # Material values: pawn, knight, bishop, rook, queen, king
    assert cfg["material_mg"][4] > cfg["material_mg"][0]  # queen > pawn


def test_engine_search_constants():
    cfg = engine()
    assert cfg["search"]["max_depth"] == 50
    assert cfg["search"]["time_allocation_divisor"] == 40
    assert cfg["search"]["time_check_interval"] == 2048


def test_eval_pst_dimensions():
    cfg = eval_()
    # Each PST should have 8 rows of 8 values
    for piece in ["pawn", "knight", "bishop", "rook", "queen", "king"]:
        for phase in ["mg", "eg"]:
            key = f"pst_{piece}_{phase}"
            pst = cfg[key]
            assert len(pst) == 8, f"{key} has {len(pst)} rows, expected 8"
            for i, row in enumerate(pst):
                assert len(row) == 8, f"{key} row {i} has {len(row)} values, expected 8"


def test_training_self_play_defaults():
    cfg = training()
    sp = cfg["self_play"]
    assert sp["num_games"] > 0
    assert sp["max_game_ply"] > 0
    assert sp["search_time_ms"] > 0
    assert sp["num_threads"] >= 1


def test_eval_returns_cached():
    cfg1 = eval_()
    cfg2 = eval_()
    assert cfg1 is cfg2


def test_training_returns_cached():
    cfg1 = training()
    cfg2 = training()
    assert cfg1 is cfg2


def test_deploy_returns_cached():
    cfg1 = deploy()
    cfg2 = deploy()
    assert cfg1 is cfg2
