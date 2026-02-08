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
