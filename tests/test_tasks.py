"""Tests for tasks.py (invoke task definitions)."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch


def make_ctx():
    """Create a mock context (dict-like is enough for .body calls)."""
    ctx = MagicMock()
    ctx.cd.return_value.__enter__ = MagicMock(return_value=None)
    ctx.cd.return_value.__exit__ = MagicMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# Simple task smoke tests
# ---------------------------------------------------------------------------


def test_gen_config_runs_script():
    import tasks

    c = make_ctx()
    tasks.gen_config.body(c)
    c.run.assert_called_once_with("python scripts/gen_config_header.py")


def test_format_runs_three_commands():
    import tasks

    c = make_ctx()
    tasks.format.body(c)
    assert c.run.call_count == 3
    calls = [a[0][0] for a in c.run.call_args_list]
    assert any("ruff format" in cmd for cmd in calls)
    assert any("ruff check" in cmd for cmd in calls)
    assert any("clang-format" in cmd for cmd in calls)


def test_test_cpp_uses_cd_and_ctest():
    import tasks

    c = make_ctx()
    tasks.test_cpp.body(c)
    c.cd.assert_called_once_with("engine/build")
    c.run.assert_called_once_with("ctest -C Release --output-on-failure")


def test_build_uses_cd_and_cmake():
    import tasks

    c = make_ctx()
    tasks.build.body(c)
    c.cd.assert_called_once_with("engine/build")
    assert c.run.call_count == 2
    calls = [a[0][0] for a in c.run.call_args_list]
    assert any("cmake .." in cmd for cmd in calls)
    assert any("cmake --build" in cmd for cmd in calls)


def test_test_calls_pytest_and_test_cpp():
    import tasks

    c = make_ctx()
    with patch.object(tasks, "test_cpp") as mock_test_cpp:
        tasks.test.body(c)
    c.run.assert_called_once_with("pytest")
    mock_test_cpp.assert_called_once_with(c)


def test_prepare_calls_all_four_steps():
    import tasks

    c = make_ctx()
    with (
        patch.object(tasks, "gen_config") as mock_gc,
        patch.object(tasks, "format") as mock_fmt,
        patch.object(tasks, "build") as mock_build,
        patch.object(tasks, "test") as mock_test,
    ):
        tasks.prepare.body(c)
    mock_gc.assert_called_once_with(c)
    mock_fmt.assert_called_once_with(c)
    mock_build.assert_called_once_with(c)
    mock_test.assert_called_once_with(c)


# ---------------------------------------------------------------------------
# train task branch coverage
# ---------------------------------------------------------------------------


def _run_train(c, **kwargs):
    import tasks

    with patch.object(tasks, "prepare"):
        tasks.train.body(c, **kwargs)
    return c.run.call_args[0][0]


def test_train_default_cmd_has_required_flags():
    c = make_ctx()
    cmd = _run_train(c)
    assert "scripts/train_loop.py" in cmd
    assert "--games" in cmd
    assert "--epochs" in cmd
    assert "--eval-weight" in cmd


def test_train_train_only_flag():
    c = make_ctx()
    cmd = _run_train(c, train_only=True)
    assert "--train-only" in cmd


def test_train_compare_only_flag():
    c = make_ctx()
    cmd = _run_train(c, compare_only=True)
    assert "--compare-only" in cmd


def test_train_candidate_path_included():
    c = make_ctx()
    cmd = _run_train(c, candidate="models/candidate.bin")
    assert "--candidate models/candidate.bin" in cmd


def test_train_no_candidate_no_candidate_flag():
    c = make_ctx()
    cmd = _run_train(c, candidate=None)
    assert "--candidate" not in cmd


def test_train_freeze_baseline_flag():
    c = make_ctx()
    cmd = _run_train(c, freeze_baseline=True)
    assert "--freeze-baseline" in cmd


def test_train_no_freeze_baseline_by_default():
    c = make_ctx()
    cmd = _run_train(c)
    assert "--freeze-baseline" not in cmd


def test_train_custom_games_and_epochs():
    c = make_ctx()
    cmd = _run_train(c, games=500, epochs=20)
    assert "--games 500" in cmd
    assert "--epochs 20" in cmd


# ---------------------------------------------------------------------------
# build_book task
# ---------------------------------------------------------------------------


def test_build_book_default_no_workers():
    import tasks

    c = make_ctx()
    tasks.build_book.body(c, pgn="games.pgn")
    cmd = c.run.call_args[0][0]
    assert "scripts/build_opening_book.py games.pgn" in cmd
    assert "--workers" not in cmd


def test_build_book_with_workers():
    import tasks

    c = make_ctx()
    tasks.build_book.body(c, pgn="games.pgn", workers=4)
    cmd = c.run.call_args[0][0]
    assert "--workers 4" in cmd


def test_build_book_custom_options():
    import tasks

    c = make_ctx()
    tasks.build_book.body(
        c, pgn="games.pgn", output="mybook.bin", min_elo=2500, min_count=20
    )
    cmd = c.run.call_args[0][0]
    assert "--output mybook.bin" in cmd
    assert "--min-elo 2500" in cmd
    assert "--min-count 20" in cmd


# ---------------------------------------------------------------------------
# challenge task
# ---------------------------------------------------------------------------


def test_challenge_rated_by_default(tmp_path, monkeypatch):
    import tasks

    monkeypatch.chdir(tmp_path)
    c = make_ctx()
    tasks.challenge.body(c, username="somebot")
    cmd = c.run.call_args[0][0]
    assert "rated" in cmd
    assert "somebot" in cmd


def test_challenge_casual_flag(tmp_path, monkeypatch):
    import tasks

    monkeypatch.chdir(tmp_path)
    c = make_ctx()
    tasks.challenge.body(c, username="somebot", casual=True)
    cmd = c.run.call_args[0][0]
    assert "casual" in cmd


def test_challenge_custom_time_and_increment(tmp_path, monkeypatch):
    import tasks

    monkeypatch.chdir(tmp_path)
    c = make_ctx()
    tasks.challenge.body(c, username="testbot", time=60, increment=2)
    cmd = c.run.call_args[0][0]
    assert "60" in cmd
    assert "2" in cmd
    assert "testbot" in cmd


def test_challenge_reads_env_file(tmp_path, monkeypatch):
    import tasks

    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_CHALLENGE_KEY=challenge_value\n# comment\n\nBAD_LINE\n")
    monkeypatch.delenv("TEST_CHALLENGE_KEY", raising=False)

    c = make_ctx()
    tasks.challenge.body(c, username="bot")
    assert os.environ.get("TEST_CHALLENGE_KEY") == "challenge_value"

    # cleanup
    del os.environ["TEST_CHALLENGE_KEY"]


def test_challenge_env_setdefault_does_not_overwrite(tmp_path, monkeypatch):
    """setdefault means existing env vars are not overwritten."""
    import tasks

    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_VAR=from_file\n")
    monkeypatch.setenv("EXISTING_VAR", "original")

    c = make_ctx()
    tasks.challenge.body(c, username="bot")
    assert os.environ["EXISTING_VAR"] == "original"


def test_challenge_no_env_file(tmp_path, monkeypatch):
    """No .env file — should not raise, just proceed."""
    import tasks

    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / ".env").exists()
    c = make_ctx()
    tasks.challenge.body(c, username="bot")
    c.run.assert_called_once()


# ---------------------------------------------------------------------------
# play_stockfish task
# ---------------------------------------------------------------------------


def test_play_stockfish_no_retrain_flag():
    import tasks

    c = make_ctx()
    with patch.object(tasks, "prepare"):
        tasks.play_stockfish.body(c, elo=2000, no_retrain=True)
    cmd = c.run.call_args[0][0]
    assert "--no-retrain" in cmd


def test_play_stockfish_default_no_no_retrain():
    import tasks

    c = make_ctx()
    with patch.object(tasks, "prepare"):
        tasks.play_stockfish.body(c, elo=1500)
    cmd = c.run.call_args[0][0]
    assert "--no-retrain" not in cmd


def test_play_stockfish_includes_elo():
    import tasks

    c = make_ctx()
    with patch.object(tasks, "prepare"):
        tasks.play_stockfish.body(c, elo=2600)
    cmd = c.run.call_args[0][0]
    assert "2600" in cmd


# ---------------------------------------------------------------------------
# sweep task
# ---------------------------------------------------------------------------


def test_sweep_runs_once_per_weight(tmp_path, monkeypatch):
    import tasks

    monkeypatch.chdir(tmp_path)
    c = make_ctx()
    with patch.object(tasks, "prepare"):
        tasks.sweep.body(c, eval_weights="0.5,0.75,0.9")
    assert c.run.call_count == 3


def test_sweep_each_cmd_has_eval_weight(tmp_path, monkeypatch):
    import tasks

    monkeypatch.chdir(tmp_path)
    c = make_ctx()
    with patch.object(tasks, "prepare"):
        tasks.sweep.body(c, eval_weights="0.3,0.8")
    cmds = [a[0][0] for a in c.run.call_args_list]
    assert any("--eval-weight 0.3" in cmd for cmd in cmds)
    assert any("--eval-weight 0.8" in cmd for cmd in cmds)


def test_sweep_freeze_baseline_default_true(tmp_path, monkeypatch):
    import tasks

    monkeypatch.chdir(tmp_path)
    c = make_ctx()
    with patch.object(tasks, "prepare"):
        tasks.sweep.body(c, eval_weights="0.5")
    cmd = c.run.call_args[0][0]
    assert "--freeze-baseline" in cmd


def test_sweep_freeze_baseline_false(tmp_path, monkeypatch):
    import tasks

    monkeypatch.chdir(tmp_path)
    c = make_ctx()
    with patch.object(tasks, "prepare"):
        tasks.sweep.body(c, eval_weights="0.5", freeze_baseline=False)
    cmd = c.run.call_args[0][0]
    assert "--freeze-baseline" not in cmd


def test_sweep_appends_to_data_when_tmp_has_content(tmp_path, monkeypatch):
    import tempfile

    import tasks

    monkeypatch.chdir(tmp_path)
    data_file = tmp_path / "training_data.bin"
    data_file.write_bytes(b"\x00" * 42)

    real_named_tmp = tempfile.NamedTemporaryFile

    def fake_named_tmp(**kwargs):
        obj = real_named_tmp(**kwargs)
        Path(obj.name).write_bytes(b"\x01" * 84)
        return obj

    c = make_ctx()
    with (
        patch.object(tasks, "prepare"),
        patch("tempfile.NamedTemporaryFile", fake_named_tmp),
    ):
        tasks.sweep.body(c, eval_weights="0.5", data=str(data_file))

    # Original 42 bytes + 84 appended = 126
    assert data_file.stat().st_size == 126


def test_sweep_does_not_append_when_tmp_empty(tmp_path, monkeypatch):
    import tasks

    monkeypatch.chdir(tmp_path)
    data_file = tmp_path / "training_data.bin"
    data_file.write_bytes(b"\x00" * 42)

    c = make_ctx()
    with patch.object(tasks, "prepare"):
        tasks.sweep.body(c, eval_weights="0.5", data=str(data_file))

    assert data_file.stat().st_size == 42


def test_sweep_tmp_file_cleaned_up(tmp_path, monkeypatch):
    import tempfile

    import tasks

    monkeypatch.chdir(tmp_path)
    created_paths = []
    real_named_tmp = tempfile.NamedTemporaryFile

    def tracking_named_tmp(**kwargs):
        obj = real_named_tmp(**kwargs)
        created_paths.append(obj.name)
        return obj

    c = make_ctx()
    with (
        patch.object(tasks, "prepare"),
        patch("tempfile.NamedTemporaryFile", tracking_named_tmp),
    ):
        tasks.sweep.body(c, eval_weights="0.5")

    assert len(created_paths) == 1
    assert not Path(created_paths[0]).exists()


def test_sweep_prints_positions_appended(tmp_path, monkeypatch, capsys):
    import tempfile

    import tasks

    monkeypatch.chdir(tmp_path)
    data_file = tmp_path / "training_data.bin"
    data_file.write_bytes(b"\x00" * 42)

    real_named_tmp = tempfile.NamedTemporaryFile

    def fake_named_tmp(**kwargs):
        obj = real_named_tmp(**kwargs)
        Path(obj.name).write_bytes(b"\x01" * 42 * 3)
        return obj

    c = make_ctx()
    with (
        patch.object(tasks, "prepare"),
        patch("tempfile.NamedTemporaryFile", fake_named_tmp),
    ):
        tasks.sweep.body(c, eval_weights="0.5", data=str(data_file))

    out = capsys.readouterr().out
    assert "3" in out
    assert "comparison positions" in out


# ---------------------------------------------------------------------------
# deploy task
# ---------------------------------------------------------------------------


def _deploy(c, **kwargs):
    import tasks

    with (
        patch.object(tasks, "gen_config"),
        patch.object(tasks, "format"),
        patch.object(tasks, "build"),
    ):
        tasks.deploy.body(c, **kwargs)
    return [a[0][0] for a in c.run.call_args_list]


def test_deploy_with_explicit_weights():
    c = make_ctx()
    calls = _deploy(c, weights="models/best.bin")
    assert any("cp models/best.bin" in cmd for cmd in calls)


def test_deploy_without_weights_uses_pointer():
    c = make_ctx()
    calls = _deploy(c)
    assert any("test -f" in cmd for cmd in calls)


def test_deploy_restarts_service():
    c = make_ctx()
    calls = _deploy(c)
    assert any("systemctl restart" in cmd for cmd in calls)


def test_deploy_pulls_code_first():
    c = make_ctx()
    _deploy(c)
    assert c.cd.called


# ---------------------------------------------------------------------------
# play_bots and play_humans smoke tests
# ---------------------------------------------------------------------------


def test_play_bots_calls_prepare_and_run():
    import tasks

    c = make_ctx()
    with patch.object(tasks, "prepare") as mock_prep:
        tasks.play_bots.body(c)
    mock_prep.assert_called_once_with(c)
    cmd = c.run.call_args[0][0]
    assert "scripts/play_bots.py" in cmd


def test_play_humans_calls_prepare_and_run():
    import tasks

    c = make_ctx()
    with patch.object(tasks, "prepare") as mock_prep:
        tasks.play_humans.body(c)
    mock_prep.assert_called_once_with(c)
    cmd = c.run.call_args[0][0]
    assert "scripts/play_humans.py" in cmd


def test_play_humans_custom_time_and_increment():
    import tasks

    c = make_ctx()
    with patch.object(tasks, "prepare"):
        tasks.play_humans.body(c, time=120, increment=5)
    cmd = c.run.call_args[0][0]
    assert "--time 120" in cmd
    assert "--increment 5" in cmd
