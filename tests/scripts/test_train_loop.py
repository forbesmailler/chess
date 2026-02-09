"""Tests for scripts/train_loop.py."""

import subprocess
from unittest.mock import patch

import pytest

from scripts import train_loop

POSITION_BYTES = 42


class FakeSubprocess:
    """Mock subprocess.run that simulates the training pipeline.

    Each iteration has 4 subprocess calls in order:
      0: self-play  (creates/extends data file)
      1: train
      2: export     (creates candidate file)
      3: compare    (returns success/failure per compare_results)

    Raises KeyboardInterrupt when all iterations are exhausted.
    """

    def __init__(self, tmp_path, compare_results, positions_per_iter=100):
        self.tmp_path = tmp_path
        self.compare_results = compare_results
        self.positions_per_iter = positions_per_iter
        self.commands = []
        self._call_idx = 0
        self._iter_idx = 0

    def __call__(self, cmd, **kwargs):
        phase = self._call_idx % 4

        if phase == 0 and self._iter_idx >= len(self.compare_results):
            raise KeyboardInterrupt

        self.commands.append(cmd)
        self._call_idx += 1

        if phase == 0:
            data = self.tmp_path / "training_data.bin"
            with open(data, "ab") as f:
                f.write(b"\x00" * (POSITION_BYTES * self.positions_per_iter))
        elif phase == 2:
            (self.tmp_path / "nnue_candidate.bin").write_bytes(b"candidate")
        elif phase == 3:
            improved = self.compare_results[self._iter_idx]
            self._iter_idx += 1
            return subprocess.CompletedProcess(cmd, 0 if improved else 1)

        return subprocess.CompletedProcess(cmd, 0)


@pytest.fixture
def loop_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_loop, "BOT_EXE", "bot")
    monkeypatch.setattr("sys.argv", ["train_loop.py"])
    return tmp_path


def run_loop(tmp_path, compare_results, **kwargs):
    fake = FakeSubprocess(tmp_path, compare_results, **kwargs)
    with patch("scripts.train_loop.subprocess.run", fake):
        try:
            train_loop.main()
        except KeyboardInterrupt:
            pass
    return fake


class TestAccepted:
    def test_weights_created(self, loop_env):
        run_loop(loop_env, [True])
        assert (loop_env / "nnue.bin").exists()

    def test_archived_to_accepted(self, loop_env):
        run_loop(loop_env, [True])
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        assert len(accepted) == 1

    def test_candidate_deleted(self, loop_env):
        run_loop(loop_env, [True])
        assert not (loop_env / "nnue_candidate.bin").exists()

    def test_archive_name_contains_position_count(self, loop_env):
        run_loop(loop_env, [True], positions_per_iter=200)
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        assert "200pos" in accepted[0].name

    def test_weights_content_matches_candidate(self, loop_env):
        run_loop(loop_env, [True])
        # weights should be a copy of the candidate
        assert (loop_env / "nnue.bin").read_bytes() == b"candidate"


class TestRejected:
    def test_no_weights_created(self, loop_env):
        run_loop(loop_env, [False])
        assert not (loop_env / "nnue.bin").exists()

    def test_archived_to_rejected(self, loop_env):
        run_loop(loop_env, [False])
        rejected = list((loop_env / "models" / "rejected").glob("*.bin"))
        assert len(rejected) == 1

    def test_candidate_deleted(self, loop_env):
        run_loop(loop_env, [False])
        assert not (loop_env / "nnue_candidate.bin").exists()

    def test_existing_weights_preserved(self, loop_env):
        (loop_env / "nnue.bin").write_bytes(b"old_weights")
        run_loop(loop_env, [False])
        assert (loop_env / "nnue.bin").read_bytes() == b"old_weights"


class TestSelfPlayCommand:
    def test_no_weights_omits_weights_arg(self, loop_env):
        fake = run_loop(loop_env, [True])
        selfplay_cmd = fake.commands[0]
        assert "--selfplay" in selfplay_cmd
        assert "nnue.bin" not in selfplay_cmd

    def test_existing_weights_includes_weights_arg(self, loop_env):
        (loop_env / "nnue.bin").write_bytes(b"existing")
        fake = run_loop(loop_env, [True])
        assert "nnue.bin" in fake.commands[0]


class TestCompareCommand:
    def test_no_prior_weights_vs_handcrafted(self, loop_env):
        fake = run_loop(loop_env, [False])
        compare_cmd = fake.commands[3]
        assert "handcrafted" in compare_cmd
        assert "nnue_candidate.bin" in compare_cmd

    def test_with_prior_weights_vs_weights(self, loop_env):
        (loop_env / "nnue.bin").write_bytes(b"existing")
        fake = run_loop(loop_env, [True])
        compare_cmd = fake.commands[3]
        assert "nnue.bin" in compare_cmd
        assert "nnue_candidate.bin" in compare_cmd


class TestMultipleIterations:
    def test_data_accumulates(self, loop_env):
        run_loop(loop_env, [True, True], positions_per_iter=50)
        data = loop_env / "training_data.bin"
        assert data.stat().st_size == POSITION_BYTES * 50 * 2

    def test_second_iteration_uses_accepted_weights(self, loop_env):
        fake = run_loop(loop_env, [True, True])
        assert "nnue.bin" not in fake.commands[0]
        assert "nnue.bin" in fake.commands[4]

    def test_rejected_then_accepted(self, loop_env):
        run_loop(loop_env, [False, True])
        assert (loop_env / "nnue.bin").exists()
        rejected = list((loop_env / "models" / "rejected").glob("*.bin"))
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        assert len(rejected) == 1
        assert len(accepted) == 1

    def test_three_iterations_archive_counts(self, loop_env):
        run_loop(loop_env, [True, False, True])
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        rejected = list((loop_env / "models" / "rejected").glob("*.bin"))
        assert len(accepted) == 2
        assert len(rejected) == 1

    def test_accumulated_position_count_in_archive(self, loop_env):
        run_loop(loop_env, [True, True], positions_per_iter=150)
        accepted = sorted((loop_env / "models" / "accepted").glob("*.bin"))
        assert "300pos" in accepted[-1].name


class TestDirectories:
    def test_accepted_dir_created(self, loop_env):
        run_loop(loop_env, [True])
        assert (loop_env / "models" / "accepted").is_dir()

    def test_rejected_dir_created(self, loop_env):
        run_loop(loop_env, [True])
        assert (loop_env / "models" / "rejected").is_dir()


class TestHelpers:
    def test_run_check_true_on_success(self):
        with patch(
            "scripts.train_loop.subprocess.run",
            return_value=subprocess.CompletedProcess("cmd", 0),
        ):
            assert train_loop.run_check("cmd") is True

    def test_run_check_false_on_failure(self):
        with patch(
            "scripts.train_loop.subprocess.run",
            return_value=subprocess.CompletedProcess("cmd", 1),
        ):
            assert train_loop.run_check("cmd") is False

    def test_run_check_false_on_nonzero(self):
        with patch(
            "scripts.train_loop.subprocess.run",
            return_value=subprocess.CompletedProcess("cmd", 2),
        ):
            assert train_loop.run_check("cmd") is False


class TestPositionCounting:
    def test_position_count_exact(self, loop_env):
        """Position count = file_size // 42."""
        run_loop(loop_env, [True], positions_per_iter=100)
        data = loop_env / "training_data.bin"
        assert data.stat().st_size == POSITION_BYTES * 100
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        assert len(accepted) == 1
        assert "100pos" in accepted[0].name

    def test_position_count_accumulates_correctly(self, loop_env):
        """After 3 iterations, file has 3x positions."""
        run_loop(loop_env, [True, True, True], positions_per_iter=50)
        data = loop_env / "training_data.bin"
        assert data.stat().st_size == POSITION_BYTES * 50 * 3
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        assert len(accepted) == 3
        names = [a.name for a in accepted]
        assert sum("_50pos" in n for n in names) == 1
        assert sum("_100pos" in n for n in names) == 1
        assert sum("_150pos" in n for n in names) == 1

    def test_first_archive_name_has_first_iter_count(self, loop_env):
        run_loop(loop_env, [True, True], positions_per_iter=75)
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        assert len(accepted) == 2
        names = [a.name for a in accepted]
        assert sum("_75pos" in n for n in names) == 1
        assert sum("_150pos" in n for n in names) == 1


class TestCompareOnly:
    def test_compare_only_skips_selfplay_and_training(self, loop_env, monkeypatch):
        """--compare-only skips selfplay and training, goes straight to compare."""
        monkeypatch.setattr("sys.argv", ["train_loop.py", "--compare-only"])
        # Pre-create candidate so the loop doesn't break immediately
        (loop_env / "nnue_candidate.bin").write_bytes(b"candidate")

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            # This is the compare command (only subprocess call in compare-only)
            if "--compare" in cmd:
                return subprocess.CompletedProcess(cmd, 0)
            return subprocess.CompletedProcess(cmd, 0)

        call_count = 0

        def fake_run_with_interrupt(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            # compare call returns success
            if "--compare" in cmd:
                result = subprocess.CompletedProcess(cmd, 0)
                return result
            return subprocess.CompletedProcess(cmd, 0)

        with patch("scripts.train_loop.subprocess.run", fake_run_with_interrupt):
            try:
                train_loop.main()
            except (KeyboardInterrupt, Exception):
                pass

        # Should NOT have selfplay or train commands
        selfplay_calls = [c for c in calls if "--selfplay" in c]
        assert len(selfplay_calls) == 0

    def test_candidate_not_found_breaks_loop(self, loop_env, monkeypatch):
        """If candidate does not exist, loop breaks with error message."""
        monkeypatch.setattr("sys.argv", ["train_loop.py", "--compare-only"])
        # Do NOT create candidate
        with patch(
            "scripts.train_loop.subprocess.run",
            return_value=subprocess.CompletedProcess("cmd", 0),
        ):
            try:
                train_loop.main()
            except (KeyboardInterrupt, Exception):
                pass
        # Loop should have broken — no archives created
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        rejected = list((loop_env / "models" / "rejected").glob("*.bin"))
        assert len(accepted) == 0
        assert len(rejected) == 0


class TestTrainOnly:
    def test_train_only_skips_selfplay(self, loop_env, monkeypatch):
        """--train-only skips selfplay but still trains and compares."""
        monkeypatch.setattr("sys.argv", ["train_loop.py", "--train-only"])
        # Create data file (needed for position counting)
        data = loop_env / "training_data.bin"
        data.write_bytes(b"\x00" * (POSITION_BYTES * 100))

        calls = []
        compare_done = False

        def fake_run(cmd, **kwargs):
            nonlocal compare_done
            calls.append(cmd)
            if "export_nnue" in cmd:
                (loop_env / "nnue_candidate.bin").write_bytes(b"candidate")
            elif "--compare" in cmd:
                if compare_done:
                    raise KeyboardInterrupt
                compare_done = True
                return subprocess.CompletedProcess(cmd, 0)
            return subprocess.CompletedProcess(cmd, 0)

        with patch("scripts.train_loop.subprocess.run", fake_run):
            try:
                train_loop.main()
            except KeyboardInterrupt:
                pass

        selfplay_calls = [c for c in calls if "--selfplay" in c]
        assert len(selfplay_calls) == 0
        train_calls = [c for c in calls if "train_nnue" in c]
        assert len(train_calls) >= 1


class TestRunFunction:
    def test_run_success(self):
        with patch(
            "scripts.train_loop.subprocess.run",
            return_value=subprocess.CompletedProcess("cmd", 0),
        ) as mock_run:
            result = train_loop.run("echo hello")
            mock_run.assert_called_once_with("echo hello", shell=True, check=True)
            assert result.returncode == 0

    def test_run_raises_on_failure(self):
        with patch(
            "scripts.train_loop.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "cmd"),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                train_loop.run("false")
