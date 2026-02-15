"""Tests for scripts/train_loop.py."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts import train_loop

POSITION_BYTES = 42
WLD_LINE = "New wins: {new}, Old wins: {old}, Draws: {draws}"


class FakeSubprocess:
    """Mock subprocess.run/Popen that simulates the training pipeline.

    Each iteration has 4 subprocess calls in order:
      0: self-play  (run - creates/extends data file)
      1: train      (run)
      2: export     (run - creates candidate file at --output path)
      3: compare    (Popen - returns W/L/D stdout + returncode)

    Raises KeyboardInterrupt when all iterations are exhausted.
    """

    def __init__(
        self,
        tmp_path,
        compare_results,
        positions_per_iter=100,
        wld=(60, 40, 0),
    ):
        self.tmp_path = tmp_path
        self.compare_results = compare_results
        self.positions_per_iter = positions_per_iter
        self.wld = wld
        self.commands = []
        self._call_idx = 0
        self._iter_idx = 0

    def __call__(self, cmd, **kwargs):
        """Handle subprocess.run calls (phases 0-2)."""
        phase = self._call_idx % 4

        if phase == 0 and self._iter_idx >= len(self.compare_results):
            raise KeyboardInterrupt

        self.commands.append(cmd)
        self._call_idx += 1

        if phase == 0:
            data = self.tmp_path / "training_data.bin"
            with open(data, "ab") as f:
                f.write(b"\x00" * (POSITION_BYTES * self.positions_per_iter))
        elif phase == 1:
            parts = cmd.split()
            for i, part in enumerate(parts):
                if part == "--log" and i + 1 < len(parts):
                    log_file = self.tmp_path / parts[i + 1]
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    log_file.write_text("# Training Log\n")
                    break
        elif phase == 2:
            parts = cmd.split()
            output_path = None
            for i, part in enumerate(parts):
                if part == "--output" and i + 1 < len(parts):
                    output_path = self.tmp_path / parts[i + 1]
                    break
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"candidate")

        return subprocess.CompletedProcess(cmd, 0, stdout="")

    def popen(self, cmd, **kwargs):
        """Handle subprocess.Popen calls (phase 3: compare)."""
        self.commands.append(cmd)
        self._call_idx += 1

        improved = self.compare_results[self._iter_idx]
        self._iter_idx += 1
        new_w, old_w, draws = self.wld
        if not improved:
            new_w, old_w = old_w, new_w
        stdout = WLD_LINE.format(new=new_w, old=old_w, draws=draws) + "\n"

        mock_proc = MagicMock()
        mock_proc.stdout = iter(stdout.splitlines(keepends=True))
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0 if improved else 1
        return mock_proc

    @property
    def compare_commands(self):
        return [c for c in self.commands if "--compare" in c]


@pytest.fixture
def loop_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_loop, "BOT_EXE", "bot")
    monkeypatch.setattr("sys.argv", ["train_loop.py"])
    return tmp_path


def run_loop(tmp_path, compare_results, **kwargs):
    fake = FakeSubprocess(tmp_path, compare_results, **kwargs)
    with (
        patch("scripts.train_loop.subprocess.run", fake),
        patch("scripts.train_loop.subprocess.Popen", fake.popen),
    ):
        try:
            train_loop.main()
        except KeyboardInterrupt:
            pass
    return fake


def pointer_file(tmp_path):
    return tmp_path / "models" / "current_best.txt"


def read_pointer(tmp_path):
    pf = pointer_file(tmp_path)
    if pf.exists():
        return pf.read_text().strip()
    return None


class TestAccepted:
    def test_pointer_file_created(self, loop_env):
        run_loop(loop_env, [True])
        assert pointer_file(loop_env).exists()
        assert read_pointer(loop_env).startswith("models")

    def test_archived_to_accepted(self, loop_env):
        run_loop(loop_env, [True])
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        assert len(accepted) == 1

    def test_no_candidate_remains(self, loop_env):
        run_loop(loop_env, [True])
        remaining = list((loop_env / "models").glob("nnue_*pos.bin"))
        assert len(remaining) == 0

    def test_archive_name_contains_position_count(self, loop_env):
        run_loop(loop_env, [True], positions_per_iter=200)
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        assert "200pos" in accepted[0].name

    def test_pointer_points_to_accepted_model(self, loop_env):
        run_loop(loop_env, [True])
        best = read_pointer(loop_env)
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        assert accepted[0].name in best


class TestRejected:
    def test_no_pointer_file_created(self, loop_env):
        run_loop(loop_env, [False])
        assert not pointer_file(loop_env).exists()

    def test_archived_to_rejected(self, loop_env):
        run_loop(loop_env, [False])
        rejected = list((loop_env / "models" / "rejected").glob("*.bin"))
        assert len(rejected) == 1

    def test_no_candidate_remains(self, loop_env):
        run_loop(loop_env, [False])
        remaining = list((loop_env / "models").glob("nnue_*pos.bin"))
        assert len(remaining) == 0

    def test_existing_pointer_preserved(self, loop_env):
        pf = pointer_file(loop_env)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text("models/accepted/old_model.bin\n")
        run_loop(loop_env, [False])
        assert read_pointer(loop_env) == "models/accepted/old_model.bin"


class TestSelfPlayCommand:
    def test_no_pointer_omits_weights_arg(self, loop_env):
        fake = run_loop(loop_env, [True])
        selfplay_cmd = fake.commands[0]
        assert "--selfplay" in selfplay_cmd
        assert "models" not in selfplay_cmd

    def test_existing_pointer_includes_weights_arg(self, loop_env):
        pf = pointer_file(loop_env)
        old_path = str(Path("models/accepted/old.bin"))
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(old_path + "\n")
        (loop_env / "models" / "accepted").mkdir(parents=True, exist_ok=True)
        (loop_env / "models" / "accepted" / "old.bin").write_bytes(b"existing")
        fake = run_loop(loop_env, [True])
        assert old_path in fake.commands[0]


class TestCompareCommand:
    def test_no_prior_weights_vs_handcrafted(self, loop_env):
        fake = run_loop(loop_env, [False])
        compare_cmd = fake.compare_commands[0]
        assert "handcrafted" in compare_cmd
        assert "models" in compare_cmd

    def test_with_prior_pointer_vs_model(self, loop_env):
        pf = pointer_file(loop_env)
        old_path = str(Path("models/accepted/old.bin"))
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(old_path + "\n")
        (loop_env / "models" / "accepted").mkdir(parents=True, exist_ok=True)
        (loop_env / "models" / "accepted" / "old.bin").write_bytes(b"existing")
        fake = run_loop(loop_env, [True])
        compare_cmd = fake.compare_commands[0]
        assert old_path in compare_cmd


class TestMultipleIterations:
    def test_data_accumulates(self, loop_env):
        run_loop(loop_env, [True, True], positions_per_iter=50)
        data = loop_env / "training_data.bin"
        assert data.stat().st_size == POSITION_BYTES * 50 * 2

    def test_second_iteration_uses_accepted_weights(self, loop_env):
        fake = run_loop(loop_env, [True, True])
        # First selfplay: no pointer yet → no model path
        assert "models" not in fake.commands[0]
        # Second selfplay: pointer exists → includes accepted path
        accepted_prefix = str(Path("models/accepted"))
        assert accepted_prefix in fake.commands[4]

    def test_rejected_then_accepted(self, loop_env):
        run_loop(loop_env, [False, True])
        assert pointer_file(loop_env).exists()
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


class TestRunCompare:
    @staticmethod
    def _mock_popen(stdout_text, returncode=0):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(stdout_text.splitlines(keepends=True))
        mock_proc.wait.return_value = None
        mock_proc.returncode = returncode
        return mock_proc

    def test_parses_wld(self):
        stdout = "Some output\nNew wins: 55, Old wins: 40, Draws: 5\nDone\n"
        with patch(
            "scripts.train_loop.subprocess.Popen",
            return_value=self._mock_popen(stdout, 0),
        ):
            result = train_loop.run_compare("cmd")
            assert result == {
                "improved": True,
                "new_wins": 55,
                "old_wins": 40,
                "draws": 5,
            }

    def test_no_match_returns_zeros(self):
        with patch(
            "scripts.train_loop.subprocess.Popen",
            return_value=self._mock_popen("no match\n", 1),
        ):
            result = train_loop.run_compare("cmd")
            assert result == {
                "improved": False,
                "new_wins": 0,
                "old_wins": 0,
                "draws": 0,
            }

    def test_failure_returncode(self):
        stdout = "New wins: 30, Old wins: 70, Draws: 0\n"
        with patch(
            "scripts.train_loop.subprocess.Popen",
            return_value=self._mock_popen(stdout, 1),
        ):
            result = train_loop.run_compare("cmd")
            assert result["improved"] is False
            assert result["new_wins"] == 30


class TestPositionCounting:
    def test_position_count_exact(self, loop_env):
        run_loop(loop_env, [True], positions_per_iter=100)
        data = loop_env / "training_data.bin"
        assert data.stat().st_size == POSITION_BYTES * 100
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        assert len(accepted) == 1
        assert "100pos" in accepted[0].name

    def test_position_count_accumulates_correctly(self, loop_env):
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


class TestReportFiles:
    def test_accepted_has_report(self, loop_env):
        run_loop(loop_env, [True])
        reports = [
            r
            for r in (loop_env / "models" / "accepted").glob("*.md")
            if not r.name.endswith("_train.md")
        ]
        assert len(reports) == 1

    def test_rejected_has_report(self, loop_env):
        run_loop(loop_env, [False])
        reports = [
            r
            for r in (loop_env / "models" / "rejected").glob("*.md")
            if not r.name.endswith("_train.md")
        ]
        assert len(reports) == 1

    def test_report_contains_status(self, loop_env):
        run_loop(loop_env, [True])
        report = [
            r
            for r in (loop_env / "models" / "accepted").glob("*.md")
            if not r.name.endswith("_train.md")
        ][0]
        content = report.read_text()
        assert "## ACCEPTED" in content

    def test_report_contains_wld_table(self, loop_env):
        run_loop(loop_env, [True], wld=(55, 40, 5))
        report = [
            r
            for r in (loop_env / "models" / "accepted").glob("*.md")
            if not r.name.endswith("_train.md")
        ][0]
        content = report.read_text()
        assert "| W | L | D |" in content
        assert "55" in content
        assert "40" in content
        assert "5" in content

    def test_report_contains_model_names(self, loop_env):
        run_loop(loop_env, [False])
        report = [
            r
            for r in (loop_env / "models" / "rejected").glob("*.md")
            if not r.name.endswith("_train.md")
        ][0]
        content = report.read_text()
        assert "handcrafted" in content
        assert "(new)" in content
        assert "(old)" in content

    def test_report_rejected_status(self, loop_env):
        run_loop(loop_env, [False])
        report = [
            r
            for r in (loop_env / "models" / "rejected").glob("*.md")
            if not r.name.endswith("_train.md")
        ][0]
        content = report.read_text()
        assert "## REJECTED" in content

    def test_multiple_iterations_have_reports(self, loop_env):
        run_loop(loop_env, [True, False, True])
        accepted_reports = [
            r
            for r in (loop_env / "models" / "accepted").glob("*.md")
            if not r.name.endswith("_train.md")
        ]
        rejected_reports = [
            r
            for r in (loop_env / "models" / "rejected").glob("*.md")
            if not r.name.endswith("_train.md")
        ]
        assert len(accepted_reports) == 2
        assert len(rejected_reports) == 1

    def test_accepted_has_train_log(self, loop_env):
        run_loop(loop_env, [True])
        logs = list((loop_env / "models" / "accepted").glob("*_train.md"))
        assert len(logs) == 1

    def test_rejected_has_train_log(self, loop_env):
        run_loop(loop_env, [False])
        logs = list((loop_env / "models" / "rejected").glob("*_train.md"))
        assert len(logs) == 1

    def test_no_train_log_in_models_root(self, loop_env):
        run_loop(loop_env, [True])
        logs = list((loop_env / "models").glob("*_train.md"))
        assert len(logs) == 0

    def test_multiple_iterations_train_logs(self, loop_env):
        run_loop(loop_env, [True, False, True])
        accepted_logs = list((loop_env / "models" / "accepted").glob("*_train.md"))
        rejected_logs = list((loop_env / "models" / "rejected").glob("*_train.md"))
        assert len(accepted_logs) == 2
        assert len(rejected_logs) == 1


class TestCompareOnly:
    @staticmethod
    def _fake_popen(calls, returncode=0):
        def popen(cmd, **kwargs):
            calls.append(cmd)
            stdout = "New wins: 60, Old wins: 40, Draws: 0\n"
            mock_proc = MagicMock()
            mock_proc.stdout = iter(stdout.splitlines(keepends=True))
            mock_proc.wait.return_value = None
            mock_proc.returncode = returncode
            return mock_proc

        return popen

    def test_compare_only_skips_selfplay_and_training(self, loop_env, monkeypatch):
        """--compare-only skips selfplay and training, goes straight to compare."""
        candidate = loop_env / "models" / "my_candidate.bin"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(b"candidate")
        monkeypatch.setattr(
            "sys.argv",
            ["train_loop.py", "--compare-only", "--candidate", str(candidate)],
        )

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="")

        with (
            patch("scripts.train_loop.subprocess.run", fake_run),
            patch("scripts.train_loop.subprocess.Popen", self._fake_popen(calls)),
        ):
            try:
                train_loop.main()
            except (KeyboardInterrupt, Exception):
                pass

        selfplay_calls = [c for c in calls if "--selfplay" in c]
        assert len(selfplay_calls) == 0

    def test_compare_only_requires_candidate(self, loop_env, monkeypatch):
        """--compare-only without --candidate breaks the loop."""
        monkeypatch.setattr("sys.argv", ["train_loop.py", "--compare-only"])
        calls = []
        with (
            patch(
                "scripts.train_loop.subprocess.run",
                return_value=subprocess.CompletedProcess("cmd", 0, stdout=""),
            ),
            patch("scripts.train_loop.subprocess.Popen", self._fake_popen(calls)),
        ):
            try:
                train_loop.main()
            except (KeyboardInterrupt, Exception):
                pass
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        rejected = list((loop_env / "models" / "rejected").glob("*.bin"))
        assert len(accepted) == 0
        assert len(rejected) == 0

    def test_candidate_not_found_breaks_loop(self, loop_env, monkeypatch):
        """If candidate does not exist, loop breaks with error message."""
        monkeypatch.setattr(
            "sys.argv",
            ["train_loop.py", "--compare-only", "--candidate", "nonexistent.bin"],
        )
        calls = []
        with (
            patch(
                "scripts.train_loop.subprocess.run",
                return_value=subprocess.CompletedProcess("cmd", 0, stdout=""),
            ),
            patch("scripts.train_loop.subprocess.Popen", self._fake_popen(calls)),
        ):
            try:
                train_loop.main()
            except (KeyboardInterrupt, Exception):
                pass
        accepted = list((loop_env / "models" / "accepted").glob("*.bin"))
        rejected = list((loop_env / "models" / "rejected").glob("*.bin"))
        assert len(accepted) == 0
        assert len(rejected) == 0


class TestTrainOnly:
    def test_train_only_skips_selfplay(self, loop_env, monkeypatch):
        """--train-only skips selfplay but still trains and compares."""
        monkeypatch.setattr("sys.argv", ["train_loop.py", "--train-only"])
        data = loop_env / "training_data.bin"
        data.write_bytes(b"\x00" * (POSITION_BYTES * 100))

        calls = []
        compare_done = False

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "export_nnue" in cmd:
                parts = cmd.split()
                for i, part in enumerate(parts):
                    if part == "--output" and i + 1 < len(parts):
                        out = loop_env / parts[i + 1]
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(b"candidate")
                        break
            return subprocess.CompletedProcess(cmd, 0, stdout="")

        def fake_popen(cmd, **kwargs):
            nonlocal compare_done
            calls.append(cmd)
            if compare_done:
                raise KeyboardInterrupt
            compare_done = True
            stdout = "New wins: 60, Old wins: 40, Draws: 0\n"
            mock_proc = MagicMock()
            mock_proc.stdout = iter(stdout.splitlines(keepends=True))
            mock_proc.wait.return_value = None
            mock_proc.returncode = 0
            return mock_proc

        with (
            patch("scripts.train_loop.subprocess.run", fake_run),
            patch("scripts.train_loop.subprocess.Popen", fake_popen),
        ):
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


class TestDataCapping:
    def test_compute_param_count(self):
        count = train_loop.compute_param_count()
        assert count == 773 * 256 + 256 + 256 * 64 + 64 + 64 * 1 + 1

    def test_cap_noop_when_under_limit(self, tmp_path):
        data = tmp_path / "data.bin"
        data.write_bytes(b"\x00" * (POSITION_BYTES * 10))
        train_loop.cap_training_data(data)
        assert data.stat().st_size == POSITION_BYTES * 10

    def test_cap_trims_oldest(self, tmp_path):
        param_count = train_loop.compute_param_count()
        max_positions = train_loop.DATA_CAP_MULTIPLIER * param_count
        total = max_positions + 100
        data = tmp_path / "data.bin"
        # Write identifiable data: each position is 42 bytes
        content = bytearray()
        for i in range(total):
            pos = bytes([i % 256]) * POSITION_BYTES
            content.extend(pos)
        data.write_bytes(bytes(content))
        train_loop.cap_training_data(data)
        result_size = data.stat().st_size
        assert result_size == max_positions * POSITION_BYTES
        # Verify oldest 100 positions were removed (kept newest)
        result = data.read_bytes()
        # First position in result should be what was position 100 originally
        expected_first_byte = 100 % 256
        assert result[0] == expected_first_byte

    def test_cap_creates_backup(self, tmp_path):
        param_count = train_loop.compute_param_count()
        max_positions = train_loop.DATA_CAP_MULTIPLIER * param_count
        total = max_positions + 10
        data = tmp_path / "training_data.bin"
        original = b"\x00" * (POSITION_BYTES * total)
        data.write_bytes(original)
        train_loop.cap_training_data(data)
        backups = list(tmp_path.glob("training_data_*.bin"))
        assert len(backups) == 1
        assert backups[0].stat().st_size == len(original)

    def test_cap_nonexistent_file(self, tmp_path):
        data = tmp_path / "missing.bin"
        train_loop.cap_training_data(data)  # should not raise


class TestHelperFunctions:
    def test_read_current_best_missing(self, tmp_path):
        assert train_loop.read_current_best(tmp_path / "missing.txt") is None

    def test_read_current_best_empty(self, tmp_path):
        f = tmp_path / "best.txt"
        f.write_text("")
        assert train_loop.read_current_best(f) is None

    def test_read_write_roundtrip(self, tmp_path):
        f = tmp_path / "models" / "best.txt"
        train_loop.write_current_best(f, "models/accepted/test.bin")
        assert train_loop.read_current_best(f) == "models/accepted/test.bin"

    def test_write_report(self, tmp_path):
        archive = tmp_path / "test.bin"
        archive.write_bytes(b"data")
        wld = {"new_wins": 55, "old_wins": 40, "draws": 5}
        train_loop.write_report(archive, "ACCEPTED", "new_model", "old_model", wld)
        report = tmp_path / "test.md"
        assert report.exists()
        content = report.read_text()
        assert "# test" in content
        assert "## ACCEPTED" in content
        assert "55" in content
        assert "new_model (new)" in content
        assert "old_model (old)" in content
