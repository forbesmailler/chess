"""Tests for scripts/iterate.py."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from scripts.iterate import (
    ClaudeResult,
    ClaudeRunner,
    RunConfig,
    Task,
    TaskOrchestrator,
    TaskResult,
    TaskStatus,
    commit_changes,
    discard_changes,
    git,
    git_head_sha,
    parse_args,
    squash_task_commits,
)


class TestTaskStatus:
    def test_converged_value(self):
        assert TaskStatus.CONVERGED.value == "converged"

    def test_failed_value(self):
        assert TaskStatus.FAILED.value == "failed"

    def test_max_iterations_value(self):
        assert TaskStatus.MAX_ITERATIONS.value == "max iterations"


class TestTask:
    def test_fields(self):
        t = Task("Test", "Do something")
        assert t.name == "Test"
        assert t.prompt == "Do something"


class TestTaskResult:
    def test_defaults(self):
        r = TaskResult("T", TaskStatus.CONVERGED)
        assert r.iterations == 0
        assert r.elapsed_minutes == 0.0

    def test_custom_values(self):
        r = TaskResult("T", TaskStatus.FAILED, iterations=5, elapsed_minutes=3.2)
        assert r.iterations == 5
        assert r.elapsed_minutes == 3.2


class TestRunConfig:
    def test_defaults(self):
        c = RunConfig()
        assert c.model is None
        assert c.max_iterations == 20
        assert c.default_wait_seconds == 900
        assert "NO_CHANGES" in c.suffix

    def test_custom(self):
        c = RunConfig(model="opus", max_iterations=5)
        assert c.model == "opus"
        assert c.max_iterations == 5


class TestClaudeResult:
    def test_succeeded_on_zero(self):
        r = ClaudeResult(output="ok", exit_code=0)
        assert r.succeeded is True

    def test_not_succeeded_on_nonzero(self):
        r = ClaudeResult(output="error", exit_code=1)
        assert r.succeeded is False

    def test_signalled_no_changes_true(self):
        r = ClaudeResult(output="NO_CHANGES", exit_code=0)
        assert r.signalled_no_changes is True

    def test_signalled_no_changes_false_on_failure(self):
        r = ClaudeResult(output="NO_CHANGES", exit_code=1)
        assert r.signalled_no_changes is False

    def test_signalled_no_changes_false_without_marker(self):
        r = ClaudeResult(output="some output", exit_code=0)
        assert r.signalled_no_changes is False

    def test_signalled_no_changes_substring(self):
        r = ClaudeResult(output="Result: NO_CHANGES found", exit_code=0)
        assert r.signalled_no_changes is True


class TestGitHelpers:
    def test_git_calls_subprocess(self):
        with patch(
            "scripts.iterate.subprocess.run",
            return_value=subprocess.CompletedProcess(["git", "status"], 0),
        ) as mock:
            git("status")
            mock.assert_called_once_with(
                ["git", "status"],
                capture_output=True,
                text=True,
                check=True,
            )

    def test_git_check_false(self):
        with patch(
            "scripts.iterate.subprocess.run",
            return_value=subprocess.CompletedProcess(["git", "clean"], 0),
        ) as mock:
            git("clean", "-fd", check=False)
            mock.assert_called_once_with(
                ["git", "clean", "-fd"],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_git_head_sha(self):
        with patch(
            "scripts.iterate.subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["git", "rev-parse", "HEAD"], 0, stdout="abc123\n"
            ),
        ):
            assert git_head_sha() == "abc123"

    def test_commit_changes_with_diff(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            if args[1] == "diff":
                return subprocess.CompletedProcess(args, 1)  # has changes
            return subprocess.CompletedProcess(args, 0)

        with patch("scripts.iterate.subprocess.run", fake_run):
            result = commit_changes("test commit")
            assert result is True
            assert ["git", "commit", "-m", "test commit"] in calls

    def test_commit_changes_no_diff(self):
        def fake_run(args, **kwargs):
            if args[1] == "diff":
                return subprocess.CompletedProcess(args, 0)  # no changes
            return subprocess.CompletedProcess(args, 0)

        with patch("scripts.iterate.subprocess.run", fake_run):
            result = commit_changes("test commit")
            assert result is False

    def test_discard_changes(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0)

        with patch("scripts.iterate.subprocess.run", fake_run):
            discard_changes()
            assert ["git", "checkout", "--", "."] in calls
            assert ["git", "clean", "-fd"] in calls

    def test_squash_task_commits_no_change(self):
        with patch(
            "scripts.iterate.subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["git", "rev-parse", "HEAD"], 0, stdout="abc123\n"
            ),
        ) as mock:
            squash_task_commits("abc123", "squash msg")
            # Only rev-parse called, no reset or commit
            assert mock.call_count == 1

    def test_squash_task_commits_with_change(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            if "rev-parse" in args:
                return subprocess.CompletedProcess(args, 0, stdout="def456\n")
            return subprocess.CompletedProcess(args, 0)

        with patch("scripts.iterate.subprocess.run", fake_run):
            squash_task_commits("abc123", "squash msg")
            assert ["git", "reset", "--soft", "abc123"] in calls
            assert ["git", "commit", "-m", "squash msg"] in calls


class TestClaudeRunner:
    def test_base_args_no_model(self):
        runner = ClaudeRunner(RunConfig())
        args = runner._base_args()
        assert args == [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
        ]

    def test_base_args_with_model(self):
        runner = ClaudeRunner(RunConfig(model="opus"))
        args = runner._base_args()
        assert "--model" in args
        assert "opus" in args

    def test_looks_like_rate_limit_true(self):
        runner = ClaudeRunner(RunConfig())
        assert runner._looks_like_rate_limit("you've hit your limit") is True
        assert runner._looks_like_rate_limit("usage limit reached") is True
        assert runner._looks_like_rate_limit("Rate Limit exceeded") is True
        assert runner._looks_like_rate_limit("too many requests") is True
        assert runner._looks_like_rate_limit("exceeded the limit") is True

    def test_looks_like_rate_limit_false(self):
        runner = ClaudeRunner(RunConfig())
        assert runner._looks_like_rate_limit("all good") is False
        assert runner._looks_like_rate_limit("") is False

    def test_parse_rate_limit_wait_minutes(self):
        runner = ClaudeRunner(RunConfig())
        wait = runner._parse_rate_limit_wait("please wait 5 minutes")
        assert wait == 5 * 60 + 30

    def test_parse_rate_limit_wait_minute_singular(self):
        runner = ClaudeRunner(RunConfig())
        wait = runner._parse_rate_limit_wait("retry in 1 minute")
        assert wait == 1 * 60 + 30

    def test_parse_rate_limit_wait_default(self):
        runner = ClaudeRunner(RunConfig(default_wait_seconds=600))
        wait = runner._parse_rate_limit_wait("some unknown format")
        assert wait == 600

    def test_invoke_success(self):
        runner = ClaudeRunner(RunConfig())
        with patch(
            "scripts.iterate.subprocess.run",
            return_value=subprocess.CompletedProcess(
                "cmd", 0, stdout='{"result": "done"}', stderr=""
            ),
        ):
            result = runner.invoke("test prompt")
            assert result.succeeded is True
            assert result.output == "done"

    def test_invoke_json_parse_error(self):
        runner = ClaudeRunner(RunConfig())
        with patch(
            "scripts.iterate.subprocess.run",
            return_value=subprocess.CompletedProcess(
                "cmd", 0, stdout="not json", stderr=""
            ),
        ):
            result = runner.invoke("test prompt")
            assert result.output == "not json"

    def test_invoke_continue_session(self):
        runner = ClaudeRunner(RunConfig())
        with patch(
            "scripts.iterate.subprocess.run",
            return_value=subprocess.CompletedProcess(
                "cmd", 0, stdout='{"result": "ok"}', stderr=""
            ),
        ) as mock:
            runner.invoke("test", continue_session=True)
            args = mock.call_args[0][0]
            assert "--continue" in args

    def test_invoke_stores_session_id(self):
        runner = ClaudeRunner(RunConfig())
        with patch(
            "scripts.iterate.subprocess.run",
            return_value=subprocess.CompletedProcess(
                "cmd", 0, stdout='{"result": "ok", "session_id": "s123"}', stderr=""
            ),
        ):
            runner.invoke("test")
            assert runner._last_session_id == "s123"

    def test_invoke_failure_exit_code(self):
        runner = ClaudeRunner(RunConfig())
        with patch(
            "scripts.iterate.subprocess.run",
            return_value=subprocess.CompletedProcess(
                "cmd", 1, stdout="", stderr="error"
            ),
        ):
            result = runner.invoke("test")
            assert result.succeeded is False
            assert result.exit_code == 1


class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["iterate.py"]):
            args = parse_args()
            assert args.model is None
            assert args.prompts is None
            assert args.max_iterations == 20

    def test_custom_model(self):
        with patch("sys.argv", ["iterate.py", "--model", "opus"]):
            args = parse_args()
            assert args.model == "opus"

    def test_custom_prompts(self):
        with patch("sys.argv", ["iterate.py", "-p", "Fix bugs", "-p", "Add tests"]):
            args = parse_args()
            assert args.prompts == ["Fix bugs", "Add tests"]
            assert len(args.prompts) == 2

    def test_max_iterations(self):
        with patch("sys.argv", ["iterate.py", "--max-iterations", "5"]):
            args = parse_args()
            assert args.max_iterations == 5


class TestTaskOrchestrator:
    def _make_orchestrator(self, tasks, **config_kwargs):
        config = RunConfig(**config_kwargs)
        return TaskOrchestrator(tasks, config)

    def test_converges_on_no_changes(self, tmp_path):
        orch = self._make_orchestrator([Task("T1", "do something")], max_iterations=5)
        orch.config.log_file = tmp_path / "log.md"

        with (
            patch.object(
                orch.runner,
                "invoke",
                return_value=ClaudeResult("NO_CHANGES", 0),
            ),
            patch("scripts.iterate.git_head_sha", return_value="abc123"),
            patch("scripts.iterate.commit_changes", return_value=False),
            patch("scripts.iterate.squash_task_commits"),
        ):
            results = orch.run_all()
            assert len(results) == 1
            assert results[0].status == TaskStatus.CONVERGED
            assert results[0].iterations == 1

    def test_fails_on_nonzero_exit(self, tmp_path):
        orch = self._make_orchestrator([Task("T1", "do something")], max_iterations=5)
        orch.config.log_file = tmp_path / "log.md"

        with (
            patch.object(
                orch.runner,
                "invoke",
                return_value=ClaudeResult("error", 1),
            ),
            patch("scripts.iterate.git_head_sha", return_value="abc123"),
            patch("scripts.iterate.discard_changes"),
            patch("scripts.iterate.squash_task_commits"),
        ):
            results = orch.run_all()
            assert len(results) == 1
            assert results[0].status == TaskStatus.FAILED

    def test_max_iterations_reached(self, tmp_path):
        orch = self._make_orchestrator([Task("T1", "do something")], max_iterations=2)
        orch.config.log_file = tmp_path / "log.md"

        with (
            patch.object(
                orch.runner,
                "invoke",
                return_value=ClaudeResult("made changes", 0),
            ),
            patch("scripts.iterate.git_head_sha", return_value="abc123"),
            patch("scripts.iterate.commit_changes", return_value=True),
            patch("scripts.iterate.squash_task_commits"),
        ):
            results = orch.run_all()
            assert len(results) == 1
            assert results[0].status == TaskStatus.MAX_ITERATIONS
            assert results[0].iterations == 2

    def test_multiple_tasks(self, tmp_path):
        tasks = [Task("T1", "first"), Task("T2", "second")]
        orch = self._make_orchestrator(tasks, max_iterations=5)
        orch.config.log_file = tmp_path / "log.md"

        with (
            patch.object(
                orch.runner,
                "invoke",
                return_value=ClaudeResult("NO_CHANGES", 0),
            ),
            patch("scripts.iterate.git_head_sha", return_value="abc123"),
            patch("scripts.iterate.commit_changes", return_value=False),
            patch("scripts.iterate.squash_task_commits"),
        ):
            results = orch.run_all()
            assert len(results) == 2
            assert all(r.status == TaskStatus.CONVERGED for r in results)

    def test_log_file_created(self, tmp_path):
        orch = self._make_orchestrator([Task("T1", "do something")], max_iterations=1)
        orch.config.log_file = tmp_path / "log.md"

        with (
            patch.object(
                orch.runner,
                "invoke",
                return_value=ClaudeResult("made changes", 0),
            ),
            patch("scripts.iterate.git_head_sha", return_value="abc123"),
            patch("scripts.iterate.commit_changes", return_value=True),
            patch("scripts.iterate.squash_task_commits"),
        ):
            orch.run_all()
            assert orch.config.log_file.exists()
            content = orch.config.log_file.read_text()
            assert "Iteration Log" in content


class TestMain:
    def test_main_default_tasks(self):
        with (
            patch("sys.argv", ["iterate.py"]),
            patch("scripts.iterate.TaskOrchestrator") as mock_orch_cls,
        ):
            mock_orch = MagicMock()
            mock_orch.run_all.return_value = [
                TaskResult("T", TaskStatus.CONVERGED, 1, 0.1)
            ]
            mock_orch_cls.return_value = mock_orch

            from scripts.iterate import main

            main()
            mock_orch.run_all.assert_called_once()

    def test_main_custom_prompts(self):
        with (
            patch("sys.argv", ["iterate.py", "-p", "Fix bugs"]),
            patch("scripts.iterate.TaskOrchestrator") as mock_orch_cls,
        ):
            mock_orch = MagicMock()
            mock_orch.run_all.return_value = [
                TaskResult("T", TaskStatus.CONVERGED, 1, 0.1)
            ]
            mock_orch_cls.return_value = mock_orch

            from scripts.iterate import main

            main()
            # First arg to TaskOrchestrator is the task list
            tasks = mock_orch_cls.call_args[0][0]
            assert len(tasks) == 1
            assert tasks[0].prompt == "Fix bugs"

    def test_main_exits_on_failure(self):
        with (
            patch("sys.argv", ["iterate.py"]),
            patch("scripts.iterate.TaskOrchestrator") as mock_orch_cls,
        ):
            mock_orch = MagicMock()
            mock_orch.run_all.return_value = [
                TaskResult("T", TaskStatus.FAILED, 1, 0.1)
            ]
            mock_orch_cls.return_value = mock_orch

            from scripts.iterate import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
