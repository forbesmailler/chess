#!/usr/bin/env python3
"""
Iterative Claude Code task runner with git integration.

Runs a series of coding tasks via Claude Code in print mode (-p), committing
after each successful iteration, squashing per task, and retrying on rate limits.

Usage:
    python iterate.py                          # Run default tasks
    python iterate.py --model opus             # Specify model
    python iterate.py -p "Fix all TODOs" -p "Add docstrings"  # Custom prompts
    python iterate.py --max-iterations 10      # Override iteration cap
    python iterate.py --parallel               # Run tasks on separate worktrees
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

# ─── Data types ───────────────────────────────────────────────────────────────


class TaskStatus(Enum):
    CONVERGED = "converged"
    FAILED = "failed"
    MAX_ITERATIONS = "max iterations"


@dataclass
class Task:
    name: str
    prompt: str


@dataclass
class TaskResult:
    name: str
    status: TaskStatus
    iterations: int = 0
    elapsed_minutes: float = 0.0


@dataclass
class RunConfig:
    model: Optional[str] = None
    max_iterations: int = 20
    default_wait_seconds: int = 900  # 15 min fallback for rate limits
    log_file: Path = field(
        default_factory=lambda: Path(__file__).parent / "iterate_log.md"
    )
    suffix: str = (
        "After making changes, run all tests and the code formatter. "
        "Only make changes you are confident are correct. "
        "If no changes are needed, respond with exactly NO_CHANGES and nothing else."
    )


DEFAULT_TASKS = [
    Task(
        "Bug fixes",
        "Search every source file in the repo for bugs. Look for: off-by-one errors, "
        "race conditions, resource leaks, incorrect boundary checks, and silent data "
        "truncation. For each bug, make the smallest fix that corrects the issue.",
    ),
    Task(
        "Test coverage",
        "Find functions and branches that have no tests or weak tests. Write focused "
        "unit tests that cover: error handling paths, boundary values (zero, empty, max, "
        "negative), off-by-one boundaries, and uncommon but valid inputs. Each assertion "
        "should check an exact expected value, not just truthiness. When testing "
        "collections, also assert the count.",
    ),
    Task(
        "Conciseness",
        "Make the codebase more concise without changing behavior. Remove dead code, "
        "unused imports, unreachable branches, and commented-out code. Inline functions "
        "that are called only once and add no clarity. Replace deeply nested if/else "
        "chains with early returns or guard clauses. Merge duplicate logic into shared "
        "helpers only when there are 3+ copies.",
    ),
    Task(
        "Optimization",
        "Find performance bottlenecks in the codebase. Look for: O(n^2) or worse "
        "algorithms that could be O(n log n) or O(n), repeated lookups that should be "
        "cached, unnecessary copies of large objects, allocations inside tight loops, and "
        "redundant recomputation. Apply targeted fixes. Do not sacrifice readability for "
        "marginal gains.",
    ),
    Task(
        "Config",
        "Find hardcoded numeric constants, string literals, URLs, timeouts, thresholds, "
        "and tuning parameters scattered across source files. Move each to an appropriate "
        "yaml config file. If a config loading mechanism already exists, use it. If moved "
        "values are needed at compile time, update any config generation scripts accordingly.",
    ),
]


# ─── Git helpers ──────────────────────────────────────────────────────────────


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def git_head_sha() -> str:
    return git("rev-parse", "HEAD").stdout.strip()


def commit_changes(message: str) -> bool:
    """Stage and commit. Returns True if a commit was created."""
    git("add", "-A", "--", ".", ":!scripts/iterate_log.md", check=False)
    result = git("diff", "--cached", "--quiet", check=False)
    if result.returncode != 0:  # there are staged changes
        git("commit", "-m", message)
        return True
    return False


def discard_changes() -> None:
    git("checkout", "--", ".", check=False)
    git("clean", "-fd", check=False)


def squash_task_commits(base_sha: str, message: str) -> None:
    if git_head_sha() != base_sha:
        git("reset", "--soft", base_sha)
        git("commit", "-m", message)


# ─── Claude Code runner ──────────────────────────────────────────────────────


@dataclass
class ClaudeResult:
    output: str
    exit_code: int

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0

    @property
    def signalled_no_changes(self) -> bool:
        """Only trust NO_CHANGES from a successful run."""
        return self.succeeded and "NO_CHANGES" in self.output


class ClaudeRunner:
    """Wraps Claude Code CLI invocation with rate-limit retry."""

    def __init__(self, config: RunConfig):
        self.config = config
        self._last_session_id: Optional[str] = None

    def _base_args(self) -> list[str]:
        args = ["claude", "-p", "--dangerously-skip-permissions"]
        if self.config.model:
            args += ["--model", self.config.model]
        # Use JSON output so we can reliably extract session_id for --continue
        args += ["--output-format", "json"]
        return args

    def _parse_rate_limit_wait(self, text: str) -> int:
        """Try to extract a wait duration from rate limit messages. Returns seconds."""
        # Match "2:30 PM" style
        match = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM))", text, re.IGNORECASE)
        if match:
            try:
                reset_time = datetime.strptime(match.group(1), "%I:%M %p").replace(
                    year=datetime.now().year,
                    month=datetime.now().month,
                    day=datetime.now().day,
                )
                if reset_time < datetime.now():
                    reset_time += timedelta(days=1)
                return int((reset_time - datetime.now()).total_seconds()) + 30
            except ValueError:
                pass
        # Match "N minutes"
        match = re.search(r"(\d+)\s*minutes?", text)
        if match:
            return int(match.group(1)) * 60 + 30
        return self.config.default_wait_seconds

    def _looks_like_rate_limit(self, text: str) -> bool:
        patterns = [
            r"you've hit your limit",
            r"usage limit",
            r"rate limit",
            r"exceeded.*limit",
            r"too many requests",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def invoke(self, prompt: str, continue_session: bool = False) -> ClaudeResult:
        """
        Call Claude Code, retrying on rate limits.

        Args:
            prompt: The prompt text to send.
            continue_session: If True, use --continue to resume the most recent
                              session. Claude Code's --continue flag works with -p
                              to resume the last conversation.
        """
        args = self._base_args()
        if continue_session:
            args.append("--continue")
        args.append(prompt)

        while True:
            result = subprocess.run(args, capture_output=True, text=True)
            combined = result.stdout + result.stderr

            if self._looks_like_rate_limit(combined):
                wait = self._parse_rate_limit_wait(combined)
                resume_at = (datetime.now() + timedelta(seconds=wait)).strftime(
                    "%-I:%M %p"
                )
                print(
                    f"  ⏳ Rate limit hit. Waiting {wait / 60:.1f} min (until ~{resume_at})...",
                    flush=True,
                )
                time.sleep(wait)
                print("  ▶ Resuming...", flush=True)
                continue

            # Parse JSON output to get the text result
            output_text = combined
            try:
                data = json.loads(result.stdout)
                output_text = data.get("result", result.stdout)
                if session_id := data.get("session_id"):
                    self._last_session_id = session_id
            except (json.JSONDecodeError, TypeError):
                output_text = result.stdout or result.stderr

            return ClaudeResult(output=output_text, exit_code=result.returncode)


# ─── Task orchestrator ────────────────────────────────────────────────────────


class TaskOrchestrator:
    def __init__(self, tasks: list[Task], config: RunConfig):
        self.tasks = tasks
        self.config = config
        self.runner = ClaudeRunner(config)
        self.results: list[TaskResult] = []

    def run_all(self) -> list[TaskResult]:
        self.config.log_file.write_text("# Iteration Log\n\n")
        overall_start = time.monotonic()

        for task in self.tasks:
            result = self._run_task(task)
            self.results.append(result)

        elapsed = (time.monotonic() - overall_start) / 60
        self._print_summary(elapsed)
        return self.results

    def _run_task(self, task: Task) -> TaskResult:
        print(f"\n{'=' * 60}")
        print(f"  Task: {task.name}")
        print(f"{'=' * 60}")

        task_start = time.monotonic()
        base_sha = git_head_sha()
        status = TaskStatus.MAX_ITERATIONS
        iterations = 0

        for iteration in range(1, self.config.max_iterations + 1):
            iterations = iteration
            print(f"\n  --- {task.name} · iteration {iteration} ---")

            if iteration == 1:
                prompt = f"{task.prompt} {self.config.suffix}"
                result = self.runner.invoke(prompt, continue_session=False)
            else:
                prompt = f"Keep going with the same task. {self.config.suffix}"
                result = self.runner.invoke(prompt, continue_session=True)

            # Print a truncated preview of output
            preview = result.output[:500] + ("..." if len(result.output) > 500 else "")
            print(f"  {preview}\n")

            # Check exit code FIRST, independently of output content
            if not result.succeeded:
                print(f"  ✗ Claude exited with code {result.exit_code}")
                self._log(
                    f"## Failed: {task.name} (iteration {iteration}, exit code {result.exit_code})\n"
                )
                status = TaskStatus.FAILED
                discard_changes()
                break

            # Only check NO_CHANGES after confirming success
            if result.signalled_no_changes:
                elapsed = (time.monotonic() - task_start) / 60
                print(f"  ✓ Converged after {iteration} iteration(s) ({elapsed:.1f}m)")
                self._log(
                    f"## Completed: {task.name} in {iteration} iteration(s) ({elapsed:.1f}m)\n"
                )
                status = TaskStatus.CONVERGED
                # Commit any final changes before declaring convergence
                commit_changes(f"{task.name} - iteration {iteration}")
                break

            # Commit after each successful iteration
            commit_changes(f"{task.name} - iteration {iteration}")
            self._log(f"### {task.name} - iteration {iteration}\n")

        if status == TaskStatus.MAX_ITERATIONS:
            print(
                f"  ⚠ Hit max iterations ({self.config.max_iterations}) for: {task.name}"
            )
            self._log(
                f"## Max iterations: {task.name} after {self.config.max_iterations} iterations\n"
            )

        # Squash per-iteration commits into one per task
        squash_task_commits(base_sha, f"{task.name} - automated iteration")

        elapsed = (time.monotonic() - task_start) / 60
        return TaskResult(task.name, status, iterations, elapsed)

    def _log(self, text: str) -> None:
        with self.config.log_file.open("a") as f:
            f.write(text)

    def _print_summary(self, elapsed_minutes: float) -> None:
        colors = {
            TaskStatus.CONVERGED: "\033[32m",  # green
            TaskStatus.FAILED: "\033[31m",  # red
            TaskStatus.MAX_ITERATIONS: "\033[33m",  # yellow
        }
        reset = "\033[0m"

        print(f"\n{'=' * 60}")
        print(f"  Summary ({elapsed_minutes:.1f}m)")
        print(f"{'=' * 60}")
        for r in self.results:
            c = colors.get(r.status, "")
            print(
                f"  {c}{r.status.value:<15}{reset} {r.name} ({r.iterations} iter, {r.elapsed_minutes:.1f}m)"
            )
        print()


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative Claude Code task runner with git integration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", "-m", help="Claude model to use (e.g. opus, sonnet)")
    parser.add_argument(
        "--prompt",
        "-p",
        action="append",
        dest="prompts",
        help="Custom task prompt (can be repeated). Overrides default tasks.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max iterations per task (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RunConfig(
        model=args.model,
        max_iterations=args.max_iterations,
    )

    if args.prompts:
        tasks = [Task(f"Task {i + 1}", p) for i, p in enumerate(args.prompts)]
    else:
        tasks = DEFAULT_TASKS

    orchestrator = TaskOrchestrator(tasks, config)
    results = orchestrator.run_all()

    # Exit with non-zero if any task failed
    if any(r.status == TaskStatus.FAILED for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
