# iterate.ps1

Automated code improvement loop using Claude Code. Runs a sequence of tasks, each iterated until Claude reports no more changes needed.

## Usage

```powershell
# Run with default tasks
.\scripts\iterate.ps1

# Specify a model
.\scripts\iterate.ps1 -Model sonnet

# Custom prompts (overrides default tasks)
.\scripts\iterate.ps1 -Prompts "Fix compiler warnings.", "Add input validation."

# Single custom prompt
.\scripts\iterate.ps1 -Prompts "Remove unused dependencies."

# Custom prompts with model
.\scripts\iterate.ps1 -Model sonnet -Prompts "Fix compiler warnings."
```

## How it works

1. Runs each task sequentially (bug fixes, test coverage, dead code, optimization, config)
2. For each task, Claude iterates until it responds with `NO_CHANGES`
3. Within a task, `--continue` preserves conversation context across iterations
4. Each iteration is committed individually to preserve progress on crash
5. At the end of each task, per-iteration commits are squashed into a single commit
6. On error, the failed iteration's changes are discarded; previous iterations' work is kept and squashed
7. The script continues to the next task after an error or hitting max iterations
8. A log is written to `scripts/iterate_log.md`

## Tasks

| Task | What it does |
|------|-------------|
| Bug fixes | Off-by-one errors, null access, race conditions, resource leaks, boundary checks |
| Test coverage | Add missing unit tests for edge cases, error branches, boundary values |
| Conciseness | Remove dead code, inline trivial helpers, simplify nested conditionals, preserve behavior |
| Optimization | Better data structures, caching, avoiding copies in hot paths |
| Config | Move hardcoded constants and magic numbers to config files |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `-Prompts` | 5 built-in tasks | Override with custom prompt list |
| `-Model` | CLI default | Claude model to use |
| `$MaxIterations` | 20 | Max iterations per task before stopping |

## Output

Terminal output is color-coded:
- **Magenta**: Task start
- **Cyan**: Iteration start
- **Green**: Task converged
- **Yellow**: Hit max iterations
- **Red**: Task failed (Claude error)

A summary table is printed at the end:

```
=== Summary (12.3m) ===
  converged       Bug fixes
  converged       Test coverage
  max iterations  Conciseness
  converged       Optimization
  failed          Config
```

## Git behavior

- Each iteration is committed individually for crash safety
- At the end of each task, iteration commits are squashed into one: `"<task name> - automated iteration"`
- On error, uncommitted changes from the failed iteration are discarded
- `scripts/iterate_log.md` is excluded from commits

## Requirements

- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated
- Git repository (for checkpointing)
