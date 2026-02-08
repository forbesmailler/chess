param(
    [string]$Model = ""
)

$Tasks = @(
    @{ Name = "Bug fixes";    Prompt = "Search every source file in the repo for bugs. Look for: off-by-one errors, race conditions, resource leaks, incorrect boundary checks, and silent data truncation. For each bug, make the smallest fix that corrects the issue." },
    @{ Name = "Test coverage"; Prompt = "Find functions and branches that have no tests or weak tests. Write focused unit tests that cover: error handling paths, boundary values (zero, empty, max, negative), off-by-one boundaries, and uncommon but valid inputs. Each assertion should check an exact expected value, not just truthiness. When testing collections, also assert the count." },
    @{ Name = "Conciseness";   Prompt = "Make the codebase more concise without changing behavior. Remove dead code, unused imports, unreachable branches, and commented-out code. Inline functions that are called only once and add no clarity. Replace deeply nested if/else chains with early returns or guard clauses. Merge duplicate logic into shared helpers only when there are 3+ copies." },
    @{ Name = "Optimization";  Prompt = "Find performance bottlenecks in the codebase. Look for: O(n^2) or worse algorithms that could be O(n log n) or O(n), repeated lookups that should be cached, unnecessary copies of large objects, allocations inside tight loops, and redundant recomputation. Apply targeted fixes. Do not sacrifice readability for marginal gains." },
    @{ Name = "Config";        Prompt = "Find hardcoded numeric constants, string literals, URLs, timeouts, thresholds, and tuning parameters scattered across source files. Move each to an appropriate yaml config file. If a config loading mechanism already exists, use it. If moved values are needed at compile time, update any config generation scripts accordingly." }
)

$Suffix = "After making changes, run all tests and the code formatter. Only make changes you are confident are correct. If no changes are needed, respond with exactly NO_CHANGES and nothing else."
$LogFile = Join-Path $PSScriptRoot "iterate_log.md"
$MaxIterations = 20
$ModelArgs = if ($Model) { @("--model", $Model) } else { @() }

Set-Content $LogFile "# Iteration Log`n"
$scriptStart = Get-Date
$completed = @()
$results = @()

function Invoke-Claude {
    param([string[]]$ExtraArgs)
    $allArgs = @("-p", "--dangerously-skip-permissions") + $ModelArgs + $ExtraArgs
    $output = & claude @allArgs 2>&1
    return $output
}

function Commit-Changes {
    param([string]$Message)
    git add -A -- . ":!scripts/iterate_log.md"
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m $Message
    }
}

function Discard-Changes {
    git checkout -- .
    git clean -fd
}

function Squash-TaskCommits {
    param([string]$BaseSha, [string]$Message)
    $headSha = git rev-parse HEAD
    if ($headSha -ne $BaseSha) {
        git reset --soft $BaseSha
        git commit -m $Message
    }
}

foreach ($task in $Tasks) {
    $name = $task.Name
    $prompt = $task.Prompt
    Write-Host "`n=== Task: $name ===" -ForegroundColor Magenta
    $taskStart = Get-Date
    $baseSha = git rev-parse HEAD
    $status = "max iterations"

    for ($iteration = 1; $iteration -le $MaxIterations; $iteration++) {
        Write-Host "--- $name iteration $iteration ---" -ForegroundColor Cyan

        if ($iteration -eq 1) {
            $output = Invoke-Claude -ExtraArgs "$prompt $Suffix"
        } else {
            $output = Invoke-Claude -ExtraArgs "--continue", "Keep going with the same task. $Suffix"
        }

        Write-Host $output
        Write-Host ""

        if ($LASTEXITCODE -ne 0 -and -not ($output -match "NO_CHANGES")) {
            Write-Host "Claude exited with error code $LASTEXITCODE" -ForegroundColor Red
            Add-Content $LogFile "## Failed: $name (iteration $iteration, exit code $LASTEXITCODE)`n"
            $status = "failed"
            Discard-Changes
            break
        }

        if ($output -match "NO_CHANGES") {
            $elapsed = [math]::Round(((Get-Date) - $taskStart).TotalMinutes, 1)
            Write-Host "Converged after $iteration iteration(s) (${elapsed}m)" -ForegroundColor Green
            $completed += $name
            Add-Content $LogFile "## Completed: $name in $iteration iteration(s) (${elapsed}m)`n"
            $status = "converged"
            break
        }

        # Commit after each successful iteration to preserve progress
        Commit-Changes "$name - iteration $iteration"
        Add-Content $LogFile "### $name - iteration $iteration`n"
    }

    if ($status -eq "max iterations") {
        Write-Host "Hit max iterations ($MaxIterations) for: $name" -ForegroundColor Yellow
        Add-Content $LogFile "## Max iterations: $name after $MaxIterations iterations`n"
    }

    # Squash per-iteration commits into one per task
    Squash-TaskCommits $baseSha "$name - automated iteration"

    $results += @{ Name = $name; Status = $status }
}

$totalElapsed = [math]::Round(((Get-Date) - $scriptStart).TotalMinutes, 1)
Write-Host "`n=== Summary (${totalElapsed}m) ===" -ForegroundColor Magenta
foreach ($r in $results) {
    $color = switch ($r.Status) { "converged" { "Green" } "failed" { "Red" } default { "Yellow" } }
    Write-Host "  $($r.Status.PadRight(15)) $($r.Name)" -ForegroundColor $color
}
