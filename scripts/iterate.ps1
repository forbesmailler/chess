$Prompts = @(
    "Search all source files for bugs: off-by-one errors, null/undefined access, unhandled error paths, race conditions, resource leaks, incorrect boundary checks, and wrong operator precedence. Fix each bug you find with a minimal targeted change.",
    "Identify untested and under-tested code paths. Add unit tests for: edge cases, error handling branches, boundary values, and any function lacking test coverage. Each test should assert exact expected values.",
    "Remove dead code, unused imports, unreachable branches, and redundant logic. Inline trivial one-use helpers. Simplify overly nested conditionals with early returns. Preserve all external behavior.",
    "Find inefficient algorithms, redundant computations, unnecessary allocations, and repeated work in hot paths. Apply targeted fixes: better data structures, caching repeated lookups, avoiding copies. Do not sacrifice readability for marginal gains.",
    "Find hardcoded numeric constants, string literals, and thresholds scattered in source files that should be centralized in config files. Move them and wire up any config loading or generation needed."
)

$Suffix = "After making changes, run all tests, and run the code formatter. Only make changes you are confident are correct. If no changes are needed, respond with exactly NO_CHANGES and nothing else."
$LogFile = Join-Path $PSScriptRoot "iterate_log.md"
$MaxIterations = 20
$completedTasks = @()

Set-Content $LogFile "# Iteration Log`n"
$scriptStart = Get-Date

foreach ($prompt in $Prompts) {
    Write-Host "`n=== Task: $prompt ===" -ForegroundColor Magenta
    $taskStart = Get-Date

    for ($iteration = 1; $iteration -le $MaxIterations; $iteration++) {
        Write-Host "--- Iteration $iteration ---" -ForegroundColor Cyan

        if ($iteration -eq 1) {
            if ($completedTasks.Count -gt 0) {
                $context = ($completedTasks | ForEach-Object { "- $_" }) -join "`n"
                $fullPrompt = "Previously completed tasks this session:`n$context`n`n$prompt $Suffix"
            } else {
                $fullPrompt = "$prompt $Suffix"
            }
            $output = claude -p --dangerously-skip-permissions $fullPrompt 2>&1
        } else {
            $output = claude -p --continue --dangerously-skip-permissions "$prompt $Suffix" 2>&1
        }

        if ($LASTEXITCODE -ne 0 -and -not ($output -match "NO_CHANGES")) {
            Write-Host "Claude exited with error code $LASTEXITCODE" -ForegroundColor Red
            break
        }

        Write-Host $output
        Write-Host ""

        if ($output -match "NO_CHANGES") {
            $elapsed = [math]::Round(((Get-Date) - $taskStart).TotalMinutes, 1)
            Write-Host "Converged after $iteration iteration(s) (${elapsed}m): $prompt" -ForegroundColor Green
            $completedTasks += $prompt
            Add-Content $LogFile "## Completed: $prompt`nConverged after $iteration iteration(s) in ${elapsed}m.`n"
            break
        }

        # Summarize changes for the log via the same conversation
        $summary = claude -p --continue --dangerously-skip-permissions "List each file you modified and what you changed, one bullet per file. No preamble, no explanation." 2>&1
        Add-Content $LogFile "### $prompt (iteration $iteration)`n$summary`n"
    }

    if ($iteration -gt $MaxIterations) {
        Write-Host "Hit max iterations ($MaxIterations) for: $prompt" -ForegroundColor Yellow
        Add-Content $LogFile "## Abandoned: $prompt`nHit max iterations ($MaxIterations).`n"
    }
}

$totalElapsed = [math]::Round(((Get-Date) - $scriptStart).TotalMinutes, 1)
Write-Host "`nAll tasks finished in ${totalElapsed}m. ($($completedTasks.Count)/$($Prompts.Count) converged)" -ForegroundColor Green
