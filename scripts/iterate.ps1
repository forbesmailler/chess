$Prompts = @(
    "Search the entire repo for bugs, logic errors, off-by-one errors, race conditions, and edge cases. Fix any you find. Run tests to verify.",
    "Review test coverage. Add missing unit tests targeting untested branches and edge cases. Aim for >90% line coverage. Run tests to verify.",
    "Search for redundant, dead, or overly verbose code. Remove duplication, inline trivial helpers, and simplify logic while preserving all functionality. Run tests to verify.",
    "Optimize hot paths. Prefer algorithmic improvements over micro-optimizations. Run tests to verify.",
    "Find hardcoded constants and magic numbers that should live in config files. Move them there and update any config generation if needed. Run tests to verify."
)

$LogFile = Join-Path $PSScriptRoot "iterate_log.md"
Set-Content $LogFile "# Iteration Log`n"

foreach ($prompt in $Prompts) {
    Write-Host "`n=== Task: $prompt ===" -ForegroundColor Magenta

    $iteration = 0
    while ($true) {
        $iteration++
        Write-Host "--- Iteration $iteration ---" -ForegroundColor Cyan

        $suffix = "If no changes are needed, respond with exactly NO_CHANGES and nothing else."

        if ($iteration -eq 1) {
            # First iteration: fresh conversation with log context
            $logContent = Get-Content $LogFile -Raw
            $fullPrompt = "Context from previous tasks in this session:`n$logContent`n`n$prompt $suffix"
            $output = claude -p --dangerously-skip-permissions $fullPrompt 2>&1
        } else {
            # Subsequent iterations: continue the conversation
            $output = claude -p --continue --dangerously-skip-permissions "$prompt $suffix" 2>&1
        }

        Write-Host $output
        Write-Host ""

        if ($output -match "NO_CHANGES") {
            Write-Host "Converged after $iteration iteration(s): $prompt" -ForegroundColor Green
            Add-Content $LogFile "## Completed: $prompt`nConverged after $iteration iteration(s).`n"
            break
        }

        # Ask Claude to summarize what it changed for the log
        $summary = claude -p --continue --dangerously-skip-permissions "Summarize what you just changed in 2-3 bullet points. Be specific about files and what changed." 2>&1
        Add-Content $LogFile "## $prompt (iteration $iteration)`n$summary`n"
    }
}

Write-Host "`nAll prompts converged." -ForegroundColor Green
