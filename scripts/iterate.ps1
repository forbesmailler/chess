param(
    [string]$Model = ""
)

$Tasks = @(
    @{ Name = "Bug fixes";    Prompt = "Search all source files for bugs: off-by-one errors, null/undefined access, unhandled error paths, race conditions, resource leaks, incorrect boundary checks, and wrong operator precedence. Fix each bug you find with a minimal targeted change." },
    @{ Name = "Test coverage"; Prompt = "Identify untested and under-tested code paths. Add unit tests for: edge cases, error handling branches, boundary values, and any function lacking test coverage. Each test should assert exact expected values." },
    @{ Name = "Dead code";     Prompt = "Remove dead code, unused imports, unreachable branches, and redundant logic. Inline trivial one-use helpers. Simplify overly nested conditionals with early returns. Preserve all external behavior." },
    @{ Name = "Optimization";  Prompt = "Find inefficient algorithms, redundant computations, unnecessary allocations, and repeated work in hot paths. Apply targeted fixes: better data structures, caching repeated lookups, avoiding copies. Do not sacrifice readability for marginal gains." },
    @{ Name = "Config";        Prompt = "Find hardcoded numeric constants, string literals, and thresholds scattered in source files that should be centralized in config files. Move them and wire up any config loading or generation needed." }
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
    param([string[]]$Args)
    $allArgs = @("-p", "--dangerously-skip-permissions") + $ModelArgs + $Args
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

foreach ($task in $Tasks) {
    $name = $task.Name
    $prompt = $task.Prompt
    Write-Host "`n=== Task: $name ===" -ForegroundColor Magenta
    $taskStart = Get-Date
    $status = "abandoned"

    for ($iteration = 1; $iteration -le $MaxIterations; $iteration++) {
        Write-Host "--- $name iteration $iteration ---" -ForegroundColor Cyan

        if ($iteration -eq 1) {
            $context = ""
            if ($completed.Count -gt 0) {
                $bullets = ($completed | ForEach-Object { "- $_" }) -join "`n"
                $context = "Previously completed tasks this session:`n$bullets`n`n"
            }
            $output = Invoke-Claude @("$context$prompt $Suffix")
        } else {
            $output = Invoke-Claude @("--continue", "Keep going with the same task. $Suffix")
        }

        Write-Host $output
        Write-Host ""

        if ($LASTEXITCODE -ne 0 -and -not ($output -match "NO_CHANGES")) {
            Write-Host "Claude exited with error code $LASTEXITCODE" -ForegroundColor Red
            Add-Content $LogFile "## Failed: $name (iteration $iteration, exit code $LASTEXITCODE)`n"
            $status = "failed"
            Commit-Changes "$name - partial (failed at iteration $iteration)"
            break
        }

        if ($output -match "NO_CHANGES") {
            $elapsed = [math]::Round(((Get-Date) - $taskStart).TotalMinutes, 1)
            Write-Host "Converged after $iteration iteration(s) (${elapsed}m)" -ForegroundColor Green
            $completed += $name
            Add-Content $LogFile "## Completed: $name in $iteration iteration(s) (${elapsed}m)`n"
            $status = "converged"
            Commit-Changes "$name - automated iteration"
            break
        }

        Add-Content $LogFile "### $name - iteration $iteration`n"
    }

    if ($status -eq "abandoned") {
        Write-Host "Hit max iterations ($MaxIterations) for: $name" -ForegroundColor Yellow
        Add-Content $LogFile "## Abandoned: $name after $MaxIterations iterations`n"
        Commit-Changes "$name - partial (hit max iterations)"
    }

    $results += @{ Name = $name; Status = $status }
}

$totalElapsed = [math]::Round(((Get-Date) - $scriptStart).TotalMinutes, 1)
Write-Host "`n=== Summary (${totalElapsed}m) ===" -ForegroundColor Magenta
foreach ($r in $results) {
    $color = switch ($r.Status) { "converged" { "Green" } "failed" { "Red" } default { "Yellow" } }
    Write-Host "  $($r.Status.PadRight(10)) $($r.Name)" -ForegroundColor $color
}
