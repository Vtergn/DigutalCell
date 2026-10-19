param([switch]$Execute)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path.TrimEnd('\')
$analysisRoot = Join-Path $projectRoot 'analysis'
$targets = @()
$targets += @(Get-ChildItem -LiteralPath $analysisRoot -Directory -Recurse -Force |
    Where-Object { $_.Name -eq '__pycache__' })
$targets += @(Get-Item -LiteralPath (Join-Path $analysisRoot 'shape_feedback\sarcomere_closed_loop_viewer\chrome_profile') -ErrorAction SilentlyContinue)
$targets += @(Get-Item -LiteralPath (Join-Path $projectRoot 'tmp\papers') -ErrorAction SilentlyContinue)
$targets += @(Get-Item -LiteralPath (Join-Path $projectRoot 'tmp\pdfs') -ErrorAction SilentlyContinue)
$targets = @($targets | Where-Object { $null -ne $_ } | Sort-Object FullName -Unique)

$totalBytes = 0L
foreach ($target in $targets) {
    $resolved = (Resolve-Path -LiteralPath $target.FullName).Path
    $inside = $resolved.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)
    $isCache = $target.Name -eq '__pycache__' -and
        $resolved.StartsWith($analysisRoot + '\', [StringComparison]::OrdinalIgnoreCase)
    $isBrowser = $resolved -eq (Join-Path $analysisRoot 'shape_feedback\sarcomere_closed_loop_viewer\chrome_profile')
    $isPaperTemp = $resolved -eq (Join-Path $projectRoot 'tmp\papers') -or
        $resolved -eq (Join-Path $projectRoot 'tmp\pdfs')
    if (-not $inside -or -not ($isCache -or $isBrowser -or $isPaperTemp) -or
        ($target.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Unsafe cleanup target: $resolved"
    }
    $files = @(Get-ChildItem -LiteralPath $resolved -Recurse -File -Force -ErrorAction Stop)
    $bytes = ($files | Measure-Object Length -Sum).Sum
    $totalBytes += $bytes
    Write-Output ("{0}: {1} files, {2:N2} MB" -f $resolved, $files.Count, ($bytes / 1MB))
    if ($Execute) {
        Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction Stop
    }
}
$mode = if ($Execute) { 'deleted' } else { 'dry-run' }
Write-Output ("{0}: {1} directories, {2:N2} MB" -f $mode, $targets.Count, ($totalBytes / 1MB))

