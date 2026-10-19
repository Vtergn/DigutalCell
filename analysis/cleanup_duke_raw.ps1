param([switch]$Execute)

$ErrorActionPreference = 'Stop'
$manifestPath = Join-Path $PSScriptRoot 'duke_raw_cleanup_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$rootPath = (Resolve-Path -LiteralPath $manifest.root).Path.TrimEnd('\')
$rootPrefix = $rootPath + '\'
$archivePath = (Resolve-Path -LiteralPath $manifest.archive).Path
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw 'Backup archive is missing.'
}

$validated = [System.Collections.Generic.List[string]]::new()
$bytes = [long]0
foreach ($item in $manifest.items) {
    $relative = [string]$item.relative_path
    if ([IO.Path]::IsPathRooted($relative)) {
        throw "Manifest contains a rooted path: $relative"
    }
    $target = [IO.Path]::GetFullPath((Join-Path $rootPath ($relative.Replace('/', '\'))))
    if (-not $target.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target escapes the Duke extraction directory: $target"
    }
    if ($item.kind -eq 'raw_asc') {
        if (-not $target.EndsWith('.asc', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Unexpected raw ASC file: $target"
        }
    } elseif ($item.kind -eq 'raw_calibration_csv') {
        $name = [IO.Path]::GetFileName($target).ToLowerInvariant()
        if (-not ($name.EndsWith('.csv') -and $name -match 'rawdat|raw_dat|specdat|spec_dat')) {
            throw "Unexpected calibration CSV: $target"
        }
    } else {
        throw "Unexpected manifest kind: $($item.kind)"
    }
    $file = Get-Item -LiteralPath $target -ErrorAction Stop
    if ($file.PSIsContainer -or $file.Length -ne [long]$item.bytes) {
        throw "Target changed after manifest creation: $target"
    }
    $validated.Add($target)
    $bytes += $file.Length
}

Write-Output "Validated $($validated.Count) files; $([math]::Round($bytes / 1GB, 2)) GiB."
if ($Execute) {
    foreach ($target in $validated) {
        Remove-Item -LiteralPath $target -Force -ErrorAction Stop
    }
    Write-Output "Deleted $($validated.Count) raw files."
} else {
    Write-Output 'Dry run only. Use -Execute after archive integrity verification.'
}
