param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRemote,

    [string]$OutputRemote = "",
    [string]$TargetSuffix = "_synth",
    [string]$WorkingDir = ".",
    [switch]$DryRun
)

if ([string]::IsNullOrWhiteSpace($OutputRemote)) {
    $OutputRemote = ($DatasetRemote.TrimEnd('/') -replace '[^/]*$') + "dataset_split"
}

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Join-RemotePath {
    param(
        [string]$Base,
        [string]$Child
    )

    $baseClean = $Base.TrimEnd('/')
    $childClean = $Child.Trim('/')

    if ([string]::IsNullOrWhiteSpace($childClean)) {
        return $baseClean
    }

    return "$baseClean/$childClean"
}

function Get-Key {
    param(
        [string]$RelativePath,
        [string]$SuffixToRemove
    )

    $name = [System.IO.Path]::GetFileNameWithoutExtension($RelativePath)

    if (-not [string]::IsNullOrWhiteSpace($SuffixToRemove) -and
        $name.EndsWith($SuffixToRemove, [System.StringComparison]::OrdinalIgnoreCase)) {
        $name = $name.Substring(0, $name.Length - $SuffixToRemove.Length)
    }

    return $name.ToLowerInvariant()
}

function Write-LinesNoBom {
    param(
        [string]$Path,
        [string[]]$Lines
    )

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($Path, $Lines, $encoding)
}

function Run-RcloneList {
    param([string]$RemotePath)

    $result = @(rclone lsf $RemotePath -R --files-only)
    if ($LASTEXITCODE -ne 0) {
        throw "rclone lsf failed for: $RemotePath"
    }

    return $result
}

$resolvedWorkDir = $WorkingDir
if (-not (Test-Path -LiteralPath $resolvedWorkDir)) {
    New-Item -ItemType Directory -Path $resolvedWorkDir -Force | Out-Null
}
$resolvedWorkDir = (Resolve-Path -LiteralPath $resolvedWorkDir).Path

$srcRemote = Join-RemotePath -Base $DatasetRemote -Child "source"
$tgtRemote = Join-RemotePath -Base $DatasetRemote -Child "target"
$outRemote = $OutputRemote

Write-Host "Source remote: $srcRemote"
Write-Host "Target remote: $tgtRemote"
Write-Host "Output remote: $outRemote"
Write-Host "Working dir:   $resolvedWorkDir"
Write-Host "Suffix rule:   remove '$TargetSuffix' from target basename"
Write-Host ""

$srcRel = Run-RcloneList -RemotePath $srcRemote
$tgtRel = Run-RcloneList -RemotePath $tgtRemote

if ($srcRel.Count -eq 0) {
    throw "No files found in source: $srcRemote"
}
if ($tgtRel.Count -eq 0) {
    throw "No files found in target: $tgtRemote"
}

$srcMap = @{}
foreach ($rel in $srcRel) {
    if ([string]::IsNullOrWhiteSpace($rel)) { continue }
    $key = Get-Key -RelativePath $rel -SuffixToRemove ""
    if (-not $srcMap.ContainsKey($key)) {
        $srcMap[$key] = $rel
    }
}

$tgtMap = @{}
foreach ($rel in $tgtRel) {
    if ([string]::IsNullOrWhiteSpace($rel)) { continue }
    $key = Get-Key -RelativePath $rel -SuffixToRemove $TargetSuffix
    if (-not $tgtMap.ContainsKey($key)) {
        $tgtMap[$key] = $rel
    }
}

$matchedKeys = @($srcMap.Keys | Where-Object { $tgtMap.ContainsKey($_) })
$srcOnly = @($srcMap.Keys | Where-Object { -not $tgtMap.ContainsKey($_) }).Count
$tgtOnly = @($tgtMap.Keys | Where-Object { -not $srcMap.ContainsKey($_) }).Count

if ($matchedKeys.Count -eq 0) {
    throw "No matched source/target pairs found. Check naming rule or suffix."
}

$matchedKeys = @($matchedKeys | Get-Random -Count $matchedKeys.Count)

$n = $matchedKeys.Count
$nTrain = [math]::Floor($n * 0.8)
$nVal = [math]::Floor($n * 0.1)
$nTest = $n - $nTrain - $nVal

$trainKeys = @($matchedKeys | Select-Object -First $nTrain)
$valKeys = @($matchedKeys | Select-Object -Skip $nTrain -First $nVal)
$testKeys = @($matchedKeys | Select-Object -Skip ($nTrain + $nVal))

$trainSourceList = Join-Path $resolvedWorkDir "train_source.txt"
$trainTargetList = Join-Path $resolvedWorkDir "train_target.txt"
$valSourceList = Join-Path $resolvedWorkDir "val_source.txt"
$valTargetList = Join-Path $resolvedWorkDir "val_target.txt"
$testSourceList = Join-Path $resolvedWorkDir "test_source.txt"
$testTargetList = Join-Path $resolvedWorkDir "test_target.txt"

Write-LinesNoBom -Path $trainSourceList -Lines @($trainKeys | ForEach-Object { $srcMap[$_] })
Write-LinesNoBom -Path $trainTargetList -Lines @($trainKeys | ForEach-Object { $tgtMap[$_] })
Write-LinesNoBom -Path $valSourceList -Lines @($valKeys | ForEach-Object { $srcMap[$_] })
Write-LinesNoBom -Path $valTargetList -Lines @($valKeys | ForEach-Object { $tgtMap[$_] })
Write-LinesNoBom -Path $testSourceList -Lines @($testKeys | ForEach-Object { $srcMap[$_] })
Write-LinesNoBom -Path $testTargetList -Lines @($testKeys | ForEach-Object { $tgtMap[$_] })

Write-Host "Matched pairs: $n"
Write-Host "Source only:   $srcOnly"
Write-Host "Target only:   $tgtOnly"
Write-Host "Train:         $nTrain"
Write-Host "Val:           $nVal"
Write-Host "Test:          $nTest"
Write-Host ""

function Copy-Split {
    param(
        [string]$SplitName,
        [string]$SourceList,
        [string]$TargetList
    )

    $splitSourceRemote = Join-RemotePath -Base $outRemote -Child "$SplitName/source"
    $splitTargetRemote = Join-RemotePath -Base $outRemote -Child "$SplitName/target"

    if ($DryRun) {
        rclone copy $srcRemote $splitSourceRemote --files-from $SourceList -P --dry-run
        if ($LASTEXITCODE -ne 0) { throw "rclone dry-run copy failed for $SplitName/source" }

        rclone copy $tgtRemote $splitTargetRemote --files-from $TargetList -P --dry-run
        if ($LASTEXITCODE -ne 0) { throw "rclone dry-run copy failed for $SplitName/target" }
    }
    else {
        rclone copy $srcRemote $splitSourceRemote --files-from $SourceList -P
        if ($LASTEXITCODE -ne 0) { throw "rclone copy failed for $SplitName/source" }

        rclone copy $tgtRemote $splitTargetRemote --files-from $TargetList -P
        if ($LASTEXITCODE -ne 0) { throw "rclone copy failed for $SplitName/target" }
    }
}

Copy-Split -SplitName "train" -SourceList $trainSourceList -TargetList $trainTargetList
Copy-Split -SplitName "val" -SourceList $valSourceList -TargetList $valTargetList
Copy-Split -SplitName "test" -SourceList $testSourceList -TargetList $testTargetList

Write-Host ""
Write-Host "Done. Split created at: $outRemote"
Write-Host "List files saved in: $resolvedWorkDir"