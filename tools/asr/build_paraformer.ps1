param(
    [string]$Compiler = 'gcc',
    [Parameter(Mandatory=$true)][string]$Output,
    [switch]$OpenMP,
    [ValidateSet('cli', 'static')][string]$Target = 'cli',
    [string]$Archiver,
    [string[]]$ExtraFlags = @()
)
$ErrorActionPreference = 'Stop'
$source = Join-Path $PSScriptRoot '../../src/model/vllm_paraformer.c'
$flags = @('-O2', '-std=c11', '-Wall', '-Wextra') + $ExtraFlags
if ($OpenMP) { $flags += '-fopenmp' }
$flags += @('-fno-fast-math', '-ffp-contract=off')
$machine = & $Compiler -dumpmachine
if ($LASTEXITCODE -ne 0) { throw 'Cannot determine compiler target' }
Write-Host "[paraformer] target=$Target machine=$machine output=$Output"
if ($Target -eq 'cli') {
    Write-Host "$Compiler $flags $source -o $Output -lm"
    & $Compiler @flags $source '-o' $Output '-lm'
    if ($LASTEXITCODE -ne 0) { throw "Paraformer build failed: $LASTEXITCODE" }
} else {
    if (-not $Archiver) {
        $Archiver = & $Compiler '-print-prog-name=ar'
        if ($LASTEXITCODE -ne 0) { throw 'Cannot determine archiver' }
    }
    $object = Join-Path ([IO.Path]::GetTempPath()) ("paraformer-" + [guid]::NewGuid().ToString('N') + '.o')
    $archive = "$object.a"
    try {
        Write-Host "$Compiler $flags -DVLLM_PARAFORMER_NO_MAIN -c $source -o $object"
        & $Compiler @flags '-DVLLM_PARAFORMER_NO_MAIN' '-c' $source '-o' $object
        if ($LASTEXITCODE -ne 0) { throw "Paraformer compile failed: $LASTEXITCODE" }
        Write-Host "$Archiver rcs $archive $object"
        & $Archiver rcs $archive $object
        if ($LASTEXITCODE -ne 0) { throw "Paraformer archive failed: $LASTEXITCODE" }
        Move-Item -LiteralPath $archive -Destination $Output -Force
    } finally {
        Remove-Item -LiteralPath $object, $archive -Force -ErrorAction SilentlyContinue
    }
    Write-Host '[paraformer] consumers must link libm (and OpenMP when enabled).'
}
