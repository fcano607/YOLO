<# Read-only regression checks for the cleanup path guard on PowerShell 5.1 and 7. #>
$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
$taskScriptPath = Join-Path $taskRoot 'scripts/cleanup_project.ps1'
$taskTokens = $null
$taskParseErrors = $null
$taskAst = [System.Management.Automation.Language.Parser]::ParseFile(
    $taskScriptPath, [ref]$taskTokens, [ref]$taskParseErrors)
if ($taskParseErrors.Count -ne 0) { throw 'Cleanup script has syntax errors' }

# Load only real guard functions; never invoke the script's deletion body.
foreach ($taskFunction in $taskAst.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst]
}, $true)) {
    . ([scriptblock]::Create($taskFunction.Extent.Text))
}

$taskRelative = 'reports/maintenance/20261004_M2_cleanup_manifest.json'
$taskExpected = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $taskRelative))
if ((Workspace-Path $taskRelative) -ne $taskExpected) {
    throw 'The actual cloud-marked manifest must resolve inside the project'
}
$taskChecks = 1
foreach ($taskRejectedPath in @('../outside.txt', '.git/config', $taskExpected)) {
    $taskRejected = $false
    try { $null = Workspace-Path $taskRejectedPath } catch { $taskRejected = $true }
    if (-not $taskRejected) { throw ('Unsafe path accepted: ' + $taskRejectedPath) }
    $taskChecks++
}

# Links, junctions and unfamiliar filters remain rejected independently of shell version.
foreach ($taskAcceptedTag in @('00000000', '9000601a')) {
    Assert-CleanupReparseTag ([Convert]::ToUInt32($taskAcceptedTag, 16)) $taskExpected
    $taskChecks++
}
foreach ($taskRejectedTag in @('a000000c', 'a0000003', '8000001b', '9000501a')) {
    $taskRejected = $false
    try {
        Assert-CleanupReparseTag ([Convert]::ToUInt32($taskRejectedTag, 16)) $taskExpected
    } catch { $taskRejected = $true }
    if (-not $taskRejected) { throw ('Unsafe reparse tag accepted: ' + $taskRejectedTag) }
    $taskChecks++
}
Write-Output ('Cleanup path regression checks passed: ' + $taskChecks +
    '; PowerShell ' + $PSVersionTable.PSVersion.ToString())
