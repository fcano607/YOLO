<#
Review the prepared cleanup by default. -Execute performs the explicitly requested cleanup.
Only a prepared stage manifest is accepted; raw truth, models, source and Git history are protected.
#>
[CmdletBinding()]
param(
    [string]$Manifest = 'reports/maintenance/20261004_M2_cleanup_manifest.json',
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))

function Get-CleanupReparseTag([string]$path) {
    # PowerShell 5.1 and 7 expose cloud attributes differently. Read the native tag.
    if (-not ('YoloCleanup.PathMetadata' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
namespace YoloCleanup {
    public static class PathMetadata {
        [StructLayout(LayoutKind.Sequential)]
        private struct AttributeTagInfo {
            public uint Attributes;
            public uint ReparseTag;
        }
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern SafeFileHandle CreateFileW(
            string path, uint access, uint share, IntPtr security,
            uint disposition, uint flags, IntPtr template);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetFileInformationByHandleEx(
            SafeFileHandle handle, int infoClass, out AttributeTagInfo info, uint size);
        public static uint ReadTag(string path) {
            // OPEN_EXISTING; no data access; open the reparse point itself, including directories.
            using (SafeFileHandle handle = CreateFileW(@"\\?\" + path, 0, 7,
                IntPtr.Zero, 3, 0x02200000, IntPtr.Zero)) {
                if (handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error());
                AttributeTagInfo info;
                if (!GetFileInformationByHandleEx(handle, 9, out info, 8))
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                return (info.Attributes & 0x400) == 0 ? 0 : info.ReparseTag;
            }
        }
    }
}
'@
    }
    return [YoloCleanup.PathMetadata]::ReadTag($path)
}

function Assert-CleanupReparseTag([uint32]$tag, [string]$path) {
    # Only the observed CLOUD_6 tag is allowed. It has no name-surrogate bit.
    # Symlinks, junctions, mount points and unknown filters remain rejected.
    $cloud6Tag = [Convert]::ToUInt32('9000601a', 16)
    if ($tag -ne 0 -and $tag -ne $cloud6Tag) {
        throw ('Reparse path is not eligible: ' + $path + ' (tag 0x' + $tag.ToString('x8') + ')')
    }
}

function Workspace-Path([string]$relativePath) {
    if ([System.IO.Path]::IsPathRooted($relativePath)) { throw 'Expected a project-relative path' }
    $targetPath = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $relativePath))
    if (-not $targetPath.StartsWith($taskRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Target outside the project'
    }
    if ($relativePath -match '(^|[/\\])\.git([/\\]|$)') { throw 'Git internals are protected' }
    $ancestorPath = $targetPath
    while ($ancestorPath.Length -ge $taskRoot.Length) {
        if (Test-Path -LiteralPath $ancestorPath) {
            Assert-CleanupReparseTag (Get-CleanupReparseTag $ancestorPath) $ancestorPath
        }
        if ($ancestorPath -eq $taskRoot) { break }
        $ancestorPath = Split-Path $ancestorPath -Parent
    }
    return $targetPath
}

function File-Sha([string]$path) {
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$manifestPath = Workspace-Path $Manifest
$plan = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($plan.root -ne $taskRoot -or $plan.task -notin @('M2_post_acceptance_cleanup', 'M3_post_baseline_cleanup')) {
    throw 'Manifest belongs to another project or task'
}
if ($plan.status -notin @('archive_and_organization_done_deletion_blocked', 'prepared_cleanup', 'manual_cleanup_verified', 'cleanup_verified')) {
    throw 'Unexpected cleanup state'
}
if ($plan.status -in @('manual_cleanup_verified', 'cleanup_verified')) {
    Write-Output 'The reviewed cleanup has already completed.'
    return
}
$archivePath = Workspace-Path $plan.archive.path
if ((File-Sha $archivePath) -ne $plan.archive.sha256) { throw 'Verified archive changed' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    if ($archive.Entries.Count -ne $plan.archive.original_files.Count + 1) { throw 'Archive entry count changed' }
    foreach ($original in $plan.archive.original_files) {
        $entry = $archive.GetEntry($original.path)
        if ($null -eq $entry -or $entry.Length -ne $original.bytes) { throw 'Archive original missing or resized' }
        $stream = $entry.Open()
        $algorithm = [System.Security.Cryptography.SHA256]::Create()
        try {
            $entrySha = [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
            if ($entrySha -ne $original.sha256) { throw ('Archived bytes changed: ' + $original.path) }
        } finally { $stream.Dispose(); $algorithm.Dispose() }
    }
} finally { $archive.Dispose() }
$preservedNames = @($plan._preserved_sha256.PSObject.Properties.Name)
$protectedNames = @($plan.protected_inputs.PSObject.Properties.Name)
$seenPaths = @{}

# Complete every validation before deleting any file.
foreach ($candidate in $plan.candidates) {
    $path = Workspace-Path $candidate.path
    if ($seenPaths.ContainsKey($path)) { throw 'Repeated cleanup path' }
    $seenPaths[$path] = $true
    if ($candidate.path -in $preservedNames -or $candidate.path -in $protectedNames) {
        throw ('Protected input appears in deletion list: ' + $candidate.path)
    }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (File-Sha $path) -ne $candidate.sha256) {
        throw ('Candidate missing or changed: ' + $candidate.path)
    }
    switch ($candidate.kind) {
        'regenerable_bytecode' {
            if ($candidate.path -notmatch '/__pycache__/[^/]+\.py[co]$' -or
                -not (Test-Path -LiteralPath (Workspace-Path $candidate.retained_source) -PathType Leaf)) {
                throw 'Bytecode has no retained Python source'
            }
        }
        'regenerable_label_cache' {
            if ($candidate.path -notmatch '^data/desktop/labels/.+\.cache$') { throw 'Unexpected label cache' }
        }
        'exact_duplicate' {
            if ((File-Sha (Workspace-Path $candidate.retained_identical_file)) -ne $candidate.sha256) {
                throw 'Retained identical copy changed'
            }
        }
        'archived_unique_history' {
            $entry = @($plan.archive.original_files | Where-Object { $_.path -eq $candidate.path })
            if ($entry.Count -ne 1 -or $entry[0].sha256 -ne $candidate.sha256 -or -not $plan.archive.sha256_verified) {
                throw 'History is not in the verified archive index'
            }
        }
        default { throw 'Unapproved cleanup category' }
    }
}
foreach ($entry in $plan._preserved_sha256.PSObject.Properties) {
    if ((File-Sha (Workspace-Path $entry.Name)) -ne $entry.Value) {
        throw ('Preserved input changed: ' + $entry.Name)
    }
}
foreach ($move in $plan.moves) {
    if ($move.status -ne 'completed_bytes_unchanged' -or
        (File-Sha (Workspace-Path $move.to)) -ne $move.sha256) { throw 'Relocated checker differs' }
}
foreach ($directory in $plan.empty_directory_candidates) { $null = Workspace-Path $directory }

Write-Output ('Verified candidates: ' + $plan.candidates.Count + '; archive originals: ' + $plan.archive.original_files.Count)
if (-not $Execute) {
    Write-Output 'Review only: nothing deleted. Use -Execute locally after reviewing the manifest.'
    return
}

$deletedFiles = @()
foreach ($candidate in $plan.candidates) {
    Remove-Item -LiteralPath (Workspace-Path $candidate.path) -Force
    $deletedFiles += $candidate.path
}
$deletedDirectories = @()
foreach ($directory in $plan.empty_directory_candidates) {
    $path = Workspace-Path $directory
    if ((Test-Path -LiteralPath $path -PathType Container) -and
        -not (Get-ChildItem -LiteralPath $path -Force | Select-Object -First 1)) {
        Remove-Item -LiteralPath $path -Force
        $deletedDirectories += $directory
    }
}
foreach ($entry in $plan._preserved_sha256.PSObject.Properties) {
    if ((File-Sha (Workspace-Path $entry.Name)) -ne $entry.Value) { throw 'A retained file changed after cleanup' }
}
$plan.status = if ($plan.task -eq 'M3_post_baseline_cleanup') { 'cleanup_verified' } else { 'manual_cleanup_verified' }
$plan | Add-Member -NotePropertyName deleted_files -NotePropertyValue $deletedFiles -Force
$plan | Add-Member -NotePropertyName deleted_directories -NotePropertyValue $deletedDirectories -Force
if ($plan.task -eq 'M2_post_acceptance_cleanup') {
    $plan | Add-Member -NotePropertyName manual_executed_at_utc -NotePropertyValue ([DateTimeOffset]::UtcNow.ToString('o')) -Force
}
$plan | Add-Member -NotePropertyName executed_at -NotePropertyValue ([DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(8)).ToString('o')) -Force
$plan | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Output ('Deleted files: ' + $deletedFiles.Count + '; empty directories: ' + $deletedDirectories.Count)
Write-Output 'All retained files match their snapshot. Recheck frozen data and baseline with Python.'
