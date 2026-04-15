param(
    [Parameter(Mandatory = $true)]
    [int]$TargetPid,
    [Parameter(Mandatory = $true)]
    [string]$WatchLog,
    [Parameter(Mandatory = $true)]
    [string]$TrainLog,
    [Parameter(Mandatory = $true)]
    [string]$ModelDir,
    [int]$IntervalSeconds = 900
)

$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Status([string]$Status) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $WatchLog -Value ("[$timestamp] " + $Status) -Encoding UTF8
}

Write-Status "Watcher started for PID=$TargetPid"

while ($true) {
    $proc = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    $logInfo = if (Test-Path $TrainLog) { Get-Item $TrainLog } else { $null }
    $artifacts = @(
        "model_pool.joblib",
        "model_pool_summary.csv",
        "model_pool_dataset_eval.csv",
        "model_pool_config.json"
    ) | Where-Object {
        Test-Path (Join-Path $ModelDir $_)
    }

    if ($proc) {
        $logMsg = if ($logInfo) {
            "train_log_size=$($logInfo.Length), last_write=$($logInfo.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))"
        } else {
            "train_log_missing"
        }
        $artifactMsg = if ($artifacts.Count -gt 0) {
            "artifacts=" + ($artifacts -join ",")
        } else {
            "artifacts=none"
        }
        Write-Status ("RUNNING | " + $logMsg + " | " + $artifactMsg)
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }

    $finalLogMsg = if ($logInfo) {
        "final_log_size=$($logInfo.Length), last_write=$($logInfo.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    } else {
        "train_log_missing"
    }
    $finalArtifactMsg = if ($artifacts.Count -gt 0) {
        "artifacts=" + ($artifacts -join ",")
    } else {
        "artifacts=none"
    }
    Write-Status ("STOPPED | " + $finalLogMsg + " | " + $finalArtifactMsg)
    break
}
