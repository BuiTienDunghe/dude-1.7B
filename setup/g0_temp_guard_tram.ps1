# Trạm giữ lính gác nhiệt (29/10): chạy setup/g0_temp_guard_v2.sh trong distro Ubuntu và tự khởi động lại khi nó thoát mã khác 0
# (ví dụ WSL VM bị tắt bởi wsl --shutdown). Mỗi lần khởi động lại ghi một dòng CANH_BAO_LINH_GAC vào results/G0/logs/temp.log
# và chép stderr của wsl.exe vào results/G0/logs/temp_guard.err. Lính gác thoát mã 0 (đã nguội sau QUA_NHIET) thì trạm dừng, không chạy lại,
# để giữ luật "worker báo điều phối trước khi chạy tiếp".
# Chạy nền, độc lập với session:
#   Start-Process powershell.exe -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','C:\Users\dungbt06\Fine-tune\setup\g0_temp_guard_tram.ps1'
# PID của trạm ghi ở results/G0/logs/linh_gac_tram.pid. Dừng: Stop-Process -Id <pid> rồi pkill -f g0_temp_guard_v2 trong WSL.
$L = 'C:\Users\dungbt06\Fine-tune\results\G0\logs'
$T = Join-Path $L 'temp.log'
$E = Join-Path $L 'temp_guard.err'
$enc = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $L 'linh_gac_tram.pid'), "$PID`n", $enc)
$n = 0
while ($true) {
    $t0 = Get-Date
    $o = [IO.Path]::GetTempFileName(); $e = [IO.Path]::GetTempFileName()
    $p = Start-Process -FilePath 'wsl.exe' -ArgumentList '-d', 'Ubuntu', '-e', 'bash', '/mnt/c/Users/dungbt06/Fine-tune/setup/g0_temp_guard_v2.sh' `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
    $code = $p.ExitCode
    $err = (Get-Content $e -Raw -ErrorAction SilentlyContinue) + (Get-Content $o -Raw -ErrorAction SilentlyContinue)
    Remove-Item $o, $e -ErrorAction SilentlyContinue
    $now = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
    $secs = [int]((Get-Date) - $t0).TotalSeconds
    if ($err) { [IO.File]::AppendAllText($E, "$now wsl.exe (ma $code): $($err.Trim())`n", $enc) }
    if ($code -eq 0) {
        [IO.File]::AppendAllText($T, "$now TRAM_DUNG linh gac thoat ma 0 sau $secs giay (da nguoi sau QUA_NHIET?); tram khong chay lai`n", $enc)
        break
    }
    $n++
    [IO.File]::AppendAllText($T, "$now CANH_BAO_LINH_GAC thoat ma $code sau $secs giay; khoi dong lai lan $n`n", $enc)
    Start-Sleep -Seconds 5
}
