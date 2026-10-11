#!/bin/sh
# Stop server/clef_server.py on the Windows GPU box.
: "${CLEF_HOST:?set CLEF_HOST=user@windows-box}"
ssh "$CLEF_HOST" 'powershell -NoProfile -Command "Get-CimInstance Win32_Process | ? { $_.Name -eq \"python.exe\" -and $_.CommandLine -like \"*clef_server.py*\" } | % { Stop-Process -Id $_.ProcessId -Force; \"stopped clef_server.py (pid $($_.ProcessId))\" }"'
