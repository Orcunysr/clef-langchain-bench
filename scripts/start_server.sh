#!/bin/sh
# Start server/clef_server.py on the Windows GPU box and forward localhost:30000 to it.
# Ctrl+C closes the tunnel and stops the server (Windows OpenSSH does not do that on its own).
#
#   export CLEF_HOST=user@windows-box            # required
#   export CLEF_REMOTE_DIR='%USERPROFILE%\clef'  # where server/*.py was copied (default shown)
#   scripts/start_server.sh
set -e
: "${CLEF_HOST:?set CLEF_HOST=user@windows-box}"
REMOTE_DIR="${CLEF_REMOTE_DIR:-%USERPROFILE%\\clef}"
DIR="$(cd "$(dirname "$0")" && pwd)"

"$DIR/stop_server.sh"  # clean up a server left over from an earlier run
trap '"$DIR/stop_server.sh"' EXIT INT TERM

ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 30000:127.0.0.1:30000 "$CLEF_HOST" \
  "cd /d $REMOTE_DIR && set PYTHONIOENCODING=utf-8&& set HF_HUB_OFFLINE=1&& python -u clef_server.py"
