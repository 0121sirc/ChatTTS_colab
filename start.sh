#!/usr/bin/env bash
# Quick foreground launcher for the ChatTTS WebUI.
# For a managed background instance (start/stop/status, pid file, log file) use
# ./webui.sh from the repo root instead.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PYTHON="$HERE/.conda_env/bin/python"

# .conda_env/ is gitignored (it *is* the conda env), so a fresh clone ships no
# interpreter at this path. Fail with the recipe instead of "command not found".
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python interpreter not found at: $PYTHON" >&2
  echo "  The conda env (.conda_env/) is not part of this checkout. Create it, then retry:" >&2
  echo "      conda create -p .conda_env python=3.11 pip" >&2
  echo "      .conda_env/bin/pip install -r requirements.txt" >&2
  exit 1
fi

exec "$PYTHON" webui_mix_update.py --source custom --local_path models
