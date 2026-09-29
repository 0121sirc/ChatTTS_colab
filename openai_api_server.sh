#!/usr/bin/env bash
# Control the ChatTTS OpenAI-compatible TTS server.
#
# Used by speech-to-speech via:
#   s2s.sh --tts-url http://127.0.0.1:${PORT:-8091}/v1
#
# Usage: openai_api_server.sh [start|stop|status] [--host HOST] [--port PORT]
#   start   Launch the server in the background (default when no command is given)
#   stop    Gracefully stop it (TERM -> wait -> KILL)
#   status  Report whether it is running and whether /v1/health responds
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"

# Default bind host: the tailscale IP, else fall back to 127.0.0.1 with a warning.
default_host() {
  local ip=""
  ip="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  if [[ -z "$ip" ]]; then
    ip="$(ip -4 -o addr show tailscale0 2>/dev/null | awk '{split($4,a,"/"); print a[1]}' || true)"
  fi
  if [[ -z "$ip" ]]; then
    echo "WARNING: no tailscale IP found (tailscale not running?); falling back to 127.0.0.1" >&2
    echo "127.0.0.1"
  else
    echo "$ip"
  fi
}

HOST="${HOST:-$(default_host)}"
PORT="${PORT:-8091}"
PYTHON="$HERE/.conda_env/bin/python"
RUN_DIR="$HERE/.run"
PID_FILE="$RUN_DIR/openai_api_server.pid"
LOG_FILE="$RUN_DIR/openai_api_server.log"

# 0.0.0.0 is a bind address, not a dialable one.
HEALTH_HOST="$HOST"
[[ "$HEALTH_HOST" == "0.0.0.0" ]] && HEALTH_HOST="127.0.0.1"

usage() {
  cat <<EOF
Usage: $(basename "$0") [start|stop|status] [--host HOST] [--port PORT]

  start    Start the ChatTTS OpenAI-compatible TTS server in the background
           (default). Waits until /v1/health is ready.
  stop     Gracefully stop the server (TERM -> wait -> KILL fallback).
  status   Show whether the server is running and healthy.

Options:
  --host HOST   Bind host (default: $HOST)
  --port PORT   Bind port (default: $PORT)
  -h, --help    Show this help
EOF
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

health_ok() {
  curl -fsS --max-time 3 --noproxy '*' "http://$HEALTH_HOST:$PORT/v1/health" >/dev/null 2>&1
}

# True when something is listening on a TCP port -- catches a server that was
# started by hand, since these scripts may bind a tailscale IP rather than
# 127.0.0.1. Same helper as the CosyVoice/Qwen3-TTS launchers.
#
# health_ok() alone cannot tell our own server from a foreign one bound to the
# same port, so the readiness loop used to report "Up" for a pid that never got
# the socket.
port_listening() {
  ss -H -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":$1$"
}

cmd_start() {
  if is_running; then
    if health_ok; then
      echo "Already running (pid $(cat "$PID_FILE")) on http://$HOST:$PORT"
      return 0
    fi
    echo "Stale process $(cat "$PID_FILE") without a healthy endpoint; restarting."
    cmd_stop
  fi

  # .conda_env/ is gitignored (it *is* the conda env), so a fresh clone ships no
  # interpreter at this path. Fail with the recipe instead of letting nohup die
  # quietly in the log.
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python interpreter not found at: $PYTHON" >&2
    echo "  The conda env (.conda_env/) is not part of this checkout. Create it, then retry:" >&2
    echo "      conda create -p .conda_env python=3.11 pip" >&2
    echo "      .conda_env/bin/pip install -r requirements.txt" >&2
    echo "  re-run: ./openai_api_server.sh start" >&2
    return 1
  fi

  # Refuse to start when the port is already taken: health_ok() would then answer
  # for a process we did not start, and the readiness loop below could report
  # "Up (pid $!)" for a pid that never bound the socket.
  if port_listening "$PORT"; then
    echo "ERROR: port $PORT is already in use; refusing to start a second server." >&2
    ss -tlnp "sport = :$PORT" 2>/dev/null | sed 's/^/    /' >&2 || true
    echo "  stop the owner first: ./openai_api_server.sh stop" >&2
    return 1
  fi

  mkdir -p "$RUN_DIR"
  : > "$LOG_FILE"
  echo "Starting ChatTTS OpenAI TTS server on http://$HOST:$PORT ..."
  nohup "$PYTHON" "$HERE/openai_tts_server.py" --host "$HOST" --port "$PORT" \
    >>"$LOG_FILE" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"

  # Liveness before health: a pid that never bound the socket must lose even if
  # another process answers on this port.
  for _ in $(seq 1 120); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: server exited during startup. Last log lines:" >&2
      tail -20 "$LOG_FILE" >&2 || true
      rm -f "$PID_FILE"
      return 1
    fi
    if health_ok; then
      echo "Up (pid $pid). Log: $LOG_FILE"
      return 0
    fi
    sleep 1
  done

  echo "WARNING: started (pid $pid) but /v1/health is not ready yet; see $LOG_FILE"
  return 0
}

cmd_stop() {
  if ! is_running; then
    rm -f "$PID_FILE"
    echo "Not running."
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  echo "Stopping pid $pid ..."
  kill -TERM "$pid" 2>/dev/null || true

  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "  force killing $pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "Stopped."
}

cmd_status() {
  if is_running; then
    echo "Running (pid $(cat "$PID_FILE")) on http://$HOST:$PORT"
    if health_ok; then
      echo "Health: ok"
    else
      echo "Health: FAILED (endpoint not responding)"
      return 1
    fi
  else
    echo "Not running."
    return 1
  fi
}

CMD="start"
while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|status) CMD="$1"; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# HEALTH_HOST/PORT may have changed via options.
HEALTH_HOST="$HOST"
[[ "$HEALTH_HOST" == "0.0.0.0" ]] && HEALTH_HOST="127.0.0.1"

case "$CMD" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
esac
