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

HOST="${HOST:-127.0.0.1}"
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

cmd_start() {
  if is_running; then
    if health_ok; then
      echo "Already running (pid $(cat "$PID_FILE")) on http://$HOST:$PORT"
      return 0
    fi
    echo "Stale process $(cat "$PID_FILE") without a healthy endpoint; restarting."
    cmd_stop
  fi

  mkdir -p "$RUN_DIR"
  : > "$LOG_FILE"
  echo "Starting ChatTTS OpenAI TTS server on http://$HOST:$PORT ..."
  nohup "$PYTHON" "$HERE/openai_tts_server.py" --host "$HOST" --port "$PORT" \
    >>"$LOG_FILE" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"

  for _ in $(seq 1 120); do
    if health_ok; then
      echo "Up (pid $pid). Log: $LOG_FILE"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: server exited during startup. Last log lines:" >&2
      tail -20 "$LOG_FILE" >&2 || true
      rm -f "$PID_FILE"
      return 1
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
