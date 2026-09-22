#!/usr/bin/env bash
# Start the ChatTTS OpenAI-compatible TTS server (used by speech-to-speech
# via `s2s.sh --tts-url http://127.0.0.1:${PORT:-8091}/v1`).
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"

exec ./.conda_env/bin/python openai_tts_server.py --host "$HOST" --port "$PORT"
