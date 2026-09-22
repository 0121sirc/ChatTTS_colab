"""OpenAI-compatible TTS server for ChatTTS.

Exposes ``POST /v1/audio/speech`` so the ``speech-to-speech`` pipeline can use
ChatTTS through its ``--tts openai`` backend (``--tts-url`` in s2s.sh), reusing
this repo's local weights (``models/``) and saved voices (``voice_pt/*.pt``).

Endpoints:
  GET  /v1/health        -> readiness probe
  GET  /v1/audio/voices  -> {"voices": [...], "default": "seed:1688"}
  POST /v1/audio/speech  -> raw PCM16 (default) or WAV

Request body (OpenAI shape + two non-standard fields)::

    {
      "model": "chattts",
      "input": "要合成的文本",
      "voice": "知心小姐姐" | "seed:1688" | "default",
      "seed": 1688,
      "response_format": "pcm" | "wav",
      "speed": 1.0,
      "params": {"speed": 5, "oral": 0, "laugh": 0, "break": 2,
                 "temperature": 0.1, "top_P": 0.7, "top_K": 20}
    }

Voice resolution order: explicit ``seed`` > ``voice="seed:<n>"`` (or a bare
integer) > ``voice_pt/<name>.pt`` > default seed 1688.
"""

from __future__ import annotations

import argparse
import logging
import sys
import wave
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Iterator

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ChatTTS  # noqa: E402  (local package next to this file)
from config import (  # noqa: E402
    DEFAULT_BK,
    DEFAULT_LAUGH,
    DEFAULT_ORAL,
    DEFAULT_SPEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from tts_model import deterministic, load_chat_tts_model  # noqa: E402
from utils import replace_tokens, restore_tokens, split_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("chattts.openai")

SAMPLE_RATE = 24000
MODELS_DIR = HERE / "models"
VOICE_DIR = HERE / "voice_pt"
DEFAULT_SEED = 1688
DEFAULT_VOICE = f"seed:{DEFAULT_SEED}"

app = FastAPI(title="ChatTTS OpenAI-compatible TTS")

_chat: ChatTTS.Chat | None = None
_load_lock = Lock()


def _ensure_loaded() -> "ChatTTS.Chat":
    global _chat
    if _chat is None:
        with _load_lock:
            if _chat is None:
                logger.info("Loading ChatTTS models from %s", MODELS_DIR)
                _chat = load_chat_tts_model(source="custom", local_path=str(MODELS_DIR))
                logger.info("ChatTTS models loaded")
    return _chat


def _load_speaker_tensor(path: Path) -> torch.Tensor:
    try:
        emb = torch.load(path)
    except Exception:
        # Older/newer torch defaults differ on weights_only; the saved file is a
        # plain embedding tensor, so a full unpickle is safe here.
        emb = torch.load(path, weights_only=False)
    if tuple(emb.shape) != (768,):
        raise HTTPException(status_code=400, detail=f"voice file {path.name} has shape {tuple(emb.shape)}, expected (768,)")
    return emb


def _resolve_speaker(voice: str | None, seed: int | None) -> tuple[torch.Tensor, str]:
    if seed is not None:
        deterministic(int(seed))
        return _ensure_loaded().sample_random_speaker(), f"seed:{int(seed)}"

    value = (voice or "").strip()
    if not value or value.lower() == "default":
        deterministic(DEFAULT_SEED)
        return _ensure_loaded().sample_random_speaker(), DEFAULT_VOICE

    if value.lower().startswith("seed:"):
        raw = value.split(":", 1)[1].strip()
        if not raw.lstrip("-").isdigit():
            raise HTTPException(status_code=400, detail=f"invalid seed voice {value!r}")
        deterministic(int(raw))
        return _ensure_loaded().sample_random_speaker(), f"seed:{int(raw)}"

    if value.lstrip("-").isdigit():
        deterministic(int(value))
        return _ensure_loaded().sample_random_speaker(), f"seed:{int(value)}"

    path = VOICE_DIR / (value if value.endswith(".pt") else f"{value}.pt")
    if path.is_file():
        return _load_speaker_tensor(path), value

    raise HTTPException(status_code=400, detail=f"unknown voice {value!r}")


def _build_params(emb: torch.Tensor, req: "SpeechRequest") -> tuple[dict, dict, bool]:
    extra = dict(req.params or {})
    speed = int(extra.get("speed", DEFAULT_SPEED))
    temperature = float(extra.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(extra.get("top_P", DEFAULT_TOP_P))
    top_k = int(extra.get("top_K", DEFAULT_TOP_K))

    params_infer_code = {
        "spk_emb": emb,
        "prompt": f"[speed_{speed}]",
        "temperature": temperature,
        "top_P": top_p,
        "top_K": top_k,
    }

    oral = int(extra.get("oral", DEFAULT_ORAL))
    laugh = int(extra.get("laugh", DEFAULT_LAUGH))
    bk = int(extra.get("break", DEFAULT_BK))
    params_refine_text = {
        "prompt": f"[oral_{oral}][laugh_{laugh}][break_{bk}]",
        "temperature": temperature,
        "top_P": top_p,
        "top_K": top_k,
    }

    # Prosody only takes effect through text refinement; keep the fast path when
    # the caller did not ask for it.
    skip_refine_text = not any(key in extra for key in ("oral", "laugh", "break"))
    return params_infer_code, params_refine_text, skip_refine_text


def _split(text: str) -> list[str]:
    cleaned = replace_tokens(text)
    pieces = [restore_tokens(piece) for piece in split_text(cleaned, min_length=80)]
    return [piece for piece in pieces if piece.strip()] or [text]


def _to_pcm16(audio: np.ndarray) -> np.ndarray:
    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (samples * 32767.0).astype("<i2")


def _synth_full(text: str, infer_code: dict, refine_text: dict, skip_refine: bool) -> np.ndarray:
    chat = _ensure_loaded()
    parts: list[np.ndarray] = []
    for piece in _split(text):
        wavs = chat.infer(
            [piece],
            params_infer_code=dict(infer_code),
            params_refine_text=dict(refine_text),
            use_decoder=True,
            skip_refine_text=skip_refine,
        )
        for wav in wavs:
            parts.append(np.asarray(wav).reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def _stream_pcm(text: str, infer_code: dict, refine_text: dict, skip_refine: bool) -> Iterator[bytes]:
    chat = _ensure_loaded()
    for piece in _split(text):
        generator = chat.infer(
            piece,
            params_infer_code=dict(infer_code),
            params_refine_text=dict(refine_text),
            use_decoder=True,
            skip_refine_text=skip_refine,
            stream=True,
        )
        for chunk in generator:
            audio = np.asarray(chunk[0]).reshape(-1)
            pcm = _to_pcm16(audio)
            if pcm.size:
                yield pcm.tobytes()


def _wav_bytes(audio: np.ndarray) -> bytes:
    pcm = _to_pcm16(audio)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    seed: int | None = None
    response_format: str = "pcm"
    speed: float | None = None
    params: dict | None = None


@app.get("/v1/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "model_loaded": _chat is not None, "sample_rate": SAMPLE_RATE})


@app.get("/v1/audio/voices")
def voices() -> JSONResponse:
    names = sorted(path.stem for path in VOICE_DIR.glob("*.pt"))
    return JSONResponse({"voices": names, "default": DEFAULT_VOICE})


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty input")

    fmt = (req.response_format or "pcm").lower()
    if fmt not in {"pcm", "wav"}:
        raise HTTPException(status_code=400, detail="response_format must be 'pcm' or 'wav'")

    emb, label = _resolve_speaker(req.voice, req.seed)
    infer_code, refine_text, skip_refine = _build_params(emb, req)
    logger.info("synthesizing %d chars (voice=%s, format=%s)", len(text), label, fmt)

    if fmt == "pcm":
        return StreamingResponse(_stream_pcm(text, infer_code, refine_text, skip_refine), media_type="audio/pcm")

    audio = _synth_full(text, infer_code, refine_text, skip_refine)
    if audio.size == 0:
        raise HTTPException(status_code=500, detail="synthesis produced no audio")
    return Response(content=_wav_bytes(audio), media_type="audio/wav")


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatTTS OpenAI-compatible TTS server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()

    _ensure_loaded()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
