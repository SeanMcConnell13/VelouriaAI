# -*- coding: utf-8 -*-
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Literal
import os
import re
import unicodedata

try:
    import torch
except Exception:
    torch = None

from tts_backend import XTTSService

app = FastAPI(title="Velouria Local TTS", version="0.2.3")

# ---------------------------- Torch fast-path tweaks ----------------------------
if torch is not None:
    try:
        torch.set_grad_enabled(False)
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ---------------------------- Utilities ----------------------------

_EMOJI_RE = re.compile(
    "["                      # conservative, high-signal ranges
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, supplemental
    "\U00002700-\U000027BF"  # dingbats
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE,
)

_ALLOWED_CHARS_RE = re.compile(r"[^a-zA-Z0-9\s\.\,\!\?\;\:\-\'\"/()]+", re.UNICODE)

def sanitize_text(s: str) -> str:
    """Normalize & strip characters that often trigger phonemizer/tokenizer edge cases."""
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    # replace curly quotes/dashes with ASCII equivalents
    s = (s
         .replace("\u201C", '"').replace("\u201D", '"')  # “ ”
         .replace("\u2018", "'").replace("\u2019", "'")  # ‘ ’
         .replace("\u2014", "-").replace("\u2013", "-")) # — –
    # remove emoji blocks
    s = _EMOJI_RE.sub("", s)
    # remove control chars except newlines
    s = "".join(ch for ch in s if ch == "\n" or (unicodedata.category(ch)[0] != "C"))
    # drop exotic symbols; keep basic punctuation
    s = _ALLOWED_CHARS_RE.sub(" ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _torch_info():
    if torch is None:
        return {"has_torch": False, "cuda_available": False, "torch_version": None, "cuda_version": None}
    return {
        "has_torch": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "device_count": torch.cuda.device_count() if hasattr(torch.cuda, "device_count") else None,
        "current_device": torch.cuda.current_device() if hasattr(torch.cuda, "current_device") and torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()) if hasattr(torch.cuda, "get_device_name") and torch and torch.cuda.is_available() else None,
    }

def _service_use_gpu(svc) -> bool:
    if hasattr(svc, "use_gpu"):
        try:
            return bool(getattr(svc, "use_gpu"))
        except Exception:
            pass
    info = _torch_info()
    return bool(info.get("cuda_available"))

def _service_to_device(svc, device: str) -> None:
    if device not in {"cuda", "cpu"}:
        return
    if hasattr(svc, "to"):
        try:
            svc.to(device)
            return
        except Exception:
            pass
    if hasattr(svc, "set_device"):
        try:
            svc.set_device(device)
            return
        except Exception:
            pass
    # Otherwise, XTTSService will remain on whatever it booted with.

# ---------------------------- Service Boot ----------------------------

svc: XTTSService = XTTSService()  # load on boot, keeps model warm

# prefer GPU on boot when available
if _torch_info().get("cuda_available"):
    _service_to_device(svc, "cuda")
print("🟢 TTS model on CUDA" if _service_use_gpu(svc) else "🟡 TTS model on CPU")

# ---------------------------- Schemas ----------------------------

class SynthesizeRequest(BaseModel):
    text: str
    speaker_wav: Optional[str] = None
    speaker: Optional[str] = None
    seed: Optional[int] = None
    device: Optional[Literal["auto", "cuda", "cpu"]] = None
    sanitize: Optional[bool] = True

class DeviceRequest(BaseModel):
    device: Literal["cuda", "cpu"]

# ---------------------------- Endpoints ----------------------------

@app.get("/health")
def health():
    ti = _torch_info()
    return {
        "status": "ok",
        "model": getattr(svc, "model_name", "xtts_v2"),
        "gpu_active": _service_use_gpu(svc),
        "torch": ti,
    }

@app.post("/device")
def set_device(req: DeviceRequest):
    try:
        _service_to_device(svc, req.device)
        return {"ok": True, "device": req.device, "gpu_active": _service_use_gpu(svc)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    """
    Returns a WAV file (FileResponse).
    Device selection is applied via _service_to_device; we do NOT pass 'device' into synth_to_wavfile.
    """
    try:
        text = sanitize_text(req.text) if (req.sanitize is None or req.sanitize) else req.text

        # Only honor device if explicitly passed. Client should not pass device per call.
        if req.device in {"cuda", "cpu"}:
            _service_to_device(svc, req.device)
        elif req.device == "auto":
            _service_to_device(svc, "cuda" if _torch_info().get("cuda_available") else "cpu")

        synth_kwargs = {}
        if req.seed is not None:
            synth_kwargs["seed"] = int(req.seed)

        if torch is not None:
            with torch.inference_mode():
                wav_path = svc.synth_to_wavfile(
                    text=text,
                    speaker_wav=req.speaker_wav,
                    speaker=req.speaker,
                    **synth_kwargs,
                )
        else:
            wav_path = svc.synth_to_wavfile(
                text=text,
                speaker_wav=req.speaker_wav,
                speaker=req.speaker,
                **synth_kwargs,
            )

        return FileResponse(
            wav_path,
            media_type="audio/wav",
            filename=os.path.basename(wav_path),
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/speak")
def speak(
    text: str = Form(...),
    speaker_wav: Optional[str] = Form(None),
    speaker: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    device: Optional[Literal["auto", "cuda", "cpu"]] = Form(None),
    sanitize: Optional[bool] = Form(True),
):
    """
    Server-side playback: synth + play locally (for testing).
    Device selection is applied via _service_to_device; do not pass 'device' into synth_to_wavfile.
    """
    try:
        text_proc = sanitize_text(text) if (sanitize is None or sanitize) else text

        # Device routing
        if device in {"cuda", "cpu"}:
            _service_to_device(svc, device)
        elif device == "auto":
            _service_to_device(svc, "cuda" if _torch_info().get("cuda_available") else "cpu")

        synth_kwargs = {}
        if seed is not None:
            synth_kwargs["seed"] = int(seed)

        if torch is not None:
            with torch.inference_mode():
                wav_path = svc.synth_to_wavfile(
                    text=text_proc,
                    speaker_wav=speaker_wav,
                    speaker=speaker,
                    **synth_kwargs,
                )
        else:
            wav_path = svc.synth_to_wavfile(
                text=text_proc,
                speaker_wav=speaker_wav,
                speaker=speaker,
                **synth_kwargs,
            )

        svc.play_wavfile(wav_path)
        return {"ok": True, "wav_path": wav_path, "gpu_active": _service_use_gpu(svc)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
