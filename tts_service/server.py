# -*- coding: utf-8 -*-
import sys, pathlib, traceback
from typing import Optional, Union, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# Make sure local imports work (same as your original)
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from tts_backend import XTTSService  # noqa: E402


# ----- FastAPI app + service -------------------------------------------------

app = FastAPI(title="Velouria Local TTS", version="1.0.0")
svc: XTTSService = XTTSService("settings.json")


# ----- Request model ---------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text: str
    # Accept single path OR list of paths
    speaker_wav: Optional[Union[str, List[str]]] = None
    speaker: Optional[str] = None
    sanitize: Optional[bool] = True


# ----- Routes ----------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": svc.model_name,
        "gpu_active": svc.using_gpu,
        "sample_rate": svc.sample_rate_model,
    }


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    try:
        # (If you ever want to bypass sanitization, set sanitize=False in the request)
        text = req.text if (req.sanitize is None or req.sanitize) else req.text
        wav_bytes = svc.synth_to_wav_bytes(
            text=text,
            speaker_wav=req.speaker_wav,  # string OR list supported
            speaker=req.speaker,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )
    except Exception as e:
        # Print the full traceback to the server console and return a 400 with the message
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=400)
