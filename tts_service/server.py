# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io, os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from tts_backend import XTTSService

from fastapi import Response


app = FastAPI(title="Velouria Local TTS", version="1.0.0")
svc: XTTSService = XTTSService("settings.json")

class SynthesizeRequest(BaseModel):
    text: str
    speaker_wav: Optional[str] = None
    speaker: Optional[str] = None
    sanitize: Optional[bool] = True

@app.get("/health")
def health():
    return {"status":"ok","model":svc.model_name,"gpu_active":svc.using_gpu,"sample_rate":svc.sample_rate_model}

@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    try:
        text = req.text if (req.sanitize is None or req.sanitize) else req.text
        wav_bytes = svc.synth_to_wav_bytes(text=text, speaker_wav=req.speaker_wav, speaker=req.speaker)
        return Response(content=wav_bytes, media_type="audio/wav", headers={"Cache-Control":"no-store"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
