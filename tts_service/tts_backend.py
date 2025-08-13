# -*- coding: utf-8 -*-
import json, os, threading, io
from typing import Optional
import numpy as np
import soundfile as sf

try:
    import torch
except Exception:
    torch = None

from TTS.api import TTS

try:
    import librosa
    _LIBROSA_OK = True
except Exception:
    _LIBROSA_OK = False


class XTTSService:
    def __init__(self, settings_path: str = "settings.json"):
        with open(settings_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.model_name = cfg.get("model_name", "tts_models/multilingual/multi-dataset/xtts_v2")
        self.language = cfg.get("language", "en")
        self.sample_rate_cfg = int(cfg.get("sample_rate", 24000))
        self.default_speaker = (cfg.get("speaker") or "").strip()
        self.reference_voice = (cfg.get("reference_voice_wav") or "").strip()

        post = cfg.get("post") or {}
        self.pitch_semitones = int(post.get("pitch_semitones", 4) or 4)
        self.speed = float(post.get("speed", 1.06) or 1.06)
        self.normalize = bool(post.get("normalize", True))

        self.tts = TTS(self.model_name)
        self._device = "cpu"
        self._synth_lock = threading.Lock()

        if torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
            self.to("cuda")
        else:
            self.to("cpu")

        if torch is not None:
            try: self.tts.eval()
            except Exception: pass

        sr = None
        try: sr = int(getattr(self.tts, "synthesizer").output_sample_rate)
        except Exception: pass
        self.sample_rate_model = sr or self.sample_rate_cfg

    @property
    def using_gpu(self) -> bool:
        return self._device == "cuda"

    def to(self, device: str = "cpu"):
        target = "cuda" if (device == "cuda" and torch is not None and getattr(torch.cuda, "is_available", lambda: False)()) else "cpu"
        try:
            self.tts.to(target); self._device = target
        except Exception:
            self._device = "cpu"
            try: self.tts.to("cpu")
            except Exception: pass

    def _synthesize_array(self, *, text: str, speaker: Optional[str], speaker_wav: Optional[str], seed: Optional[int]) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text is empty.")


        if speaker_wav and os.path.exists(speaker_wav):
            return self.tts.tts(text=text, speaker_wav=speaker_wav, language=self.language)


        if self.reference_voice and os.path.exists(self.reference_voice):
            return self.tts.tts(text=text, speaker_wav=self.reference_voice, language=self.language)


        spk = (speaker or self.default_speaker or "").strip()
        if spk:
            return self.tts.tts(text=text, speaker=spk, language=self.language)


        return self.tts.tts(text=text, language=self.language)

    def _post_fx(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0: return y
        if _LIBROSA_OK:
            if self.pitch_semitones != 0:
                try: y = librosa.effects.pitch_shift(y, sr=self.sample_rate_model, n_steps=self.pitch_semitones)
                except Exception: pass
            if abs(self.speed - 1.0) > 1e-3:
                try: y = librosa.effects.time_stretch(y, rate=self.speed)
                except Exception: pass
        if self.normalize:
            peak = float(np.max(np.abs(y)))
            if peak > 0: y = y / peak * 0.97
        return y

    def synth_to_wav_bytes(self, *, text: str, speaker_wav: Optional[str] = None, speaker: Optional[str] = None, seed: Optional[int] = None) -> bytes:
        with self._synth_lock:
            arr = self._synthesize_array(text=text, speaker=speaker, speaker_wav=speaker_wav, seed=seed)
        y = self._post_fx(arr)
        buf = io.BytesIO()
        sf.write(buf, y, int(self.sample_rate_model), subtype="PCM_16", format="WAV")
        return buf.getvalue()
