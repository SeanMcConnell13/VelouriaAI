# -*- coding: utf-8 -*-
import json, os, threading, io
from typing import Optional, Union, List
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


def _to_list(val: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalize a string-or-list value into a list of strings (paths)."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    s = str(val).strip()
    return [s] if s else []


def _existing_paths(paths: List[str]) -> List[str]:
    """Return only paths that exist on disk."""
    return [p for p in paths if os.path.exists(p)]


class XTTSService:
    def __init__(self, settings_path: str = "settings.json"):
        with open(settings_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.model_name = cfg.get("model_name", "tts_models/multilingual/multi-dataset/xtts_v2")
        self.language = cfg.get("language", "en")
        self.sample_rate_cfg = int(cfg.get("sample_rate", 24000))
        self.default_speaker = (cfg.get("speaker") or "").strip()

        # Accept string OR array in settings.json for reference voice
        # e.g. "reference_voice_wav": "C:/Velouria/voices/ref1.wav"
        # or   "reference_voice_wav": ["C:/Velouria/voices/ref1.wav","C:/Velouria/voices/ref2.wav"]
        self.reference_voice_list: List[str] = _to_list(cfg.get("reference_voice_wav"))

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
            try:
                self.tts.eval()
            except Exception:
                pass

        sr = None
        try:
            sr = int(getattr(self.tts, "synthesizer").output_sample_rate)
        except Exception:
            pass
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
            try:
                self.tts.to("cpu")
            except Exception:
                pass

    def _tts_call(self, **kwargs):
        """
        Call self.tts.tts safely:
        - Prefer split_sentences=False to keep prosody intact.
        - Fall back if this TTS build doesn't accept that kwarg.
        """
        try:
            return self.tts.tts(**kwargs, split_sentences=False)
        except TypeError:
            return self.tts.tts(**kwargs)

    def _resolve_refs(
        self,
        request_refs: Optional[Union[str, List[str]]]
    ) -> List[str]:
        """
        Decide which reference clips to use (request wins, else settings.json),
        normalize to list, then filter for existing files.
        """
        # 1) From request (highest priority)
        req_list = _existing_paths(_to_list(request_refs))
        if req_list:
            return req_list

        # 2) From settings.json (fallback)
        cfg_list = _existing_paths(self.reference_voice_list)
        return cfg_list

    def _synthesize_array(
        self,
        *,
        text: str,
        speaker: Optional[str],
        speaker_wav: Optional[Union[str, List[str]]],
        seed: Optional[int]
    ) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text is empty.")

        # Try reference clips (supports single string OR list)
        refs = self._resolve_refs(speaker_wav)
        if refs:
            return self._tts_call(text=text, speaker_wav=refs, language=self.language)

        # Else fallback to named speaker if provided
        spk = (speaker or self.default_speaker or "").strip()
        if spk:
            return self._tts_call(text=text, speaker=spk, language=self.language)

        # Else pure TTS with no conditioning
        return self._tts_call(text=text, language=self.language)

    @staticmethod
    def _soft_limiter(y: np.ndarray, ceiling: float = 0.96, drive: float = 2.2) -> np.ndarray:
        """
        Simple soft-clip limiter:
        1) apply gentle pre-gain (drive)
        2) tanh saturator
        3) scale to ceiling
        Keeps transients from spiking while sounding natural.
        """
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return y
        z = np.tanh(y * drive)
        z /= np.tanh(drive) + 1e-9
        return (z * ceiling).astype(np.float32)

    def _post_fx(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return y

        # Optional pitch/speed (keep modest for clarity)
        if _LIBROSA_OK:
            if self.pitch_semitones != 0:
                try:
                    y = librosa.effects.pitch_shift(y, sr=self.sample_rate_model, n_steps=self.pitch_semitones)
                except Exception:
                    pass
            if abs(self.speed - 1.0) > 1e-3:
                try:
                    y = librosa.effects.time_stretch(y, rate=self.speed)
                except Exception:
                    pass

        # Gentle RMS leveling BEFORE peak limiting (approx. -18 dBFS target)
        rms = float(np.sqrt(np.mean(y*y) + 1e-12))
        if rms > 1e-6:
            target_rms = 0.125  # ~-18 dBFS
            gain = min(2.5, max(0.4, target_rms / rms))
            y = y * gain

        # Soft limiter for peaks and harsh transients
        y = self._soft_limiter(y, ceiling=0.96, drive=2.2)

        # Final small safety
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0.98:
            y = (y / peak) * 0.98

        return y

    def synth_to_wav_bytes(
        self,
        *,
        text: str,
        speaker_wav: Optional[Union[str, List[str]]] = None,
        speaker: Optional[str] = None,
        seed: Optional[int] = None
    ) -> bytes:
        with self._synth_lock:
            arr = self._synthesize_array(
                text=text,
                speaker=speaker,
                speaker_wav=speaker_wav,
                seed=seed
            )
        y = self._post_fx(arr)
        buf = io.BytesIO()
        sf.write(buf, y, int(self.sample_rate_model), subtype="PCM_16", format="WAV")
        return buf.getvalue()
