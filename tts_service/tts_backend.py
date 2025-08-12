
import json, os, threading, tempfile
from typing import Optional
import soundfile as sf
from TTS.api import TTS

class XTTSService:
    def __init__(self, settings_path: str = "settings.json"):
        with open(settings_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.model_name = cfg.get("model_name", "tts_models/multilingual/multi-dataset/xtts_v2")
        self.use_gpu = bool(cfg.get("use_gpu", True))
        self.language = cfg.get("language", "en")
        self.sample_rate = int(cfg.get("sample_rate", 24000))
        self.ref_voice = cfg.get("reference_voice_wav", "")
        self.default_speaker = cfg.get("default_speaker", "female-en-5")

        # Load once, keep warm
        self.tts = TTS(self.model_name, gpu=self.use_gpu)

        # simple playback lock so overlapping calls don't fight the sound device
        self._play_lock = threading.Lock()

    def synth_to_wavfile(self, text: str, speaker_wav: Optional[str] = None, speaker: Optional[str] = None) -> str:
        if not text or not text.strip():
            raise ValueError("Text is empty.")

        # pick voice strategy
        wav_arr = None
        if speaker_wav and os.path.exists(speaker_wav):
            wav_arr = self.tts.tts(text=text, speaker_wav=speaker_wav, language=self.language)
        elif self.ref_voice and os.path.exists(self.ref_voice):
            wav_arr = self.tts.tts(text=text, speaker_wav=self.ref_voice, language=self.language)
        else:
            spk = speaker or self.default_speaker
            wav_arr = self.tts.tts(text=text, speaker=spk, language=self.language)

        # write to temp WAV
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(out.name, wav_arr, self.tts.synthesizer.output_sample_rate)
        return out.name

    def play_wavfile(self, wav_path: str):
        import sounddevice as sd, soundfile as sf
        with self._play_lock:
            audio, sr = sf.read(wav_path, dtype='float32')
            sd.play(audio, sr)
            sd.wait()
