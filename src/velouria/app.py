# -*- coding: utf-8 -*-
"""
Velouria app:
Push-to-talk ASR → LLM → TTS with paced playback and audio-driven motion
injection to VTube Studio via WebSocket.

Flow
----
1) Hold mouse side button to record mic
2) Transcribe with Whisper (faster-whisper preferred)
3) Stream prompt to Ollama; chunk on punctuation/pauses
4) Synthesize each chunk to WAV (XTTS), decode to fixed-size frames
5) Play frames at CHUNK_MS cadence and tap them to the motion loop
6) Motion loop computes envelope → mouth + head/body pose and injects
   custom parameters into VTS
"""

import os, io, re, json, time, asyncio, unicodedata, threading, collections, random
from typing import Optional, Dict, List
import numpy as np
import sounddevice as sd
from pynput import mouse
import httpx
from pydub import AudioSegment
import csv
import websockets  # WebSocket client for VTS

# --- Motion shaping knobs (mouth) -------------------------------------------

MOUTH_OUTPUT_GAIN   = 0.80   # final scale before sending to VTS (post-mapping)
MOUTH_OUTPUT_OFFSET = 0.00   # bias after scaling
MOUTH_SOFT_MAX      = 0.60   # software clamp (keeps VTS input from saturating)

MOUTH_CHEW_ENABLE   = True   # optional low-rate modulation to mimic syllables
MOUTH_CHEW_RATE_HZ  = 4.4
MOUTH_CHEW_DEPTH    = 0.28   # 0..0.4

# Plosive/sibilant dip to avoid "hissing mouth-open" moments
CONSONANT_DUCK_ENABLE  = True
CONSONANT_RATIO_THRESH = 0.65
CONSONANT_DUCK_AMOUNT  = 0.55  # 0..0.6

# Envelope and mapping (audio energy → mouth open)
ENV_ATTACK_MS   = 10.0
ENV_RELEASE_MS  = 140.0
MOUTH_KNEE      = 0.38
MOUTH_CEIL      = 0.95

# Normalization reference (lower → more open on the same audio)
ENV_NORM_REF    = 0.015

# Global pose scale multipliers
HEAD_SCALE_MULT = 1.00
BODY_SCALE_MULT = 1.00

# Queues, pacing, logging
FRAME_QUEUE_MAX = 3           # small on purpose; latest-wins
VTS_SEND_TIMEOUT_MS = 25
LOG_EVERY_SEC = 1.0

# LLM early-flush thresholds (feel snappier)
EARLY_FLUSH_MIN_CHARS = 14
EARLY_FLUSH_MAX_STALL_MS = 600

# Extra mouth gain on top of VTS_MOUTH_GAIN from settings
MOUTH_EXTRA_GAIN = 1.35

# --- Settings ---------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")
with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    settings = json.load(f)

PERSONA = settings.get("persona", {})
AUDIO = settings.get("audio", {})
FEWSHOT = settings.get("fewshot", {"path": "data/velouria_qna.csv", "k": 3})

MIC_DEVICE_INDEX = AUDIO.get("mic_device_index", None)
SAMPLE_RATE = int(AUDIO.get("sample_rate", 16000))
CHUNK_MS = int(AUDIO.get("chunk_ms", 20))
PTT_BUTTON = (AUDIO.get("ptt_button", "x1") or "x1").lower()
MAX_HOLD = int(AUDIO.get("max_hold_seconds", 30))
BEEPS = bool(AUDIO.get("beeps", True))

# ASR preferences (faster-whisper if available)
ASR_CFG = settings.get("asr", {})
ASR_ENGINE = ASR_CFG.get("engine", "faster-whisper")
ASR_MODEL = ASR_CFG.get("model", "small.en")
ASR_COMPUTE = ASR_CFG.get("compute_type", "float16")
ASR_VAD = bool(ASR_CFG.get("vad", True))
ASR_BEAM_SIZE = int(ASR_CFG.get("beam_size", 1))

USE_TTS_STREAM = False  # currently using request/response WAV

# LLM (Ollama)
LLM = settings.get("llm", {})
OLLAMA_ENDPOINT = (LLM.get("endpoint", "http://127.0.0.1:11434") or "").rstrip("/")
OLLAMA_MODEL = LLM.get("model", "velouria")
OLLAMA_KEEP_ALIVE = LLM.get("keep_alive", "10m")
OLLAMA_NUM_CTX = int(LLM.get("num_ctx", 2048))
OLLAMA_NUM_PREDICT = int(LLM.get("num_predict", 256))
OLLAMA_TEMP = float(LLM.get("temperature", 0.8))
OLLAMA_TOP_P = float(LLM.get("top_p", 0.9))

# Sentence splitting / pacing
SENT_CFG = LLM.get("sentence_split", {"min_chars": 20, "punctuation": ".?!\n"})
SENT_MIN_CHARS = int(SENT_CFG.get("min_chars", 20))

# TTS (XTTS endpoint)
TTS = settings.get("tts", {})
XTTS = TTS.get("xtts", {})
TTS_URL = (XTTS.get("endpoint", "http://127.0.0.1:8123/synthesize") or "").rstrip("/")
TTS_SPEAKER = (XTTS.get("speaker") or "").strip()
TTS_SPEAKER_WAV = (XTTS.get("speaker_wav") or "").strip()

# HTTP timeouts and logging
TO = settings.get("timeouts", {})
HTTP_CONNECT_S = float(TO.get("http_connect_s", 10))
HTTP_READ_S = float(TO.get("http_read_s", 60))
LATENCY_REPORT = bool(settings.get("logging", {}).get("latency_report", True))

# Short rolling history for persona memory
MEMORY_PATH = os.path.join(BASE_DIR, "memory.json")
MAX_TURNS = int(settings.get("max_history_turns", 6))

# VTube Studio
VTS_CFG = settings.get("vts", {})
VTS_ENABLED = bool(VTS_CFG.get("enabled", True))
VTS_WS_URL = VTS_CFG.get("ws_url", "ws://127.0.0.1:8001")
VTS_FRAME_MS = int(VTS_CFG.get("frame_ms", 20))
VTS_MOUTH_GAIN = float(VTS_CFG.get("mouth_gain", 1.2))

mp = PERSONA.get("motion_profile", {})
VTS_HEAD_SCALE = float(mp.get("head_range_deg", VTS_CFG.get("head_scale_deg", 30.0))) * HEAD_SCALE_MULT
VTS_BODY_SCALE = float(mp.get("body_range_deg", VTS_CFG.get("body_scale_deg", 30.0))) * BODY_SCALE_MULT
VTS_PLUGIN_NAME = VTS_CFG.get("plugin_name", "Velouria Motion Driver")
VTS_PLUGIN_DEV = VTS_CFG.get("plugin_developer", "Sean + GPT-5")
VTS_TOKEN_PATH = os.path.join(BASE_DIR, VTS_CFG.get("token_file", ".vts_token.json"))

# Playback tails (keeps lips moving briefly between chunks)
POSTROLL_MS = int(settings.get("audio", {}).get("postroll_ms", 120))
INTER_SENTENCE_GAP_MS = int(settings.get("audio", {}).get("intersentence_gap_ms", 60))

# Persistent audio/motion singletons
_MOTION = {"queue": None, "task": None, "audio": None, "sr": 24000}

# --- Audio ring buffer + callback output ------------------------------------

class AudioRing:
    """Thread-safe FIFO of float32 audio frames used by the callback."""
    def __init__(self, frame_samples: int):
        self.frame_samples = frame_samples
        self.q = collections.deque()
        self.lock = threading.Lock()
    def push(self, frame_f32: np.ndarray):
        if frame_f32.ndim != 1:
            frame_f32 = frame_f32.reshape(-1)
        with self.lock:
            self.q.append(frame_f32.astype(np.float32, copy=False))
    def pop(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.q:
                return self.q.popleft()
            return None
    def depth(self) -> int:
        with self.lock:
            return len(self.q)

class CallbackOutput:
    """
    Single-channel OutputStream that reads from AudioRing.
    Optionally taps frames into an asyncio queue for the motion loop.
    """
    def __init__(self, samplerate: int, frame_ms: int):
        self.sr = int(samplerate)
        self.frame_samples = max(1, int(self.sr * (frame_ms / 1000.0)))
        self.ring = AudioRing(self.frame_samples)
        self._started = False
        self._tap_queue: Optional[asyncio.Queue] = None
        self._tap_loop: Optional[asyncio.AbstractEventLoop] = None
        self._tap_frames = 0
        self._tap_dropped = 0
        self._tap_last_log = time.time()
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            latency='low',
            callback=self._cb
        )
    def set_tap(self, loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[np.ndarray]"):
        """Enable motion tapping by wiring an asyncio queue + loop."""
        self._tap_loop = loop
        self._tap_queue = queue
    def _offer_to_tap(self, frame: np.ndarray):
        """Non-blocking offer to the tap queue (drop oldest on overflow)."""
        q = self._tap_queue
        if q is None:
            return
        while True:
            try:
                q.put_nowait(frame)
                self._tap_frames += 1
                break
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    self._tap_dropped += 1
                except Exception:
                    break
        now = time.time()
        if (now - self._tap_last_log) >= LOG_EVERY_SEC:
            try:
                depth = q.qsize()
            except Exception:
                depth = -1
            fps = self._tap_frames / max(1e-6, (now - self._tap_last_log))
            print(f"[TAP] fps~={fps:.1f} depth={depth} dropped={self._tap_dropped}")
            self._tap_frames = 0
            self._tap_dropped = 0
            self._tap_last_log = now
    def _cb(self, outdata, frames, time_info, status):
        """PortAudio callback. Fills with next frame or silence."""
        frame = self.ring.pop()
        if frame is None:
            out = np.zeros((frames,), dtype=np.float32)
        else:
            if frame.size < frames:
                out = np.zeros((frames,), dtype=np.float32)
                out[:frame.size] = frame
            else:
                out = frame[:frames]
        outdata[:, 0] = out
        if self._tap_queue is not None and self._tap_loop is not None:
            self._tap_loop.call_soon_threadsafe(self._offer_to_tap, out.copy())
    def drain(self, max_wait_s: Optional[float] = None, frame_ms: int = 20):
        """Let playback finish gracefully (optionally bounded)."""
        start = time.time()
        while self.ring.depth() > 0:
            time.sleep(max(0.001, frame_ms / 1000.0))
            if max_wait_s is not None and (time.time() - start) > max_wait_s:
                break
        time.sleep(3 * (frame_ms / 1000.0))
    def start(self):
        if self._started: return
        self.stream.start(); self._started = True
    def stop(self):
        if not self._started: return
        try: self.stream.stop()
        finally:
            self.stream.close(); self._started = False

# --- Push-to-talk (mouse side button) ---------------------------------------

hold_lock = threading.Lock()
is_holding = False

def beep(freq=880, ms=90):
    """Optional UI beep for state changes."""
    if not BEEPS: return
    t = np.linspace(0, ms/1000, int(SAMPLE_RATE*ms/1000), False)
    tone = 0.12*np.sin(2*np.pi*freq*t).astype(np.float32)
    sd.play(tone, SAMPLE_RATE); sd.wait()

def ensure_input_device(device_index):
    """Validate input device index early; fail loud if misconfigured."""
    try:
        info = sd.query_devices(device_index)
        if info["max_input_channels"] < 1:
            raise ValueError("No input channels.")
        return device_index
    except Exception:
        print(sd.query_devices()); raise SystemExit("Fix settings.json audio.mic_device_index")

def _pynput_button_from_str(s: str):
    """Map settings string to pynput mouse button."""
    from pynput import mouse as _pm
    return _pm.Button.x2 if s == "x2" else _pm.Button.x1

def start_ptt_listener():
    """Background listener that tracks the side-button hold state."""
    target = _pynput_button_from_str(PTT_BUTTON)
    def on_click(x, y, button, pressed):
        global is_holding
        if button == target:
            with hold_lock:
                is_holding = pressed
    listener = mouse.Listener(on_click=on_click); listener.daemon = True; listener.start()

# --- Memory & prompting ------------------------------------------------------

def load_memory():
    if not os.path.exists(MEMORY_PATH): return []
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return []

def save_memory(history):
    """Persist a small rolling conversation window."""
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history[-MAX_TURNS:], f, ensure_ascii=False, indent=2)
    except Exception: pass

def _load_style_pack_csv(path: str) -> List[Dict[str, str]]:
    """Load persona few-shot examples from CSV with 'instruction','output'."""
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                ins = (row.get("instruction") or "").strip()
                out = (row.get("output") or "").strip()
                if ins and out:
                    rows.append({"instruction": ins, "output": out})
    except FileNotFoundError:
        pass
    return rows

_STYLE_PACK_CACHE = None
def _get_style_pack_rows() -> List[Dict[str, str]]:
    """Cache the CSV because we sample randomly per prompt."""
    global _STYLE_PACK_CACHE
    if _STYLE_PACK_CACHE is None:
        _STYLE_PACK_CACHE = _load_style_pack_csv(settings.get("persona", {}).get("fewshot", {}).get("path", "data/velouria_qna.csv"))
    return _STYLE_PACK_CACHE

def build_fewshot_block(k: int = None) -> str:
    """Pick k random persona exemplars as few-shot context."""
    rows = _get_style_pack_rows()
    if not rows: return ""
    kk = int(settings.get("persona", {}).get("fewshot", {}).get("k", 3) if k is None else k)
    kk = max(0, min(kk, len(rows)))
    if kk == 0: return ""
    shuffled = rows[:]; random.shuffle(shuffled); chosen = shuffled[:kk]
    lines = []
    for r in chosen:
        lines.append(f"User: {r['instruction']}")
        lines.append(f"Assistant: {r['output']}")
    return "\n".join(lines) + "\n\n"

def build_system_prompt(persona):
    """Compose persona system content with style/taboos/refusal notes."""
    rules = "\n".join(f"- {r}" for r in persona.get("style_rules", []))
    taboos = "\n".join(f"- {t}" for t in persona.get("taboos", []))
    return (
        f"{persona.get('system_prompt', '')}\n\n"
        f"STYLE RULES:\n{rules}\n\n"
        f"TABOOS:\n{taboos}\n\n"
        f"REFUSAL STYLE:\n{persona.get('refusal_style', '')}\n"
        f"Stay in character as {persona.get('name', 'Velouria')}."
    )

def build_prompt_with_history(user_text, history):
    """Interleave recent turns and the new user line."""
    lines = []
    for turn in history[-MAX_TURNS:]:
        u = (turn.get("user") or "").strip()
        a = (turn.get("assistant") or "").strip()
        if u: lines.append(f"User: {u}")
        if a: lines.append(f"Assistant: {a}")
    lines.append(f"User: {user_text.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)

# --- ASR ---------------------------------------------------------------------

_FASTER_OK = False
try:
    from faster_whisper import WhisperModel as FWModel
    _FASTER_OK = True
except Exception:
    _FASTER_OK = False
    import whisper as openai_whisper

class ASREngine:
    """Select faster-whisper if possible; otherwise fall back to openai/whisper."""
    def __init__(self):
        self.engine = ASR_ENGINE
        if _FASTER_OK and self.engine == "faster-whisper":
            try:
                self.model = FWModel(ASR_MODEL, device="cuda", compute_type=ASR_COMPUTE)
                print(f"🟩 faster-whisper ({ASR_MODEL},{ASR_COMPUTE})")
            except Exception:
                self._fallback_openai()
        else:
            self._fallback_openai()
    def _fallback_openai(self):
        try:
            self.model = openai_whisper.load_model(ASR_MODEL, device="cuda")
        except Exception:
            self.model = openai_whisper.load_model(ASR_MODEL, device="cpu")
    def transcribe_np16(self, audio_i16: np.ndarray, sr: int) -> str:
        """Transcribe int16 mono buffer to text."""
        if audio_i16.ndim > 1: audio_i16 = audio_i16.reshape(-1)
        audio_f32 = (audio_i16.astype(np.float32) / 32768.0)
        if _FASTER_OK and self.engine == "faster-whisper":
            segments, _ = self.model.transcribe(audio_f32, language="en", vad_filter=ASR_VAD, beam_size=ASR_BEAM_SIZE)
            return (" ".join(seg.text for seg in segments)).strip()
        else:
            result = self.model.transcribe(audio_f32, language="en")
            return (result.get("text") or "").strip()

# --- Text sanitization -------------------------------------------------------

_EMOJI_RE = re.compile("[" "\U0001F300-\U0001FAFF" "\U00002700-\U000027BF" "\U00002600-\U000026FF" "]+", re.UNICODE)
_ALLOWED_CHARS_RE = re.compile(r"[^a-zA-Z0-9\s\.\,\!\?\;\:\-\'\"/()]+", re.UNICODE)

def sanitize_text(s: str) -> str:
    """Normalize punctuation, strip emojis/control chars, shrink whitespace."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = s.replace("—", "-").replace("–", "-")
    s = _EMOJI_RE.sub("", s)
    s = "".join(ch for ch in s if ch == "\n" or (unicodedata.category(ch)[0] != "C"))
    s = _ALLOWED_CHARS_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

# --- Recording ---------------------------------------------------------------

def record_audio_while_held_np16() -> Optional[np.ndarray]:
    """
    Record mic as long as the configured side button is held.
    Returns mono int16 samples or None if nothing captured.
    """
    dev = ensure_input_device(MIC_DEVICE_INDEX)
    print("🎙️ Hold side button…"); frames = []; start_ts = None
    def callback(indata, frames_count, time_info, status):
        with hold_lock:
            if is_holding: frames.append(indata.copy())
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=1024, device=dev, callback=callback):
        while True:
            with hold_lock:
                if is_holding: start_ts = time.time(); break
            time.sleep(0.01)
        print("🎤 Recording…")
        while True:
            with hold_lock: holding = is_holding
            if not holding or (time.time()-start_ts)>MAX_HOLD: break
            time.sleep(0.01)
    if not frames: print("…no audio."); return None
    return np.concatenate(frames, axis=0).reshape(-1)

# --- Warmup ------------------------------------------------------------------

async def warmup(client_llm: httpx.AsyncClient, client_tts: httpx.AsyncClient, asr: ASREngine):
    """Touch ASR/TTS/LLM once so the first user turn isn't cold."""
    try:
        zeros = np.zeros((SAMPLE_RATE//2,), dtype=np.int16); _ = asr.transcribe_np16(zeros, SAMPLE_RATE)
    except Exception: pass
    try:
        await client_tts.post(TTS_URL, json={"text":"ready"}, timeout=httpx.Timeout(HTTP_READ_S, connect=HTTP_CONNECT_S))
    except Exception: pass
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": "User: say 'ready'\nAssistant:", "stream": False,
                   "keep_alive": OLLAMA_KEEP_ALIVE, "options": {"temperature":0.0, "num_predict":1}}
        await client_llm.post(f"{OLLAMA_ENDPOINT}/api/generate", json=payload, timeout=httpx.Timeout(HTTP_READ_S, connect=HTTP_CONNECT_S))
    except Exception: pass

# --- WAV → frames ------------------------------------------------------------

def decode_wav_to_frames(body: bytes, out_sr: int, frame_samples: int) -> List[np.ndarray]:
    """
    Decode WAV bytes to mono float32 frames at out_sr.
    Pads last frame so all frames are exactly frame_samples long.
    """
    seg = AudioSegment.from_file(io.BytesIO(body), format="wav")
    src_sr = int(seg.frame_rate)
    ch = int(seg.channels)
    sw = int(seg.sample_width)

    a = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if ch > 1:
        a = a.reshape((-1, ch)).mean(axis=1)

    # normalize to [-1, 1]
    if sw == 2:
        a = a / 32768.0
    elif sw == 1:
        a = (a - 128.0) / 128.0
    else:
        peak = float(np.max(np.abs(a))) if a.size else 1.0
        a = a / (peak + 1e-9)

    # resample (linear) if needed
    if src_sr != out_sr and a.size:
        n_src = a.size
        n_dst = max(1, int(round(n_src * (out_sr / float(src_sr)))))
        x_src = np.linspace(0.0, 1.0, num=n_src, endpoint=False, dtype=np.float32)
        x_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False, dtype=np.float32)
        a = np.interp(x_dst, x_src, a).astype(np.float32, copy=False)

    frames: List[np.ndarray] = []
    i = 0
    while i < a.size:
        fr = a[i:i+frame_samples]
        if fr.size < frame_samples:
            fr = np.pad(fr, (0, frame_samples - fr.size))
        frames.append(fr)
        i += frame_samples
    return frames

# --- VTS token helpers -------------------------------------------------------

def _vts_load_token(path: str) -> Optional[str]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return (json.load(f) or {}).get("token")
    except Exception: pass
    return None

def _vts_save_token(path: str, token: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token": token}, f)
    except Exception: pass

# --- VTS client --------------------------------------------------------------

class VTSClient:
    """
    Minimal VTube Studio RPC client with auto-auth and soft reconnect.
    Provides an inject_inline() helper for per-frame parameter writes.
    """
    def __init__(self, url: str, plugin_name: str, plugin_dev: str, token_path: str):
        self.url = url; self.plugin_name = plugin_name; self.plugin_dev = plugin_dev
        self.token_path = token_path
        self.ws: websockets.WebSocketClientProtocol | None = None
        self._req_id = 0
        self._rpc_lock = asyncio.Lock()
        self._inject_lock = asyncio.Lock()
        self._reader_task: asyncio.Task | None = None
        self._running = False
        self._pending: Dict[str, asyncio.Future] = {}
        self.authed = False
        self._last_token: Optional[str] = _vts_load_token(self.token_path)
        self._next_reconnect_ts = 0.0

    async def connect(self):
        if self.ws is not None: return
        if not self._running: self._running = True
        self.ws = await websockets.connect(
            self.url, max_size=4_000_000, max_queue=None,
            ping_interval=30, ping_timeout=20, close_timeout=1,
        )
        if self._reader_task is None or self._reader_task.done():
            self._reader_task = asyncio.create_task(self._reader_loop())

    async def _reconnect_soft(self):
        """Close/reopen the socket and redo auth, with a small backoff."""
        now = time.time()
        if now < self._next_reconnect_ts: return
        self._next_reconnect_ts = now + 0.5
        old_ws = self.ws; self.ws = None
        if old_ws is not None:
            try: await old_ws.close()
            except Exception: pass
        try:
            await self.connect()
            await self._handshake_and_auth()
        except Exception as e:
            print(f"[VTS] soft reconnect failed: {type(e).__name__}: {e!r}")

    async def close(self):
        """Tear everything down and fail any waiting RPCs."""
        self._running = False
        try:
            if self.ws: await self.ws.close()
        finally:
            self.ws = None
        if self._reader_task:
            try: await asyncio.wait_for(self._reader_task, timeout=0.5)
            except Exception: pass
            self._reader_task = None
        for fut in list(self._pending.values()):
            if not fut.done(): fut.set_exception(RuntimeError("VTS connection closed"))
        self._pending.clear()
        self.authed = False

    async def _reader_loop(self):
        """Read messages, resolve futures, track auth state, print errors."""
        try:
            while self._running and self.ws is not None:
                raw = await self.ws.recv()
                try: msg = json.loads(raw)
                except Exception: continue
                rid = str(msg.get("requestID", "") or "")
                if rid and rid in self._pending:
                    fut = self._pending.pop(rid)
                    if not fut.done(): fut.set_result(msg)
                if msg.get("messageType") == "AuthenticationResponse":
                    self.authed = bool(msg.get("data", {}).get("authenticated", False))
                if msg.get("messageType") == "AuthenticationTokenResponse":
                    tok = (msg.get("data") or {}).get("authenticationToken")
                    if tok:
                        self._last_token = tok; _vts_save_token(self.token_path, tok)
                if str(msg.get("messageType", "")).endswith("Error"):
                    print(f"[VTS] error reply: {msg}")
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            print(f"[VTS] reader loop error: {e}")
        finally:
            self.authed = False

    async def rpc(self, message_type: str, data: Dict, timeout: float = 3.0) -> Dict:
        """Send a typed RPC and await its response with a timeout."""
        async with self._rpc_lock:
            if self.ws is None: await self.connect()
            self._req_id += 1; rid = str(self._req_id)
            payload = {"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0",
                       "requestID":rid,"messageType":message_type,"data":data}
            fut = asyncio.get_running_loop().create_future()
            self._pending[rid] = fut
            try:
                await self.ws.send(json.dumps(payload))
            except (websockets.ConnectionClosedError, websockets.ProtocolError):
                await self._reconnect_soft()
                if self.ws is None: raise
                await self.ws.send(json.dumps(payload))
        return await asyncio.wait_for(fut, timeout=timeout)

    async def _handshake_and_auth(self):
        """Obtain or reuse a plugin token and authenticate."""
        if not self._last_token:
            tok_resp = await self.rpc("AuthenticationTokenRequest", {
                "pluginName": self.plugin_name, "pluginDeveloper": self.plugin_dev
            }, timeout=5.0)
            tok = (tok_resp.get("data") or {}).get("authenticationToken")
            if not tok: raise RuntimeError("VTS did not return an authentication token")
            self._last_token = tok; _vts_save_token(self.token_path, tok)
        auth_resp = await self.rpc("AuthenticationRequest", {
            "pluginName": self.plugin_name, "pluginDeveloper": self.plugin_dev,
            "authenticationToken": self._last_token,
        }, timeout=5.0)
        ok = bool(auth_resp.get("data", {}).get("authenticated", False))
        if not ok:
            _vts_save_token(self.token_path, ""); self._last_token = None
            tok_resp = await self.rpc("AuthenticationTokenRequest", {
                "pluginName": self.plugin_name, "pluginDeveloper": self.plugin_dev
            }, timeout=5.0)
            tok = (tok_resp.get("data") or {}).get("authenticationToken")
            if not tok: raise RuntimeError("VTS did not return a new authentication token")
            self._last_token = tok; _vts_save_token(self.token_path, tok)
            auth_resp = await self.rpc("AuthenticationRequest", {
                "pluginName": self.plugin_name,"pluginDeveloper": self.plugin_dev,
                "authenticationToken": self._last_token,
            }, timeout=5.0)
            ok = bool(auth_resp.get("data", {}).get("authenticated", False))
            if not ok: raise RuntimeError("VTS authentication failed")
        self.authed = True

    async def ensure_ready(self):
        """Connect + authenticate if needed."""
        if self.ws is None: await self.connect()
        if not self.authed: await self._handshake_and_auth()

    async def inject_inline(self, param_values: Dict[str, float], weight: float = 1.0):
        """
        Fast path for per-frame parameter writes (single InjectParameterDataRequest).
        """
        pvals = [{"id": k, "value": float(v), "weight": float(weight)} for k, v in param_values.items()]
        payload = {"apiName":"VTubeStudioPublicAPI","apiVersion":"1.0",
                   "requestID": str(int(time.time()*1000)&0x7FFFFFFF),
                   "messageType":"InjectParameterDataRequest",
                   "data":{"parameterValues": pvals, "mode": "set"}}
        try:
            if (self.ws is None) or (not self.authed):
                await self._reconnect_soft()
            if self.ws is None or not self.authed:
                return
            async with self._inject_lock:
                await asyncio.wait_for(self.ws.send(json.dumps(payload)), timeout=VTS_SEND_TIMEOUT_MS/1000.0)
        except (asyncio.TimeoutError,
                websockets.ConnectionClosed,
                websockets.ConnectionClosedError,
                websockets.ProtocolError):
            await self._reconnect_soft()
        except Exception as e:
            print(f"[VTS] inline send unexpected: {type(e).__name__}: {e!r}")

async def vts_ensure_custom_params(vts: "VTSClient"):
    """Create the custom tracking params the rig expects (idempotent)."""
    required = [
        ("VelMouthOpen", 0.0, 1.0, 0.0, "Velouria mouth open (0..1)"),
        ("VelFaceAngleX", -30.0, 30.0, 0.0, "Velouria head angle X (deg)"),
        ("VelFaceAngleY", -30.0, 30.0, 0.0, "Velouria head angle Y (deg)"),
        ("VelFaceAngleZ", -30.0, 30.0, 0.0, "Velouria head angle Z (deg)"),
        ("VelBodyAngleX", -30.0, 30.0, 0.0, "Velouria body angle X (deg)"),
        ("VelBodyAngleY", -30.0, 30.0, 0.0, "Velouria body angle Y (deg)"),
        ("VelBodyAngleZ", -30.0, 30.0, 0.0, "Velouria body angle Z (deg)"),
    ]
    for name, mn, mx, dv, expl in required:
        try:
            await vts.rpc(
                "ParameterCreationRequest",
                {"parameterName": name, "explanation": expl, "min": mn, "max": mx, "defaultValue": dv},
                timeout=3.0,
            )
        except Exception:
            pass

# --- Motion helpers ----------------------------------------------------------

def _map_mouth(envelope: float, ceil: float = MOUTH_CEIL, knee: float = MOUTH_KNEE) -> float:
    """Nonlinear curve: gentle near zero, faster rise after the knee."""
    e = float(np.clip(envelope, 0.0, 1.0))
    if e <= 1e-6: return 0.0
    if e < knee:
        return (e / knee) ** 1.6 * (0.25 * ceil)
    x = (e - knee) / (1 - knee)
    x = x ** 0.70
    return x * ceil

class SmoothParam:
    """Critically/under-damped 2nd-order smoother (feels like spring-mass)."""
    def __init__(self, omega: float, zeta: float = 1.0, y0: float = 0.0):
        self.omega = float(omega); self.zeta = float(zeta); self.y = float(y0); self.v = 0.0
    def reset(self, y0: float = 0.0):
        self.y = float(y0); self.v = 0.0
    def update(self, dt: float, target: float) -> float:
        w = self.omega; z = self.zeta
        a = (w*w) * (target - self.y) - (2.0*z*w) * self.v
        self.v += a * dt; self.y += self.v * dt
        return self.y

class SpeechEnvelope:
    """
    RMS envelope with attack/release smoothing and DC removal.
    Produces a stable 0..~1-ish loudness measure from float32 audio.
    """
    def __init__(self, sr: int, atk_ms: float = ENV_ATTACK_MS, rel_ms: float = ENV_RELEASE_MS):
        self.sr = float(sr)
        self.alpha_up = np.exp(-1.0 / max(1.0, (atk_ms/1000.0) * self.sr))
        self.alpha_dn = np.exp(-1.0 / max(1.0, (rel_ms/1000.0) * self.sr))
        self.env = 0.0
    def process(self, frame: np.ndarray) -> float:
        if frame.size == 0:
            self.env *= self.alpha_dn
            return self.env
        x = frame.astype(np.float32, copy=False).reshape(-1)
        x = x - float(np.mean(x))  # DC removal is important when tapping DAC
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        a = self.alpha_up if rms > self.env else self.alpha_dn
        self.env = a * self.env + (1.0 - a) * rms
        return self.env
    def reset(self):
        self.env = 0.0

# --- Motion consumer (driven by tapped audio) --------------------------------

_VTS_SINGLETON = {"client": None, "ready": False}

async def get_vts():
    """Singleton VTS client with ready-checked auth + parameter creation."""
    if not VTS_ENABLED: return None
    if _VTS_SINGLETON["client"] is None:
        _VTS_SINGLETON["client"] = VTSClient(VTS_WS_URL, VTS_PLUGIN_NAME, VTS_PLUGIN_DEV, VTS_TOKEN_PATH)
        await _VTS_SINGLETON["client"].ensure_ready()
        await vts_ensure_custom_params(_VTS_SINGLETON["client"])
        _VTS_SINGLETON["ready"] = True
        print("[VTS] connected and authed. Custom params ensured.")
    else:
        if not _VTS_SINGLETON["client"].authed:
            await _VTS_SINGLETON["client"].ensure_ready()
    return _VTS_SINGLETON["client"]

async def motion_consumer(frame_queue: asyncio.Queue, samplerate: int):
    """
    Consume audio frames from the output tap and drive VTS parameters.
    Uses a talk/idle state machine, noise-floor tracking, consonant ducking,
    and a little "chew" modulation for life.
    """
    # Pull tunables from globals but tolerate missing ones (sane defaults).
    CHUNK_MS_L         = globals().get("CHUNK_MS", 20)
    ENV_NORM_REF_L     = float(globals().get("ENV_NORM_REF", 0.04))
    AUTO_NORM_GAIN_L   = float(globals().get("AUTO_NORM_GAIN", 0.80))
    VTS_MOUTH_GAIN_L   = float(globals().get("VTS_MOUTH_GAIN", 1.2))
    MOUTH_EXTRA_GAIN_L = float(globals().get("MOUTH_EXTRA_GAIN", 1.0))
    MOUTH_CEIL_L       = float(globals().get("MOUTH_CEIL", 0.90))
    MOUTH_KNEE_L       = float(globals().get("MOUTH_KNEE", 0.38))
    MOUTH_OUTPUT_GAIN_L   = float(globals().get("MOUTH_OUTPUT_GAIN", 0.35))
    MOUTH_OUTPUT_OFFSET_L = float(globals().get("MOUTH_OUTPUT_OFFSET", 0.0))
    MOUTH_SOFT_MAX_L      = float(globals().get("MOUTH_SOFT_MAX", 0.85))
    LOG_EVERY_SEC_L    = float(globals().get("LOG_EVERY_SEC", 1.0))
    VTS_HEAD_SCALE_L   = float(globals().get("VTS_HEAD_SCALE", 30.0)) * float(globals().get("HEAD_SCALE_MULT", 1.0))
    VTS_BODY_SCALE_L   = float(globals().get("VTS_BODY_SCALE", 30.0)) * float(globals().get("BODY_SCALE_MULT", 1.0))

    CHEW_EN    = bool(globals().get("MOUTH_CHEW_ENABLE", True))
    CHEW_RATE  = float(globals().get("MOUTH_CHEW_RATE_HZ", 4.4))
    CHEW_DEPTH = float(globals().get("MOUTH_CHEW_DEPTH", 0.20))
    DUCK_EN    = bool(globals().get("CONSONANT_DUCK_ENABLE", True))
    DUCK_THR   = float(globals().get("CONSONANT_RATIO_THRESH", 0.95))
    DUCK_AMT   = float(globals().get("CONSONANT_DUCK_AMOUNT", 0.33))

    if not globals().get("VTS_ENABLED", True):
        # Drain and exit cleanly if motion is disabled
        while True:
            item = await frame_queue.get()
            if item is None:
                break
        return

    vts = await get_vts()
    print(f"[VTS] motion loop started (audio-driven, sr={samplerate}, frame≈{CHUNK_MS_L}ms).")

    # Smoothers (mouth a bit underdamped for snappier feel)
    mouth_s   = SmoothParam(omega=26.0, zeta=0.80, y0=0.0)
    head_x_s  = SmoothParam(omega=8.5,  zeta=1.00, y0=0.0)
    head_y_s  = SmoothParam(omega=8.5,  zeta=1.00, y0=0.0)
    head_z_s  = SmoothParam(omega=7.5,  zeta=1.00, y0=0.0)
    body_x_s  = SmoothParam(omega=4.2,  zeta=1.00, y0=0.0)
    body_y_s  = SmoothParam(omega=4.0,  zeta=1.00, y0=0.0)
    body_z_s  = SmoothParam(omega=3.8,  zeta=1.00, y0=0.0)

    # Envelope detectors (main + a slightly different one for consonants)
    try:
        envdet = SpeechEnvelope(sr=samplerate, atk_ms=ENV_ATTACK_MS, rel_ms=ENV_RELEASE_MS, hp_alpha=0.97)
    except (TypeError, NameError):
        try:
            envdet = SpeechEnvelope(sr=samplerate, atk_ms=6.0, rel_ms=120.0, hp_alpha=0.97)
        except TypeError:
            envdet = SpeechEnvelope(sr=samplerate, atk_ms=6.0, rel_ms=120.0)

    ANGLE_MIN, ANGLE_MAX = -30.0, 30.0
    MOUTH_MIN, MOUTH_MAX = 0.0, 1.0

    try:
        conson_env = SpeechEnvelope(sr=samplerate, atk_ms=6.0, rel_ms=80.0, hp_alpha=0.995)
    except (TypeError, NameError):
        try:
            conson_env = SpeechEnvelope(sr=samplerate, atk_ms=6.0, rel_ms=80.0)
        except TypeError:
            conson_env = envdet

    chew_phase = 0.0
    rng = np.random.default_rng()
    frame_len = max(1, int(round(samplerate * (CHUNK_MS_L / 1000.0))))
    silence = np.zeros((frame_len,), dtype=np.float32)

    # Centers recenter slowly when quiet
    t0 = time.monotonic()
    head_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    body_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    recenter_tc = 0.8  # seconds

    # Noise floor tracking and peak-based normalization
    noise_floor = 0.0
    NOISE_ALPHA = 0.004
    NOISE_MARGIN = 1.25
    NOISE_DECAY_WHEN_LOUD = 0.001
    SUBTRACT_FACTOR = 1.3

    peak_env = 1e-6
    peak_decay_tc = 0.75

    # Talk/idle with hysteresis + watchdogs for true silence
    ACTIVATE_THR = 0.10
    DEACTIVATE_THR = 0.04
    ACTIVATE_HANG_S = 0.06
    DEACTIVATE_HANG_S = 0.25
    talking = False
    above_since = None
    last_voice_ts = 0.0

    LOW_ENV = 0.006
    SILENCE_FORCE_IDLE_S = 0.35
    ZERO_PCM_FORCE_IDLE_S = 0.25
    low_env_streak = 0.0
    zero_pcm_streak = 0.0

    last_log = time.monotonic()
    ticks = 0
    send_p95_window = collections.deque(maxlen=64)
    timeouts = 0
    timeouts_in_a_row = 0

    try:
        while True:
            # Try to keep cadence; fill with silence if tap is starved
            try:
                frame = await asyncio.wait_for(frame_queue.get(), timeout=0.08)
                if frame is None:
                    break
                if frame.size != frame_len:
                    if frame.size < frame_len:
                        frame = np.pad(frame, (0, frame_len - frame.size))
                    else:
                        frame = frame[:frame_len]
                dt = frame.size / float(samplerate)
            except asyncio.TimeoutError:
                frame = silence
                dt = frame.size / float(samplerate)
                timeouts += 1
                timeouts_in_a_row += 1
                if timeouts_in_a_row == 5:
                    print("[MOTION] starving: no frames for ~", round(5 * 0.08, 2), "s")
            else:
                timeouts_in_a_row = 0

            raw_env = envdet.process(frame)
            now = time.monotonic()

            # Zero-PCM watchdog (helps when audio truly stops)
            if np.max(np.abs(frame)) <= 1e-9:
                zero_pcm_streak += dt
            else:
                zero_pcm_streak = 0.0

            # Track a moving noise floor and subtract it from envelope
            is_noise_like = raw_env <= (max(noise_floor, 1e-6) * NOISE_MARGIN)
            if is_noise_like:
                noise_floor = (1.0 - NOISE_ALPHA) * noise_floor + NOISE_ALPHA * raw_env
            else:
                noise_floor *= (1.0 - NOISE_DECAY_WHEN_LOUD)
            raw_env_adj = max(0.0, raw_env - SUBTRACT_FACTOR * noise_floor)

            # Normalize by a slow peak to keep dynamics stable
            peak_decay = np.exp(-dt / max(1e-3, peak_decay_tc))
            peak_env = max(raw_env_adj, peak_env * peak_decay)
            dyn_ref = max(ENV_NORM_REF_L, AUTO_NORM_GAIN_L * peak_env)
            norm_env = float(np.clip(raw_env_adj / max(dyn_ref, 1e-6), 0.0, 1.0))

            # Brief sustain across micro-gaps
            speech_active_until = getattr(motion_consumer, "_speech_active_until", 0.0)
            if norm_env > 0.015:
                speech_active_until = now + 0.09
            if now < speech_active_until:
                norm_env = max(norm_env, 0.010)
            if (now >= speech_active_until) and (norm_env < 0.01):
                norm_env = 0.0
            motion_consumer._speech_active_until = speech_active_until

            # Talk/idle with hysteresis
            if norm_env >= ACTIVATE_THR:
                if above_since is None:
                    above_since = now
                last_voice_ts = now
                if not talking and (now - above_since) >= ACTIVATE_HANG_S:
                    talking = True
            else:
                above_since = None
                if talking and (now - last_voice_ts) >= DEACTIVATE_HANG_S and norm_env <= DEACTIVATE_THR:
                    talking = False
                    envdet.reset(); mouth_s.reset(0.0)
                    peak_env = 1e-6
                    speech_active_until = now

            # Force idle if envelope stays tiny or PCM is zeros
            if raw_env_adj <= LOW_ENV:
                low_env_streak += dt
            else:
                low_env_streak = 0.0
            if (low_env_streak >= SILENCE_FORCE_IDLE_S) or (zero_pcm_streak >= ZERO_PCM_FORCE_IDLE_S):
                if talking:
                    talking = False
                    envdet.reset(); mouth_s.reset(0.0)
                    peak_env = 1e-6
                    speech_active_until = now
                low_env_streak = 0.0
                zero_pcm_streak = 0.0

            # Mouth target with consonant duck + chew modulation
            if talking:
                env_for_mouth = np.clip(norm_env, 0.0, 1.0) ** 0.80
                mouth_target = _map_mouth(env_for_mouth, ceil=MOUTH_CEIL_L, knee=MOUTH_KNEE_L)
                mouth_target *= (VTS_MOUTH_GAIN_L * MOUTH_EXTRA_GAIN_L)

                if DUCK_EN:
                    hf_raw = conson_env.process(frame)
                    ratio  = hf_raw / (raw_env + 1e-6)
                    spill  = np.clip((ratio - DUCK_THR) / max(1e-6, (1.5 - DUCK_THR)), 0.0, 1.0)
                    mouth_target *= (1.0 - DUCK_AMT * spill)

                if CHEW_EN:
                    speech_drive = np.sqrt(norm_env)
                    chew_phase += 2*np.pi * CHEW_RATE * dt * (0.6 + 0.4*speech_drive)
                    mod  = 0.5 * (1.0 + np.sin(chew_phase))   # 0..1
                    mouth_target *= (1.0 - (CHEW_DEPTH * (1.0 - mod)))
            else:
                mouth_target = 0.0

            # Final software-side shaping to match VTS input ranges
            mouth_target = (mouth_target * MOUTH_OUTPUT_GAIN_L) + MOUTH_OUTPUT_OFFSET_L
            mouth_target = float(np.clip(mouth_target, 0.0, MOUTH_SOFT_MAX_L))

            # Expressiveness: idle sway + jitter, scaled by speech drive
            speech_drive = np.sqrt(norm_env) if talking else 0.0
            talk_boost   = (1.0 + 1.2 * speech_drive) if talking else 1.0
            idle_mix     = (0.6 * (1.0 - speech_drive)) if talking else 1.0

            t = now - t0
            sway_hx = idle_mix * np.sin(2*np.pi*0.20 * t)
            sway_hy = idle_mix * np.sin(2*np.pi*0.17 * t + 1.1)
            sway_hz = idle_mix * np.sin(2*np.pi*0.23 * t + 0.4)
            sway_bx = idle_mix * np.sin(2*np.pi*0.15 * t + 0.8)
            sway_by = idle_mix * np.sin(2*np.pi*0.13 * t + 1.9)
            sway_bz = idle_mix * np.sin(2*np.pi*0.19 * t + 0.2)

            jitter = min(1.0, 0.30 + 0.70 * speech_drive)
            hx_n = rng.normal(0.0, 1.0) * 0.30 * jitter
            hy_n = rng.normal(0.0, 1.0) * 0.30 * jitter
            hz_n = rng.normal(0.0, 1.0) * 0.28 * jitter
            bx_n = rng.normal(0.0, 1.0) * 0.26 * jitter
            by_n = rng.normal(0.0, 1.0) * 0.26 * jitter
            bz_n = rng.normal(0.0, 1.0) * 0.24 * jitter

            # Recenter slowly when quiet
            rec_k = 1.0 - np.exp(-dt / recenter_tc)
            if not talking or norm_env < 0.05:
                head_center *= (1.0 - rec_k)
                body_center *= (1.0 - rec_k)

            # Pose amplitudes
            base_h = 0.40; base_b = 0.45
            amp_h  = (base_h + 0.60 * speech_drive) * talk_boost
            amp_b  = (base_b + 0.60 * speech_drive) * talk_boost

            # Speech-synced nod/bob
            nod = (8.0 * speech_drive * np.sin(2*np.pi*2.2 * t)) if talking else 0.0
            bob = (5.0 * speech_drive * np.sin(2*np.pi*1.8 * t + 0.3)) if talking else 0.0

            hx_t = head_center[0] + VTS_HEAD_SCALE_L * (amp_h * (sway_hx*0.5 + hx_n*0.5))
            hy_t = head_center[1] + VTS_HEAD_SCALE_L * (amp_h * (sway_hy*0.5 + hy_n*0.5) + nod * 0.25)
            hz_t = head_center[2] + VTS_HEAD_SCALE_L * (amp_h * (sway_hz*0.5 + hz_n*0.5))
            bx_t = body_center[0] + VTS_BODY_SCALE_L * (amp_b * (sway_bx*0.5 + bx_n*0.5))
            by_t = body_center[1] + VTS_BODY_SCALE_L * (amp_b * (sway_by*0.5 + by_n*0.5) + bob * 0.35)
            bz_t = body_center[2] + VTS_BODY_SCALE_L * (amp_b * (sway_bz*0.5 + bz_n*0.5))

            # Smooth and clamp to rig limits
            m  = float(np.clip(mouth_s.update(dt, mouth_target), MOUTH_MIN, MOUTH_MAX))
            hx = float(np.clip(head_x_s.update(dt, hx_t), -30.0, 30.0))
            hy = float(np.clip(head_y_s.update(dt, hy_t), -30.0, 30.0))
            hz = float(np.clip(head_z_s.update(dt, hz_t), -30.0, 30.0))
            bx = float(np.clip(body_x_s.update(dt, bx_t), -30.0, 30.0))
            by = float(np.clip(body_y_s.update(dt, by_t), -30.0, 30.0))
            bz = float(np.clip(body_z_s.update(dt, bz_t), -30.0, 30.0))

            pose = {
                "VelMouthOpen": m,
                "VelFaceAngleX": hx, "VelFaceAngleY": hy, "VelFaceAngleZ": hz,
                "VelBodyAngleX": bx, "VelBodyAngleY": by, "VelBodyAngleZ": bz,
            }

            t_send0 = time.perf_counter()
            await vts.inject_inline(pose, weight=1.0)
            send_ms = (time.perf_counter() - t_send0) * 1000.0
            send_p95_window.append(send_ms)

            ticks += 1
            now_end = time.monotonic()
            if (now_end - last_log) >= LOG_EVERY_SEC_L:
                window = list(send_p95_window)
                p95 = 0.0 if not window else sorted(window)[int(max(0, 0.95*len(window)-1))]
                try:
                    qdepth = frame_queue.qsize()
                except Exception:
                    qdepth = -1
                fps = ticks / (now_end - last_log + 1e-9)
                print(
                    f"[MOTION] fps={fps:.1f} q={qdepth} timeouts={timeouts} vts_p95={p95:.1f}ms "
                    f"state={'TALK' if talking else 'IDLE'} "
                    f"env(raw={raw_env:.4f}, adj={raw_env_adj:.4f}, floor={noise_floor:.4f}, "
                    f"ref={dyn_ref:.4f}, norm={norm_env:.3f}, m={m:.2f})"
                )
                ticks = 0; timeouts = 0; last_log = now_end

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[VTS] motion loop error: {type(e).__name__}: {e!r}")
        try:
            await asyncio.sleep(0.05); await get_vts()
        except Exception:
            pass

# --- Streaming WAV parser ----------------------------------------------------

class StreamingWavToFrames:
    """Incrementally parse a WAV byte stream and emit fixed-size float32 frames."""
    def __init__(self, target_sr: int, frame_ms: int):
        self.buf = bytearray()
        self.header_done = False
        self.channels = 1
        self.bits = 16
        self.sr = target_sr
        self.frame_samples = max(1, int(target_sr * (frame_ms / 1000.0)))
        self.pcm_accum = np.zeros((0,), dtype=np.float32)

    def _parse_header(self) -> bool:
        b = self.buf
        if len(b) < 44:
            return False
        if b[0:4] != b"RIFF" or b[8:12] != b"WAVE":
            return False
        i = 12
        fmt_found = False
        data_ofs = None
        while i + 8 <= len(b):
            chunk_id = b[i:i+4]
            chunk_sz = int.from_bytes(b[i+4:i+8], "little", signed=False)
            j = i + 8
            if chunk_id == b"fmt " and j + 16 <= len(b):
                audio_format = int.from_bytes(b[j:j+2], "little")
                ch = int.from_bytes(b[j+2:j+4], "little")
                sr = int.from_bytes(b[j+4:j+8], "little")
                bits = int.from_bytes(b[j+14:j+16], "little")
                if audio_format != 1:
                    return False  # PCM only
                self.channels = ch
                self.bits = bits
                self.sr = sr
                fmt_found = True
            elif chunk_id == b"data":
                data_ofs = j
                break
            i = j + chunk_sz + (chunk_sz & 1)
        if not fmt_found or data_ofs is None:
            return False
        del self.buf[:data_ofs]
        self.header_done = True
        return True

    def push_bytes(self, chunk: bytes) -> List[np.ndarray]:
        """Feed new bytes, return any whole frames now available."""
        self.buf += chunk
        frames: List[np.ndarray] = []
        if not self.header_done and not self._parse_header():
            return frames
        if self.bits != 16:
            return frames  # only 16-bit PCM here
        bytes_per_samp = self.bits // 8
        n_samp_total = len(self.buf) // bytes_per_samp
        n_frames_interleaved = n_samp_total // self.channels
        if n_frames_interleaved <= 0:
            return frames
        take_bytes = n_frames_interleaved * self.channels * bytes_per_samp
        chunk_copy = bytes(self.buf[:take_bytes])
        arr = np.frombuffer(chunk_copy, dtype=np.int16).astype(np.float32) / 32768.0
        if self.channels > 1:
            arr = arr.reshape(-1, self.channels).mean(axis=1)
        else:
            arr = arr.reshape(-1)
        del self.buf[:take_bytes]
        if self.pcm_accum.size:
            arr = np.concatenate([self.pcm_accum, arr], axis=0)
        i = 0
        while i + self.frame_samples <= arr.size:
            frames.append(arr[i:i+self.frame_samples])
            i += self.frame_samples
        self.pcm_accum = arr[i:]
        return frames

    def flush_tail(self) -> List[np.ndarray]:
        """Pad and flush any remaining partial frame."""
        frames: List[np.ndarray] = []
        if self.pcm_accum.size:
            pad = np.zeros((self.frame_samples - self.pcm_accum.size,), dtype=np.float32)
            frames.append(np.concatenate([self.pcm_accum, pad], axis=0))
            self.pcm_accum = np.zeros((0,), dtype=np.float32)
        return frames

# --- LLM → TTS → paced playback → motion ------------------------------------

async def run_pipeline(client_llm: httpx.AsyncClient, client_tts: httpx.AsyncClient, user_text: str):
    """
    Build persona/system + history, stream from LLM, chunk on punctuation
    or stalls, synth each chunk, decode to frames, and feed audio at a
    fixed cadence so the motion tap stays aligned.
    """
    history = load_memory()
    system_prompt = build_system_prompt(PERSONA)
    fewshot = build_fewshot_block()
    prompt = fewshot + build_prompt_with_history(user_text, history)

    url = f"{OLLAMA_ENDPOINT}/api/generate"
    req = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_prompt}\n\n{prompt}",
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": OLLAMA_TEMP, "top_p": OLLAMA_TOP_P,
            "num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT,
            "stop": ["\nUser:", "\nAssistant:", "\nSYSTEM:"]
        },
    }

    audio_cb = _MOTION["audio"]; out_sr = _MOTION["sr"]
    fs = max(1, int(out_sr * (CHUNK_MS / 1000.0)))
    silent_frame = np.zeros((fs,), dtype=np.float32)
    def _push_audio(fr: np.ndarray): audio_cb.ring.push(fr)

    async def feed_frames_paced(frames: List[np.ndarray], postroll_ms: int):
        """Emit frames at CHUNK_MS cadence + a short tail."""
        for fr in frames:
            _push_audio(fr)
            await asyncio.sleep(CHUNK_MS / 1000.0)
        npost = max(1, int(postroll_ms / CHUNK_MS))
        zeros = np.zeros((fs,), dtype=np.float32)
        for _ in range(npost):
            _push_audio(zeros)
            await asyncio.sleep(CHUNK_MS / 1000.0)

    audio_feed_lock = asyncio.Lock()  # serialize per-reply feed
    first_tts = None
    buf = ""
    parts: List[str] = []
    t0 = time.time()
    last_emit_ts = time.time()

    async def synth_and_stream(text_piece: str, postroll_ms: int):
        """TTS one chunk, decode WAV to frames, and feed paced."""
        nonlocal first_tts
        clean = sanitize_text(text_piece)
        if not clean:
            return
        try:
            _push_audio(silent_frame.copy())  # tiny preroll gap

            r = await client_tts.post(
                TTS_URL, json={"text": clean},
                timeout=httpx.Timeout(HTTP_READ_S, connect=HTTP_CONNECT_S)
            )
            r.raise_for_status()
            body = r.content or b""
            if len(body) < 64 or not body.startswith(b"RIFF"):
                print(f"[TTS] Bad/short WAV (len={len(body)})."); return

            frames = await asyncio.to_thread(decode_wav_to_frames, body, out_sr, fs)

            async with audio_feed_lock:
                await feed_frames_paced(frames, postroll_ms)
                if first_tts is None and frames:
                    first_tts = time.time()
            print(f"[AUDIO] paced frames={len(frames)}")
        except Exception as e_post:
            print(f"[TTS] POST failed ({type(e_post).__name__}): {e_post}")

    tts_tasks: List[asyncio.Task] = []
    try:
        async with client_llm.stream("POST", url, json=req,
                                     timeout=httpx.Timeout(HTTP_READ_S, connect=HTTP_CONNECT_S)) as resp:
            async for line in resp.aiter_lines():
                if not line: continue
                try: data = json.loads(line)
                except Exception: continue
                chunk = (data.get("response") or "")
                if chunk:
                    buf += chunk
                    now = time.time()
                    should_flush = False

                    # Flush on punctuation or if we've stalled long enough
                    if len(buf) >= max(SENT_MIN_CHARS, EARLY_FLUSH_MIN_CHARS) and re.search(r'[.!?;,]\s+', buf):
                        should_flush = True
                    elif len(buf) >= EARLY_FLUSH_MIN_CHARS and (now - last_emit_ts) * 1000.0 >= EARLY_FLUSH_MAX_STALL_MS:
                        should_flush = True

                    if should_flush:
                        s = buf.strip(); buf = ""; last_emit_ts = now
                        if s:
                            parts.append(s)
                            tts_tasks.append(asyncio.create_task(
                                synth_and_stream(s, postroll_ms=INTER_SENTENCE_GAP_MS)
                            ))

                if data.get("done"):
                    break

        tail = buf.strip()
        if tail:
            parts.append(tail)
            tts_tasks.append(asyncio.create_task(
                synth_and_stream(tail, postroll_ms=POSTROLL_MS)
            ))
    finally:
        if tts_tasks:
            try: await asyncio.gather(*tts_tasks, return_exceptions=True)
            except Exception: pass
        # Small closing pad; then let PortAudio drain
        for _ in range(max(1, int(POSTROLL_MS / CHUNK_MS))):
            _push_audio(silent_frame.copy())
            await asyncio.sleep(CHUNK_MS / 1000.0)
        try: _MOTION["audio"].drain(max_wait_s=None, frame_ms=CHUNK_MS)
        except Exception: pass

    reply = " ".join(parts).strip()
    history.append({"user": user_text, "assistant": reply}); save_memory(history)
    if LATENCY_REPORT:
        if first_tts: print(f"[TIMINGS] LLM_total={time.time()-t0:.2f}s FirstSpeech={(first_tts-t0):.2f}s")
        else: print(f"[TIMINGS] LLM_total={time.time()-t0:.2f}s")

# --- VTS self-test -----------------------------------------------------------

async def vts_motion_self_test(duration_s: float = 2.5):
    """Quick param sweep so we know VTS is alive before the motion loop."""
    vts = await get_vts()
    if not vts or not vts.authed:
        print("[VTS] self-test aborted: VTS not authed."); return
    print("[VTS] self-test: sweeping params for ~2.5s...")
    tick_dt = max(1e-3, VTS_FRAME_MS / 1000.0)
    next_tick = time.monotonic(); t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        t = time.monotonic() - t0
        mouth = 0.5 + 0.5*np.sin(2*np.pi*1.2*t)
        ang   = 15.0*np.sin(2*np.pi*0.8*t)
        pose = {
            "VelMouthOpen": float(np.clip(mouth, 0.0, 1.0)),
            "VelFaceAngleX": float(ang),
            "VelFaceAngleY": float(-ang*0.8),
            "VelFaceAngleZ": float(ang*0.6),
            "VelBodyAngleX": float(ang*0.5),
            "VelBodyAngleY": float(-ang*0.4),
            "VelBodyAngleZ": float(ang*0.3),
        }
        await vts.inject_inline(pose, weight=1.0)
        next_tick += tick_dt
        sleep_for = next_tick - time.monotonic()
        if sleep_for <= -0.5 * tick_dt:
            next_tick = time.monotonic() + tick_dt
            sleep_for = tick_dt
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
    print("[VTS] self-test complete.")

# --- Main --------------------------------------------------------------------

async def main():
    """Bring up listeners, warm dependencies, run the PTT loop."""
    start_ptt_listener()

    print("[BOOT] Loading ASR...")
    asr = await asyncio.to_thread(ASREngine)  # load off the loop

    async with httpx.AsyncClient() as client_llm, httpx.AsyncClient() as client_tts:
        print("[BOOT] Warming up LLM/TTS/ASR...")
        await warmup(client_llm, client_tts, asr)

        # Do a quick VTS sweep before starting the motion loop
        try:
            await vts_motion_self_test()
        except Exception as e:
            print(f"[VTS] self-test error: {e!r}")

        # Start the shared audio output + motion tap
        out_sr = _MOTION["sr"]
        _MOTION["queue"] = asyncio.Queue(maxsize=FRAME_QUEUE_MAX)
        _MOTION["audio"] = CallbackOutput(out_sr, frame_ms=CHUNK_MS)
        _MOTION["audio"].start()
        _MOTION["audio"].set_tap(asyncio.get_running_loop(), _MOTION["queue"])
        _MOTION["task"] = asyncio.create_task(motion_consumer(_MOTION["queue"], out_sr))

        print("🟢 Ready. Hold your mouse side button to talk. (Ctrl+C to quit)")
        try:
            while True:
                audio_i16 = await asyncio.to_thread(record_audio_while_held_np16)
                if audio_i16 is None or audio_i16.size == 0:
                    continue

                text = asr.transcribe_np16(audio_i16, SAMPLE_RATE)
                if not text:
                    print("[ASR] Heard nothing clear. Try again.")
                    continue

                print(f"🗣️ You: {text}")
                await run_pipeline(client_llm, client_tts, text)

        except KeyboardInterrupt:
            print("\n👋 Bye.")
        finally:
            # Graceful shutdown
            try:
                if _MOTION["queue"] is not None:
                    await _MOTION["queue"].put(None)
                if _MOTION["task"] is not None:
                    await _MOTION["task"]
            except Exception:
                pass
            try:
                if _MOTION["audio"] is not None:
                    _MOTION["audio"].stop()
            except Exception:
                pass

print("[BOOT] launching from", __file__)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
