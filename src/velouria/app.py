import os
import io
import re
import json
import time
import tempfile
import threading
import asyncio
import base64
import math
import random
import unicodedata
import string
import concurrent.futures

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
import requests
import whisper  # openai-whisper
from pynput import mouse
import websockets  # 15.x+

SAFE_ASCII = set(string.ascii_letters + string.digits + " .,!?'-")

# NEW: VTube Studio WS client (websocket-client)
try:
    from websocket import WebSocketApp
except Exception:
    WebSocketApp = None

# ---- Config ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")

with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    settings = json.load(f)

MIC_DEVICE_INDEX = settings.get("microphone_device_index", None)
WHISPER_MODEL    = settings.get("whisper_model", "tiny.en")  # faster for English
SAMPLE_RATE      = 16000

# Local TTS service (XTTS-v2 FastAPI)
LOCAL_TTS_URL    = (settings.get("local_tts_url", "http://127.0.0.1:8123/synthesize") or "").rstrip("/")
# Device hint; we only use this at warmup or explicit CPU fallback.
TTS_DEVICE_DEFAULT = (settings.get("tts_device", "auto") or "auto").lower()  # "auto" | "cuda" | "cpu"

# ---- LLM / Ollama ----
OLLAMA_ENABLED   = bool(settings.get("ollama_enabled", True))
OLLAMA_ENDPOINT  = (settings.get("ollama_endpoint", "http://127.0.0.1:11434") or "").rstrip("/")
OLLAMA_MODEL     = settings.get("ollama_model", "velouria")
OLLAMA_TIMEOUT_S = float(settings.get("ollama_timeout_seconds", 10.0))
OLLAMA_COLD_TIMEOUT_S = float(settings.get("ollama_cold_timeout_seconds", 20.0))
OLLAMA_BG_KEEPALIVE_SEC = int(settings.get("ollama_keepalive_seconds", 300))

# ---- VTS Settings ----
VTS_ENABLED         = bool(settings.get("vts_enabled", True))
VTS_URL             = settings.get("vts_url", "ws://localhost:8001")
VTS_PLUGIN_NAME     = settings.get("vts_plugin_name", "Velouria")
VTS_PLUGIN_DEV      = settings.get("vts_plugin_developer", "Sean")
VTS_PLUGIN_VERSION  = settings.get("vts_plugin_version", "1.0")
VTS_TOKEN           = settings.get("vts_token", "")
VTS_DRIVE_HEAD      = bool(settings.get("vts_drive_head", True))

# Trim/curve knobs
MOUTH_TRIM  = float(settings.get("vts_mouth_trim", 0.65))
MOUTH_BIAS  = float(settings.get("vts_mouth_bias", 0.02))
MOUTH_GAMMA = float(settings.get("vts_mouth_gamma", 1.25))
HEAD_TRIM   = float(settings.get("vts_head_trim", 0.9))

# Behavior toggles
VTS_FORCE_TEST_ON_AUTH   = bool(settings.get("vts_force_test_on_auth", False))
VTS_PARAMS_PROBE_ON_AUTH = bool(settings.get("vts_params_probe_on_auth", False))
VTS_INPUTS_MODE          = (settings.get("vts_inputs_mode", "add") or "add").lower()
VTS_DEBUG_RAW            = bool(settings.get("vts_debug_raw", False))

# Input parameter IDs
INPUT_MOUTH_OPEN = settings.get("vts_input_mouth_open", "MouthOpen")
INPUT_MOUTH_FORM = settings.get("vts_input_mouth_form", "")
INPUT_HEAD_PITCH = settings.get("vts_input_head_pitch", "FaceAngleY")
INPUT_HEAD_YAW   = settings.get("vts_input_head_yaw",   "FaceAngleX")
INPUT_HEAD_ROLL  = settings.get("vts_input_head_roll",  "FaceAngleZ")
MOUTH_INPUT_GAIN = float(settings.get("vts_mouth_input_gain", 0.6))  # 0..1
ANALYSIS_PEAK_TARGET = float(settings.get("vts_analysis_peak", 0.75))  # 0..1, soft limiter

# Raw Live2D head param names
HEAD_PARAM_NAMES = settings.get("vts_head_params", ["ParamAngleX", "ParamAngleY", "ParamAngleZ"])
HEAD_PARAM_NAMES_FALL = settings.get("vts_head_params_fallback", ["ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ"])

# ---- Load Whisper ONCE ----
print(f"🔧 Loading Whisper model: {WHISPER_MODEL}")
try:
    WHISPER = whisper.load_model(WHISPER_MODEL, device="cuda")
    print("🟩 Whisper on CUDA")
except Exception as e:
    print(f"🟨 CUDA unavailable ({e}); falling back to CPU")
    WHISPER = whisper.load_model(WHISPER_MODEL, device="cpu")

# --- PTT config/state ---
PTT_BUTTON = settings.get("ptt_button", "x1").lower()
MAX_HOLD   = int(settings.get("max_hold_seconds", 30))
BEEPS      = bool(settings.get("beeps", True))

hold_lock  = threading.Lock()
is_holding = False  # toggled by mouse listener

def beep(freq=880, ms=90):
    if not BEEPS:
        return
    t = np.linspace(0, ms/1000, int(SAMPLE_RATE*ms/1000), False)
    tone = 0.1*np.sin(2*np.pi*freq*t).astype(np.float32)
    sd.play(tone, SAMPLE_RATE); sd.wait()

# ========== VTS INTEGRATION ==========
_vts_ws = None
_vts_connected = False
_vts_authed = False

ANGLE_ALIASES = [
    ("ParamAngleX", "PARAM_ANGLE_X"),
    ("ParamAngleY", "PARAM_ANGLE_Y"),
    ("ParamAngleZ", "PARAM_ANGLE_Z"),
]
_actual_head_ids = None

_idle_running = False
_speaking = False

def _save_token_to_settings(token: str):
    try:
        settings["vts_token"] = token
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        print("✅ Saved VTS plugin token to settings.json")
    except Exception as e:
        print("⚠️ Couldn't save VTS token:", e)

def _vts_send(obj: dict):
    global _vts_ws, _vts_connected
    if not (_vts_ws and _vts_connected):
        return
    try:
        _vts_ws.send(json.dumps(obj))
    except Exception as e:
        print("VTS send error:", e)
        _vts_connected = False

def _pick_first_present(candidates, have):
    for n in candidates:
        if n in have:
            return n
    return None

def _vts_request_live2d_params():
    _vts_send({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "Live2DParams",
        "messageType": "Live2DParameterListRequest",
        "data": {}
    })

def vts_force_inputs_test():
    if not (_vts_ws and _vts_connected and _vts_authed):
        return
    def _runner():
        for val in (1.0, -1.0, 0.0):
            _vts_send({
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "force_inputs",
                "messageType": "InjectParameterDataRequest",
                "data": {
                    "faceFound": True,
                    "mode": "set",
                    "parameterValues": [
                        {"id": INPUT_HEAD_YAW,   "value": val},
                        {"id": INPUT_HEAD_PITCH, "value": 0.0},
                        {"id": INPUT_HEAD_ROLL,  "value": 0.0},
                    ]
                }
            })
            time.sleep(0.6)
        print("✅ Inputs test sent (set FaceAngle ±1). Toggle camera tracking OFF to see it clearly.")
    threading.Thread(target=_runner, daemon=True).start()

def vts_force_raw_test_all():
    if not (_vts_ws and _vts_connected and _vts_authed):
        return
    ids_primary  = _actual_head_ids or HEAD_PARAM_NAMES
    ids_fallback = HEAD_PARAM_NAMES_FALL
    def _runner():
        for val in (30.0, -30.0, 0.0):
            for ids in (ids_primary, ids_fallback):
                try:
                    _vts_send({
                        "apiName": "VTubeStudioPublicAPI",
                        "apiVersion": "1.0",
                        "requestID": "force_raw",
                        "messageType": "SetLive2DParameterValuesRequest",
                        "data": {"parameterValues": [{"id": ids[0], "value": float(val)}]}
                    })
                except Exception:
                    pass
            time.sleep(0.6)
        print("✅ RAW test sent on discovered + fallback IDs (±30° yaw).")
    threading.Thread(target=_runner, daemon=True).start()

def _vts_on_message(ws, message):
    global _vts_authed, _actual_head_ids
    try:
        data = json.loads(message)
    except Exception:
        print("VTS [raw]:", message)
        return

    mtype = data.get("messageType")

    if mtype in ("InjectParameterDataResponse", "SetLive2DParameterValuesResponse"):
        rid = data.get("requestID")
        print(f"↩️  VTS ack {mtype} for {rid}, errorID={data.get('errorID')}, msg={data.get('message')}")

    if data.get("errorID", 0) != 0 and mtype not in ("InjectParameterDataResponse", "SetLive2DParameterValuesResponse"):
        print("❗VTS error:", data.get("message", data))

    if mtype == "AuthenticationTokenResponse":
        tok = data.get("data", {}).get("authenticationToken")
        if tok:
            print("🔐 VTS granted token.")
            _save_token_to_settings(tok)

    if mtype == "AuthenticationResponse":
        _vts_authed = bool(data.get("data", {}).get("authenticated", False))
        print("🔓 VTS authenticated:", _vts_authed)
        if _vts_authed:
            if VTS_FORCE_TEST_ON_AUTH:
                vts_force_inputs_test()
                vts_force_raw_test_all()
            if VTS_PARAMS_PROBE_ON_AUTH:
                _vts_request_live2d_params()
        else:
            print("❌ Auth failed. If you just clicked Allow, restart so it uses the saved token.")

    if mtype == "Live2DParameterListResponse":
        try:
            params = [p.get("name") or p.get("id") for p in data.get("data", {}).get("parameters", [])]
            have = {n for n in params if n}
            x = _pick_first_present([a for a in ANGLE_ALIASES[0]], have)
            y = _pick_first_present([a for a in ANGLE_ALIASES[1]], have)
            z = _pick_first_present([a for a in ANGLE_ALIASES[2]], have)
            if x and y and z:
                _actual_head_ids = [x, y, z]
                print(f"✅ Using head params (auto-mapped): {_actual_head_ids}")
            else:
                _actual_head_ids = None
                print("⚠️ Could not map ParamAngle* automatically; using configured names.")
            needed = set(HEAD_PARAM_NAMES) | set(HEAD_PARAM_NAMES_FALL)
            missing = [n for n in needed if n not in have]
            print(f"🧪 Live2D params found: {len(have)}; missing among configured: {missing}")
        except Exception as e:
            print("⚠️ Param probe parse error:", e)

def _vts_on_open(ws):
    global _vts_connected
    _vts_connected = True
    if VTS_TOKEN:
        _vts_send({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": VTS_PLUGIN_NAME,
                "pluginDeveloper": VTS_PLUGIN_DEV,
                "pluginVersion": VTS_PLUGIN_VERSION,
                "authenticationToken": VTS_TOKEN
            }
        })
    else:
        print("👉 In VTube Studio → Settings → Plugins, click **Allow** for Velouria when prompted.")
        _vts_send({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "tokenRequest",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": VTS_PLUGIN_NAME,
                "pluginDeveloper": VTS_PLUGIN_DEV
            }
        })

def _vts_on_error(ws, error):
    print("💥 VTS websocket error:", error)

def _vts_on_close(ws, code, reason):
    global _vts_connected, _vts_authed
    _vts_connected = False
    _vts_authed = False
    print(f"VTS connection closed: {code} {reason}")

def vts_connect():
    global _vts_ws
    if not VTS_ENABLED:
        print("ℹ️ VTS disabled in settings.")
        return
    if WebSocketApp is None:
        print("⚠️ Missing dependency 'websocket-client'. Run: pip install websocket-client")
        return
    try:
        _vts_ws = WebSocketApp(
            VTS_URL,
            on_open=_vts_on_open,
            on_message=_vts_on_message,
            on_error=_vts_on_error,
            on_close=_vts_on_close
        )
        t = threading.Thread(target=_vts_ws.run_forever, daemon=True)
        t.start()
        print(f"🔌 Connecting to VTube Studio at {VTS_URL} ...")
    except Exception as e:
        print(f"⚠️ Failed to connect to VTS: {e}")

# ---- Prosody & motion helpers ----
class Prosody:
    def __init__(self, sr: int):
        self.sr = sr
        self.env = 0.0
        self.noise = 0.003
        self.peak = 0.0
        self.t = 0.0
        self.last_above_t = 0.0

    def step_time(self, n_samples: int):
        self.t += n_samples / float(self.sr)

    def track(self, x16: np.ndarray):
        if x16.size == 0:
            return 0.0, 0.0, 0.0
        xf = x16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(xf * xf)))

        if rms < self.noise * 1.5:
            self.noise = 0.995 * self.noise + 0.005 * rms

        a_att, a_rel = 0.28, 0.04
        self.env = (1 - (a_att if rms > self.env else a_rel)) * self.env + (a_att if rms > self.env else a_rel) * rms

        pk = float(np.max(np.abs(xf)))
        p_att, p_rel = 0.35, 0.02
        self.peak = (1 - (p_att if pk > self.peak else p_rel)) * self.peak + (p_att if pk > self.peak else p_rel) * pk

        gate = self.noise * 2.2 + 0.002
        if rms > gate:
            self.last_above_t = self.t

        return rms, self.env, self.peak

prosody = Prosody(22050)

_mouth_follow = 0.0
def _map_openness(x16: np.ndarray) -> float:
    global _mouth_follow
    _, env, peak = prosody.track(x16)
    prosody.step_time(x16.size)

    gate = prosody.noise * 2.2 + 0.002
    drive = max(0.0, env - gate) * 14.0
    comp  = math.tanh(drive)
    burst = min(1.0, peak * 1.1)

    target = 0.85 * comp + 0.15 * burst

    fast = 0.35
    slow = 0.08
    very_fast = 0.65
    since_voice_ms = (prosody.t - prosody.last_above_t) * 1000.0
    release = very_fast if since_voice_ms > 60.0 else slow

    if target > _mouth_follow:
        _mouth_follow = (1 - fast) * _mouth_follow + fast * target
    else:
        _mouth_follow = (1 - release) * _mouth_follow + release * target

    if since_voice_ms > 140.0:
        _mouth_follow *= 0.6
        if since_voice_ms > 260.0:
            _mouth_follow *= 0.5

    return max(0.0, min(1.0, _mouth_follow))

_head_pitch = 0.0
_yaw_phase  = random.random()*math.tau
def _map_head(x16: np.ndarray, sr: int):
    global _head_pitch, _yaw_phase

    openv = _map_openness(x16)
    form  = max(0.0, min(1.0, 0.5 + (openv - 0.5)*0.4))

    _, env, _ = prosody.track(x16)
    _head_pitch = 0.88*_head_pitch + 0.18*(env*10.0)
    pitch = max(-1.0, min(1.0, _head_pitch))

    prosody.step_time(x16.size)
    freq = 0.28
    _yaw_phase += (freq * (x16.size/float(sr))) * math.tau
    yaw  = 0.95*math.sin(_yaw_phase)

    roll = 0.35*yaw + 0.12*math.sin(prosody.t*1.8*math.tau)
    return openv, form, pitch, yaw, roll

def _finite(x, default=0.0):
    try:
        if x is None or isinstance(x, bool): return float(default)
        x = float(x)
        if math.isfinite(x): return x
    except Exception:
        pass
    return float(default)

def vts_set_speech_pose(openness: float, mouth_form: float, pitch: float, yaw: float, roll: float):
    if not (_vts_ws and _vts_connected and _vts_authed and VTS_ENABLED):
        return

    def c01(x): return max(0.0, min(1.0, _finite(x)))
    def c11(x): return max(-1.0, min(1.0, _finite(x)))

    o = c01(openness)
    if MOUTH_GAMMA > 0:
        o = pow(o, MOUTH_GAMMA)
    o = c01(o * MOUTH_TRIM + MOUTH_BIAS)
    mf = c01(mouth_form) if INPUT_MOUTH_FORM else 0.5

    mouth_vals = [{"id": INPUT_MOUTH_OPEN, "value": o}]
    if INPUT_MOUTH_FORM:
        mouth_vals.append({"id": INPUT_MOUTH_FORM, "value": mf})
    _vts_send({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "speechPose_mouth",
        "messageType": "InjectParameterDataRequest",
        "data": {"faceFound": True, "mode": "set", "parameterValues": mouth_vals}
    })

    if not VTS_DRIVE_HEAD:
        return

    p = c11(pitch) * 0.9 * HEAD_TRIM
    y = c11(yaw)   * 0.9 * HEAD_TRIM
    r = c11(roll)  * 0.6 * HEAD_TRIM

    _vts_send({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "speechPose_head",
        "messageType": "InjectParameterDataRequest",
        "data": {
            "faceFound": True,
            "mode": ("add" if VTS_INPUTS_MODE not in {"set"} else "set"),
            "parameterValues": [
                {"id": INPUT_HEAD_YAW,   "value": y},
                {"id": INPUT_HEAD_PITCH, "value": p},
                {"id": INPUT_HEAD_ROLL,  "value": r},
            ]
        }
    })

    _vts_send_raw_head(pitch=p, yaw=y, roll=r, names=None,                  scale_deg=40.0)
    _vts_send_raw_head(pitch=p, yaw=y, roll=r, names=HEAD_PARAM_NAMES_FALL, scale_deg=40.0)

def _vts_send_raw_head(pitch, yaw, roll, names=None, scale_deg=30.0):
    if not (_vts_ws and _vts_connected and _vts_authed and VTS_ENABLED):
        return
    use_names = names or _actual_head_ids or HEAD_PARAM_NAMES
    vals = []
    try:
        if len(use_names) >= 1 and use_names[0]:
            vals.append({"id": use_names[0], "value": float(yaw)   * scale_deg})
        if len(use_names) >= 2 and use_names[1]:
            vals.append({"id": use_names[1], "value": float(pitch) * scale_deg})
        if len(use_names) >= 3 and use_names[2]:
            vals.append({"id": use_names[2], "value": float(roll)  * scale_deg})
    except Exception:
        return
    if not vals:
        return

    if VTS_DEBUG_RAW:
        print("→ RAW set:", vals)

    _vts_send({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "head_raw",
        "messageType": "SetLive2DParameterValuesRequest",
        "data": {"parameterValues": vals}
    })

# ---- Idle micro-motion ----
def _idle_loop():
    global _idle_running
    if _idle_running:
        return
    _idle_running = True

    phase = random.random() * math.tau
    while True:
        time.sleep(0.05)
        if not (_vts_ws and _vts_connected and _vts_authed and VTS_ENABLED):
            continue
        if _speaking or not VTS_DRIVE_HEAD:
            continue

        phase += 0.05
        yaw   = 0.55 * math.sin(phase * 0.35)
        roll  = 0.22 * math.sin(phase * 0.9 + 1.1)
        pitch = 0.30 * math.sin(phase * 0.5 + 2.0)

        p = max(-1, min(1, pitch)) * 0.9 * HEAD_TRIM
        y = max(-1, min(1, yaw))   * 0.9 * HEAD_TRIM
        r = max(-1, min(1, roll))  * 0.6 * HEAD_TRIM

        _vts_send_raw_head(pitch=p, yaw=y, roll=r, names=None,                  scale_deg=40.0)
        _vts_send_raw_head(pitch=p, yaw=y, roll=r, names=HEAD_PARAM_NAMES_FALL, scale_deg=40.0)

# =====================================

# ---- Helpers ----
def ensure_input_device(device_index):
    try:
        info = sd.query_devices(device_index)
        if info["max_input_channels"] < 1:
            raise ValueError(f"Device {device_index} has no input channels.")
        return device_index
    except Exception:
        print("⚠️  Invalid or missing microphone_device_index in settings.json.")
        print("🔎 Available devices:")
        print(sd.query_devices())
        raise SystemExit("Fix settings.json → microphone_device_index and rerun.")

def _pynput_button_from_str(s: str):
    from pynput.mouse import Button
    return Button.x2 if s == "x2" else Button.x1

def start_ptt_listener():
    target = _pynput_button_from_str(PTT_BUTTON)
    def on_click(x, y, button, pressed):
        global is_holding
        if button == target:
            with hold_lock:
                is_holding = pressed
    listener = mouse.Listener(on_click=on_click)
    listener.daemon = True
    listener.start()

def record_audio_while_held():
    dev = ensure_input_device(MIC_DEVICE_INDEX)
    print("🎙️ Hold your side mouse button to talk… release to stop.")

    frames = []
    start_ts = None

    def callback(indata, frames_count, time_info, status):
        if status:
            pass
        with hold_lock:
            if is_holding:
                frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=1024, device=dev, callback=callback):
        while True:
            with hold_lock:
                if is_holding:
                    start_ts = time.time()
                    break
            time.sleep(0.01)

        beep(880, 80)
        print("🎤 Recording…")

        while True:
            with hold_lock:
                holding = is_holding
            if not holding:
                break
            if (time.time() - start_ts) > MAX_HOLD:
                print("⏱️ Reached max_hold_seconds; stopping.")
                break
            time.sleep(0.01)

    if not frames:
        print("…No audio captured.")
        return None

    beep(660, 80)
    audio = np.concatenate(frames, axis=0)
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(tmp_wav.name, SAMPLE_RATE, audio)
    print(f"💾 Saved recording to {tmp_wav.name}")
    return tmp_wav.name

def transcribe_audio(file_path):
    print(f"🪶 Transcribing with Whisper ({WHISPER_MODEL}) ...")
    result = WHISPER.transcribe(file_path)
    text = (result.get("text") or "").strip()
    print("📝 Transcription result:")
    print(text)
    return text

# ---------- Persona + Memory ----------
MEMORY_PATH = os.path.join(BASE_DIR, "memory.json")
MAX_TURNS   = int(settings.get("max_history_turns", 6))

def load_memory():
    if not os.path.exists(MEMORY_PATH):
        return []
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_memory(history):
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history[-MAX_TURNS:], f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def build_system_persona():
    name   = settings.get("persona_name", "Velouria")
    tone   = settings.get("persona_tone", "")
    rules  = settings.get("persona_rules", "")
    flavor = settings.get("domain_flavor", "")
    return (
        f"You are {name}, a helpful AI companion.\n"
        f"Tone: {tone}\n"
        f"Flavor: {flavor}\n"
        f"Rules: {rules}\n"
        "When speaking aloud, keep responses short and practical (1–2 sentences). "
        "Prefer simple words and minimal clauses. If unsure, say so briefly and suggest a next step."
    )

def build_prompt_with_history(user_text, history):
    lines = []
    for turn in history[-MAX_TURNS:]:
        u = turn.get("user", "").strip()
        a = turn.get("assistant", "").strip()
        if u: lines.append(f"User: {u}")
        if a: lines.append(f"Assistant: {a}")
    lines.append(f"User: {user_text.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)

def _ollama_payload(prompt: str, system: str):
    return {
        "model": OLLAMA_MODEL,
        "prompt": f"{system}\n\n{prompt}",
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.5,
            "num_predict": 96,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "stop": ["\nUser:", "\nAssistant:", "\nSYSTEM:"]
        }
    }

# ---- Ollama robust client ----
_OLLAMA_SESSION = requests.Session()

def _ollama_call(payload, timeout_connect=2.0, timeout_read=25.0):
    url = f"{OLLAMA_ENDPOINT}/api/generate"
    return _OLLAMA_SESSION.post(url, json=payload, timeout=(timeout_connect, timeout_read))

def ask_ollama(prompt: str, system: str = "You are concise, friendly, and brief. Answer in 1–2 sentences."):
    if not OLLAMA_ENABLED or not OLLAMA_ENDPOINT:
        return "Okay."

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system}\n\nUser: {prompt}\nAssistant:",
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.5,
            "num_predict": int(settings.get("ollama_num_predict", 128)),  # was 96
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            # ✨ only role tags; no '.', '!', '?', or bare '\n'
            "stop": ["\nUser:", "\nAssistant:", "\nSYSTEM:"]
        }
    }

    try:
        r = _ollama_call(payload, timeout_connect=2.0, timeout_read=25.0)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip() or "Okay."
    except requests.exceptions.ReadTimeout:
        print("ℹ️ Ollama: read timeout on normal path. Model may be cold or swapping. Attempting warm ping...")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Ollama quick attempt failed ({e.__class__.__name__}: {e}). Attempting warm ping...")

    warm_payload = {
        "model": OLLAMA_MODEL,
        "prompt": "User: say 'ready'\nAssistant:",
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.0, "num_predict": 1}
    }
    try:
        r = _ollama_call(warm_payload, timeout_connect=2.0, timeout_read=45.0)
        r.raise_for_status()
        r2 = _ollama_call(payload, timeout_connect=2.0, timeout_read=25.0)
        r2.raise_for_status()
        data = r2.json()
        return (data.get("response") or "").strip() or "Okay."
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama not reachable after warm ping ({e.__class__.__name__}: {e}). Using fallback.")
        return "Okay."

# ---- Background keep-alive ping (cheap) ----
def _ollama_bg_keepalive():
    if not (OLLAMA_ENABLED and OLLAMA_ENDPOINT and OLLAMA_BG_KEEPALIVE_SEC > 0):
        return
    url = f"{OLLAMA_ENDPOINT}/api/tags"
    while True:
        try:
            requests.get(url, timeout=1.5)
        except Exception:
            pass
        time.sleep(OLLAMA_BG_KEEPALIVE_SEC)

# ------------------- TTS helpers -------------------
_EMOJI_RE = re.compile(
    "["                       # conservative emoji ranges
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+", flags=re.UNICODE
)
_ALLOWED_CHARS_RE = re.compile(r"[^a-zA-Z0-9\s\.\,\!\?\;\:\-\'\"/()]+", re.UNICODE)

def sanitize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = s.replace("—", "-").replace("–", "-")
    s = _EMOJI_RE.sub("", s)
    s = "".join(ch for ch in s if ch == "\n" or (unicodedata.category(ch)[0] != "C"))
    s = _ALLOWED_CHARS_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class _CudaTTSException(Exception):
    pass

def _synth_to_segment(text: str, force_cpu: bool = False) -> AudioSegment:
    payload = {"text": text}

    # Prefer server-side default voice/embedding; do NOT pass remote speaker_wav each call.
    sp     = settings.get("tts_speaker") or ""
    if sp:
        payload["speaker"] = sp

    # Only force CPU when needed. Otherwise, omit device so server stays on CUDA once warmed.
    if force_cpu:
        payload["device"] = "cpu"

    url = LOCAL_TTS_URL
    r = requests.post(url, json=payload, timeout=45)

    ctype = r.headers.get("Content-Type", "")
    if r.status_code != 200 or "audio" not in ctype:
        msg = ""
        try:
            msg = r.json().get("error", "")
        except Exception:
            msg = r.text or ""
        if "CUDA error" in msg and not force_cpu:
            raise _CudaTTSException(msg)
        raise RuntimeError(f"TTS {r.status_code} (Content-Type={ctype}) {msg[:200]}")

    return AudioSegment.from_file(io.BytesIO(r.content), format="wav")

# Add these near your other settings pulls:
MOUTH_INPUT_GAIN = float(settings.get("vts_mouth_input_gain", 0.6))  # 0..1
ANALYSIS_PEAK_TARGET = float(settings.get("vts_analysis_peak", 0.75))  # 0..1, soft limiter

def _play_np_audio_with_vts(np_audio: np.ndarray, samplerate: int, chunk_ms: int = 20):
    """Play numpy float32 [-1,1] audio and drive a conversational pose (with input gain + soft limiting for VTS)."""
    global _speaking
    # --- normalize to float32 mono for output ---
    if np_audio.dtype != np.float32:
        np_audio = np_audio.astype(np.float32)
    if np_audio.ndim == 2 and np_audio.shape[1] > 1:
        mono = np.mean(np_audio, axis=1)
    else:
        mono = np_audio.reshape(-1)

    # Int16 stream for actual audio playback (unchanged loudness)
    int16_audio = (mono * 32767.0).clip(-32768, 32767).astype(np.int16)

    chunk_samples = max(1, int(samplerate * (chunk_ms / 1000.0)))
    _speaking = True
    try:
        with sd.OutputStream(samplerate=samplerate, channels=1, dtype="int16") as out:
            pos = 0
            N = int16_audio.shape[0]
            while pos < N:
                end = min(N, pos + chunk_samples)
                chunk_i16 = int16_audio[pos:end]

                if chunk_i16.size > 0:
                    # --- build a *separate* analysis signal for VTS mapping ---
                    # to float in [-1,1]
                    analysis = chunk_i16.astype(np.float32) / 32768.0

                    # base attenuation
                    analysis *= MOUTH_INPUT_GAIN

                    # soft limiter toward ANALYSIS_PEAK_TARGET
                    # (keeps peaks from ever hitting 1.0 → prevents full-mouth peg)
                    peak = float(np.max(np.abs(analysis))) if analysis.size else 0.0
                    if peak > 1e-6 and peak > ANALYSIS_PEAK_TARGET:
                        analysis *= (ANALYSIS_PEAK_TARGET / peak)

                    # back to int16 for the existing _map_head(int16) API
                    analysis_i16 = (np.clip(analysis, -1.0, 1.0) * 32767.0).astype(np.int16)

                    # drive VTS from the *tamed* analysis signal
                    op, form, pitch, yaw, roll = _map_head(analysis_i16, samplerate)
                    vts_set_speech_pose(op, form, pitch, yaw, roll)

                    # play the original audio chunk (full loudness to speakers)
                    try:
                        out.write(chunk_i16)
                    except Exception:
                        pass

                pos = end
    except KeyboardInterrupt:
        pass
    finally:
        _speaking = False
        vts_set_speech_pose(0.0, 0.5, 0.0, 0.0, 0.0)


_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

def _chunk_sentences(text: str, target_len: int = 180) -> list[str]:
    # split on sentence boundaries, then pack to ~target_len
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return []
    chunks, buf = [], ""
    for s in sents:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= target_len:
            buf = f"{buf} {s}"
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks

_EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def _tts_future(chunk):
    return _EXEC.submit(_synth_to_segment, chunk, False)

def speak_text(text: str):
    """
    Low-latency TTS:
      - Sanitize
      - Chunk to ~180 chars
      - Prefetch next chunk
      - For each chunk: synth (CUDA; fallback CPU on CUDA error) -> play immediately
    """
    clean = sanitize_text(text)
    if not clean:
        return

    parts = _chunk_sentences(clean, target_len=180) or [clean]

    # kick off first
    fut = _tts_future(parts[0])
    for i in range(len(parts)):
        # prefetch next
        nxt = _tts_future(parts[i+1]) if i+1 < len(parts) else None

        seg = None
        t_s = time.time()
        try:
            seg = fut.result(timeout=30)
        except _CudaTTSException as e:
            print(f"❌ TTS CUDA failed (chunk {i+1}/{len(parts)}). Retrying on CPU: {e}")
            seg = _synth_to_segment(parts[i], force_cpu=True)
        except Exception as e:
            print(f"❌ TTS failed (chunk {i+1}): {e}")
            fut = nxt
            continue
        t_e = time.time()
        print(f"[TTS] synth={t_e - t_s:.2f}s len={len(parts[i])} chars")

        if seg:
            samples = np.array(seg.get_array_of_samples()).astype(np.float32) / (2**15)
            np_audio = samples.reshape((-1, seg.channels))
            _play_np_audio_with_vts(np_audio, seg.frame_rate)

        fut = nxt

# ---- Warmup (kill cold-start lag) ----
def warmup():
    # Whisper warmup (quick no-op transcribe)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        write(tmp.name, SAMPLE_RATE, np.zeros((SAMPLE_RATE // 2,), dtype="int16"))
        try:
            _ = WHISPER.transcribe(tmp.name)
        except Exception:
            pass
    finally:
        try: os.remove(tmp.name)
        except: pass

    # TTS warmup: set device once & compile kernels
    try:
        dev = "cpu" if TTS_DEVICE_DEFAULT == "cpu" else "cuda"
        rr = requests.post(LOCAL_TTS_URL, json={"text": "ready", "device": dev}, timeout=30)
        print("🔊 TTS warmup:", rr.status_code, rr.headers.get("Content-Type",""))
    except Exception as e:
        print("ℹ️ TTS warmup skipped:", e)

    # Ollama warmup — force model load with a 1-token gen
    if OLLAMA_ENABLED and OLLAMA_ENDPOINT:
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": "User: say 'ready'\nAssistant:",
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.0, "num_predict": 1}
            }
            r = requests.post(f"{OLLAMA_ENDPOINT}/api/generate", json=payload, timeout=(2.0, 60.0))
            r.raise_for_status()
            print("🔥 Ollama warmed.")
        except Exception as e:
            print("ℹ️ Ollama warmup skipped/failed:", e)

    # Keepalive
    if OLLAMA_ENABLED and OLLAMA_ENDPOINT and OLLAMA_BG_KEEPALIVE_SEC > 0:
        threading.Thread(target=_ollama_bg_keepalive, daemon=True).start()

# ---- Main ----
if __name__ == "__main__":
    warmup()
    if VTS_ENABLED:
        vts_connect()
        threading.Thread(target=_idle_loop, daemon=True).start()

    # PTT
    start_ptt_listener()
    print("🖱️ Push-to-talk ready. Hold your side mouse button to speak.")

    while True:
        wav_file = record_audio_while_held()
        if not wav_file:
            continue
        try:
            t0 = time.time()
            user_text = transcribe_audio(wav_file)
            t1 = time.time()
            if not user_text.strip():
                print("…Heard nothing useful.")
                continue

            print(f"🤔 LLM thinking about: {user_text!r}")
            history = load_memory()
            system_prompt = build_system_persona()
            prompt = build_prompt_with_history(user_text, history)

            reply = ask_ollama(prompt, system=system_prompt)
            t2 = time.time()
            print("🗣️ Assistant:", reply)
            speak_text(reply)
            t3 = time.time()
            print(f"[TIMINGS] ASR={t1-t0:.2f}s  LLM={t2-t1:.2f}s  TTS={t3-t2:.2f}s  TOTAL={t3-t0:.2f}s")

            history.append({"user": user_text, "assistant": reply})
            save_memory(history)
        finally:
            if wav_file and os.path.exists(wav_file):
                os.remove(wav_file)
