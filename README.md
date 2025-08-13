# VelouriaAI

**Push-to-talk voice assistant, fully local. (Functional WIP)**  
🎤 ASR (Whisper / faster-whisper) → 🤖 LLM (Ollama) → 🔊 TTS (XTTS) → 🕺 VTube Studio motion — paced so lips + audio stay in sync.

> Hold your mouse side button to talk (Not left or right, but a mouse that has extra buttons, easy to change the default \velouria\src\velouria\settings.json "ptt_button": "x1",). Built for snappy, real-time feel on Windows. No cloud keys, no venvs in Git.

## What it does (quick)
- Transcribes mic input fast (prefers `faster-whisper`)
- Streams LLM text and **speaks as it generates**
- Plays audio at a fixed cadence so the motion loop stays aligned
- Drives VTube Studio with custom params (mouth/head/body), with mouth “chew” and consonant ducking for realism

## Commands (one big block)
```powershell
# ========= Repo setup (run at the repo root) =========
cd "$(git rev-parse --show-toplevel)"
"* text=auto" | Out-File -Encoding utf8 .gitattributes
git add .gitattributes
git commit -m "Normalize line endings" 2>$null

# ========= Python env =========
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# ========= ffmpeg =========
# Ensure ffmpeg.exe is on PATH (winget install Gyan.FFmpeg or your own build)

# ========= XTTS service (separate console) =========
cd tts_service
python server.py
# http://127.0.0.1:8123/synthesize

# ========= VTube Studio =========
# Enable API/WebSocket; token saved to src/velouria/.vts_token.json

# ========= Configure app settings =========
# Edit src/velouria/settings.json: mic index, llm model/endpoint, tts endpoint/speaker, vts.ws_url, ptt button

# ========= Run the app =========
cd "$(git rev-parse --show-toplevel)"
python .\src\velouria\app.py

# ========= Controls =========
# PTT: x1 (default) or x2; Exit: Ctrl+C

# ========= Git hygiene =========
git rm -r --cached .venv venv 2>$null
git commit -m "Stop tracking virtualenv" 2>$null

# ========= Troubleshooting =========
python - << 'PY'
import sounddevice as sd
for i, d in enumerate(sd.query_devices()):
    print(i, d['name'], 'in=', d['max_input_channels'], 'out=', d['max_output_channels'])
PY

# ========= Maintenance =========
git gc --prune=now
