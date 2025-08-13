# quick_xtts_stream_probe.py
import httpx, time

TTS_URL = "http://127.0.0.1:8123/synthesize"  # change if needed
payload = {"text": "This is a streaming probe. One two three four five."}

t0 = time.time()
with httpx.stream("POST", TTS_URL, json=payload, timeout=60.0) as resp:
    resp.raise_for_status()
    total = 0
    chunks = 0
    for chunk in resp.iter_bytes():
        if not chunk:
            continue
        if chunks == 0:
            print(f"first bytes at +{time.time()-t0:.3f}s")
        total += len(chunk); chunks += 1
        if chunks <= 8:
            print(f"chunk {chunks} (+{time.time()-t0:.3f}s, {len(chunk)} bytes)")
    print(f"done: {chunks} chunks, {total} bytes, total +{time.time()-t0:.3f}s")