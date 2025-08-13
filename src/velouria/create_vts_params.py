# pip install websockets
import asyncio, json, os
import websockets

VTS_WS_URL = "ws://127.0.0.1:8001"
PLUGIN_NAME = "Velouria Motion Driver"
PLUGIN_DEV  = "Sean + GPT-5"
TOKEN_FILE  = ".vts_token.json"

REQUIRED_PARAMS = [
    # name,        min,   max,  default, explanation
    ("VelMouthOpen", 0.0,   1.0, 0.0,    "Velouria mouth open (0..1)"),
    ("VelFaceAngleX",-30.0, 30.0, 0.0,   "Velouria head angle X (deg)"),
    ("VelFaceAngleY",-30.0, 30.0, 0.0,   "Velouria head angle Y (deg)"),
    ("VelFaceAngleZ",-30.0, 30.0, 0.0,   "Velouria head angle Z (deg)"),
    ("VelBodyAngleX",-30.0, 30.0, 0.0,   "Velouria body angle X (deg)"),
    ("VelBodyAngleY",-30.0, 30.0, 0.0,   "Velouria body angle Y (deg)"),
    ("VelBodyAngleZ",-30.0, 30.0, 0.0,   "Velouria body angle Z (deg)"),
]

async def vts_send(ws, req_id, message_type, data):
    payload = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": str(req_id),
        "messageType": message_type,
        "data": data,
    }
    await ws.send(json.dumps(payload))
    resp = json.loads(await ws.recv())
    # If APIError, bubble up so we can decide to ignore/continue
    return resp

def load_token():
    if os.path.exists(TOKEN_FILE):
        try:
            return json.load(open(TOKEN_FILE,"r",encoding="utf-8")).get("token")
        except Exception:
            pass
    return None

def save_token(token: str):
    try:
        json.dump({"token": token}, open(TOKEN_FILE,"w",encoding="utf-8"))
    except Exception:
        pass

async def main():
    req_id = 0
    async with websockets.connect(VTS_WS_URL, max_size=4_000_000) as ws:
        # 1) auth token
        token = load_token()
        if not token:
            req_id += 1
            tok = await vts_send(ws, req_id, "AuthenticationTokenRequest", {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_DEV,
            })
            if tok.get("messageType","").endswith("Error"):
                raise RuntimeError(f"Auth token request failed: {tok}")
            token = tok["data"]["authenticationToken"]
            save_token(token)

        # 2) authenticate
        req_id += 1
        auth = await vts_send(ws, req_id, "AuthenticationRequest", {
            "pluginName": PLUGIN_NAME,
            "pluginDeveloper": PLUGIN_DEV,
            "authenticationToken": token,
        })
        if auth.get("messageType","").endswith("Error"):
            raise RuntimeError(f"Authentication failed: {auth}")

        # 3) try to create each required parameter (ignore 'already exists' errors)
        created, skipped = [], []
        for name, mn, mx, dv, expl in REQUIRED_PARAMS:
            req_id += 1
            resp = await vts_send(ws, req_id, "ParameterCreationRequest", {
                "parameterName": name,
                "explanation": expl,
                "min": mn, "max": mx, "defaultValue": dv
            })
            if resp.get("messageType","").endswith("Error"):
                # If it already exists, VTS returns an APIError. We treat that as "skip".
                skipped.append(name)
            else:
                created.append(name)

        if created:
            print("Created:", ", ".join(created))
        if skipped:
            print("Skipped (already existed):", ", ".join(skipped))
        if not created and not skipped:
            print("No response; check VTS is running with API enabled.")

if __name__ == "__main__":
    asyncio.run(main())
