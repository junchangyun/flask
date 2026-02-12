from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

running = False
key_mask = "-"
status_msg = "idle"

def mask(k: str) -> str:
    if not k:
        return "-"
    if len(k) <= 8:
        return k[:2] + "****"
    return k[:4] + "****" + k[-4:]

@app.get("/")
def home():
    return "OK - Bybit Trade Journal API"

@app.get("/status")
def status():
    return jsonify({"running": running, "key_mask": key_mask, "status": status_msg})

@app.post("/start")
def start():
    global running, key_mask, status_msg
    data = request.get_json(silent=True) or request.form
    api_key = data.get("bybit_api_key")
    secret = data.get("bybit_secret_key")
    if not api_key or not secret:
        return "키가 비었습니다.", 400
    running = True
    key_mask = mask(api_key)
    status_msg = "started"
    return "모니터 시작", 200

@app.post("/stop")
def stop():
    global running, status_msg
    running = False
    status_msg = "stopped"
    return "모니터 중지", 200
