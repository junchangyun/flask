# app.py
import os
import time
import csv
import base64
import threading
from datetime import datetime

import ccxt
import pandas as pd
import mplfinance as mpf
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

# =========================
# 기본 설정
# =========================
TIMEFRAME = "15m"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "chart")
RECORD_DIR = os.path.join(BASE_DIR, "record")
CSV_PATH = os.path.join(RECORD_DIR, "trading_journal.csv")

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(RECORD_DIR, exist_ok=True)

# OpenAI 키는 서버 고정 (env로 받는 걸 추천하지만 MVP로 그대로 둠)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# 전역 상태 (MVP)
# =========================
app = Flask(__name__)

state_lock = threading.Lock()
monitor_thread = None
monitor_running = False

exchange = None
key_mask = "-"
last_status = "idle"
last_order_id = None

recent_records = []  # 최근 10건(웹 표시용)


# =========================
# 유틸 함수들 (원래 코드 기반)
# =========================
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_chart_with_gpt(image_path: str, symbol: str, side: str) -> str:
    if client is None:
        return "분석 생략(OpenAI 미설정)"

    print(f"🤖 AI가 {symbol} 차트를 기술적으로 분석 중...")
    try:
        b64 = encode_image(image_path)
        prompt_text = (
            f"이 차트는 {symbol}의 15분봉 차트다. "
            f"초록색 화살표(▲)는 Buy, 빨간색 화살표(▼)는 Sell 지점이다. "
            f"내 포지션은 {side}였다. "
            f"오직 기술적 분석 관점(캔들 패턴, 지지/저항, 추세선)에서 "
            f"진입과 청산 자리가 적절했는지 평가해줘. 핵심만 3줄 요약."
        )

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional Technical Analyst. Focus only on chart analysis."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]}
            ],
            max_tokens=500
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"분석 실패: {e}"


def create_chart(symbol: str, position_side: str, entry_time_ms, exit_time_ms, order_id: str):
    global exchange

    print(f"📈 차트 생성 시도: {symbol}")
    try:
        if entry_time_ms:
            since_time = entry_time_ms - (15 * 60 * 1000 * 10)
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since_time)
        else:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=200)

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # 오른쪽 여백
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=15), periods=15, freq="15min")
        future_df = pd.DataFrame(index=future_dates, columns=df.columns)
        df_extended = pd.concat([df, future_df])

        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        file_name = f"Trade_{safe_symbol}_{order_id}.png"
        save_path = os.path.join(CHART_DIR, file_name)

        buy_marker = [float("nan")] * len(df)
        sell_marker = [float("nan")] * len(df)
        offset_ratio = 0.008

        # entry
        if entry_time_ms:
            entry_dt = pd.to_datetime(entry_time_ms, unit="ms")
            try:
                entry_idx = df.index.get_indexer([entry_dt], method="nearest")[0]
                if position_side == "LONG":
                    buy_marker[entry_idx] = df["low"].iloc[entry_idx] * (1 - offset_ratio)
                else:
                    sell_marker[entry_idx] = df["high"].iloc[entry_idx] * (1 + offset_ratio)
            except:
                pass

        # exit
        if exit_time_ms:
            exit_dt = pd.to_datetime(exit_time_ms, unit="ms")
            try:
                exit_idx = df.index.get_indexer([exit_dt], method="nearest")[0]
                if position_side == "LONG":
                    sell_marker[exit_idx] = df["high"].iloc[exit_idx] * (1 + offset_ratio)
                else:
                    buy_marker[exit_idx] = df["low"].iloc[exit_idx] * (1 - offset_ratio)
            except:
                pass

        pad_len = len(df_extended) - len(df)
        buy_marker_ext = buy_marker + [float("nan")] * pad_len
        sell_marker_ext = sell_marker + [float("nan")] * pad_len

        add_plots = [
            mpf.make_addplot(buy_marker_ext, type="scatter", markersize=200, marker="^", color="green"),
            mpf.make_addplot(sell_marker_ext, type="scatter", markersize=200, marker="v", color="red"),
        ]

        mc = mpf.make_marketcolors(up="red", down="blue", edge="inherit", wick="inherit", volume="in")
        s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style="yahoo", gridstyle="", facecolor="white")

        mpf.plot(
            df_extended,
            type="candle",
            volume=True,
            style=s,
            addplot=add_plots,
            title=symbol,
            savefig=save_path,
            figscale=1.5,
            tight_layout=True,
        )

        print(f"📸 차트 저장 완료: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ 차트 생성 실패: {e}")
        return None


def save_to_csv(row: dict):
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if (not file_exists) or os.stat(CSV_PATH).st_size == 0:
            w.writerow(["거래시간", "주문ID", "종목", "포지션", "레버리지", "진입수량", "진입가",
                        "청산가", "손익금", "손익률", "승패여부", "AI분석", "차트파일"])
        w.writerow([
            row["time"], row["order_id"], row["symbol"], row["side"], row["leverage"], row["qty"],
            row["entry_price"], row["exit_price"], row["pnl"], row["roi"], row["result"],
            row["ai_analysis"], row["chart_file"]
        ])


def get_leverage(symbol: str) -> int:
    global exchange
    try:
        positions = exchange.fetch_positions([symbol])
        if positions:
            return positions[0].get("leverage", 1) or 1
    except:
        return 1
    return 1


def mask_key(k: str) -> str:
    if not k:
        return "-"
    if len(k) <= 8:
        return k[0:2] + "****"
    return f"{k[:4]}****{k[-4:]}"


# =========================
# 감시 루프 (기존 main()을 스레드로)
# =========================
def monitor_loop():
    global monitor_running, last_status, last_order_id, exchange, recent_records

    with state_lock:
        last_status = "monitor_loop started"

    # 시작 시 최신 주문 id 저장(중복 방지)
    try:
        orders = exchange.fetch_closed_orders(limit=1)
        if orders:
            last_order_id = orders[0]["id"]
    except:
        last_order_id = None

    while True:
        with state_lock:
            if not monitor_running:
                last_status = "stopped"
                break

        try:
            orders = exchange.fetch_closed_orders(limit=1)
            if not orders:
                time.sleep(1)
                continue

            latest_order = orders[0]
            current_id = latest_order["id"]

            if current_id == last_order_id:
                time.sleep(1)
                continue

            symbol = latest_order["symbol"]
            order_side = latest_order["side"]

            # ⚠️ 기존 로직 유지 (MVP): 종료 주문 기준으로 LONG/SHORT 추정
            position_side = "LONG" if order_side.lower() == "sell" else "SHORT"

            with state_lock:
                last_status = f"new order detected: {symbol} {current_id}"

            time.sleep(2)

            leverage = get_leverage(symbol)

            trades = exchange.fetch_my_trades(symbol, limit=100)

            pnl = 0.0
            qty = float(latest_order.get("amount") or 0)
            exit_price = float(latest_order.get("price") or 0)
            entry_price = exit_price

            exit_time_ms = latest_order.get("timestamp")
            entry_time_ms = None

            if trades:
                closing_trade = next((t for t in reversed(trades) if t.get("order") == latest_order["id"]), None)
                if closing_trade:
                    info = closing_trade.get("info", {})
                    if "closedPnl" in info:
                        pnl = float(info["closedPnl"])
                    if "execPrice" in info:
                        exit_price = float(info["execPrice"])
                    if "execQty" in info:
                        qty = float(info["execQty"])
                    exit_time_ms = closing_trade.get("timestamp")

                entry_side = "buy" if position_side == "LONG" else "sell"
                opening_trade = next(
                    (t for t in reversed(trades) if t.get("timestamp", 0) < (exit_time_ms or 0) and t.get("side") == entry_side),
                    None
                )
                if opening_trade:
                    entry_price = float(opening_trade.get("price") or entry_price)
                    entry_time_ms = opening_trade.get("timestamp")

            if entry_time_ms is None and qty > 0:
                if position_side == "LONG":
                    entry_price = exit_price - (pnl / qty)
                else:
                    entry_price = exit_price + (pnl / qty)

            margin = (entry_price * qty) / float(leverage) if leverage else 0
            roi_val = (pnl / margin) * 100 if margin > 0 else 0
            roi = f"{roi_val:.2f}%"
            result_str = "WIN" if pnl > 0 else "LOSE"

            chart_path = create_chart(symbol, position_side, entry_time_ms, exit_time_ms, current_id)
            chart_file = os.path.basename(chart_path) if chart_path else ""

            ai_comment = "분석 생략"
            if chart_path and client is not None:
                ai_comment = analyze_chart_with_gpt(chart_path, symbol, position_side)

            row = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "order_id": current_id,
                "symbol": symbol,
                "side": position_side,
                "leverage": leverage,
                "qty": qty,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "roi": roi,
                "result": result_str,
                "ai_analysis": ai_comment,
                "chart_file": chart_file,
            }

            save_to_csv(row)

            # 최근 10건 캐시
            with state_lock:
                recent_records.insert(0, row)
                recent_records = recent_records[:10]
                last_order_id = current_id
                last_status = f"saved: {symbol} {result_str} {roi}"

        except Exception as e:
            with state_lock:
                last_status = f"error: {e}"
            time.sleep(1)


# =========================
# API 엔드포인트 (Firebase에서 호출)
# =========================
@app.route("/start", methods=["POST"])
def start():
    """
    Firebase 웹 폼에서 action="/start"로 POST하면 여기로 들어온다.
    bybit_api_key, bybit_secret_key를 받아 exchange를 만들고 감시 스레드 시작.
    """
    global exchange, monitor_running, monitor_thread, key_mask, last_status

    api_key = request.form.get("bybit_api_key") or (request.json.get("bybit_api_key") if request.is_json else None)
    secret_key = request.form.get("bybit_secret_key") or (request.json.get("bybit_secret_key") if request.is_json else None)

    if not api_key or not secret_key:
        return "키가 비었습니다.", 400

    with state_lock:
        if monitor_running:
            return "이미 실행 중입니다.", 200

        # exchange 생성
        exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": secret_key,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

        monitor_running = True
        key_mask = mask_key(api_key)
        last_status = "starting..."

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    return "모니터 시작", 200


@app.route("/stop", methods=["POST"])
def stop():
    global monitor_running, last_status
    with state_lock:
        monitor_running = False
        last_status = "stop requested"
    return "모니터 중지", 200


@app.route("/status", methods=["GET"])
def status():
    with state_lock:
        return jsonify({
            "running": monitor_running,
            "key_mask": key_mask,
            "status": last_status,
        })


@app.route("/recent", methods=["GET"])
def recent():
    with state_lock:
        return jsonify(recent_records)


@app.route("/charts/<path:filename>", methods=["GET"])
def charts(filename):
    # chart 폴더의 이미지를 브라우저로 서빙
    return send_from_directory(CHART_DIR, filename)


# =========================
# 실행
# =========================
if __name__ == "__main__":
    # Cloud Run 호환 포트
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
# app.py
import os
import time
import csv
import base64
import threading
from datetime import datetime

import ccxt
import pandas as pd
import mplfinance as mpf
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

# =========================
# 기본 설정
# =========================
TIMEFRAME = "15m"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "chart")
RECORD_DIR = os.path.join(BASE_DIR, "record")
CSV_PATH = os.path.join(RECORD_DIR, "trading_journal.csv")

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(RECORD_DIR, exist_ok=True)

# OpenAI 키는 서버 고정 (env로 받는 걸 추천하지만 MVP로 그대로 둠)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# 전역 상태 (MVP)
# =========================
app = Flask(__name__)

state_lock = threading.Lock()
monitor_thread = None
monitor_running = False

exchange = None
key_mask = "-"
last_status = "idle"
last_order_id = None

recent_records = []  # 최근 10건(웹 표시용)


# =========================
# 유틸 함수들 (원래 코드 기반)
# =========================
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_chart_with_gpt(image_path: str, symbol: str, side: str) -> str:
    if client is None:
        return "분석 생략(OpenAI 미설정)"

    print(f"🤖 AI가 {symbol} 차트를 기술적으로 분석 중...")
    try:
        b64 = encode_image(image_path)
        prompt_text = (
            f"이 차트는 {symbol}의 15분봉 차트다. "
            f"초록색 화살표(▲)는 Buy, 빨간색 화살표(▼)는 Sell 지점이다. "
            f"내 포지션은 {side}였다. "
            f"오직 기술적 분석 관점(캔들 패턴, 지지/저항, 추세선)에서 "
            f"진입과 청산 자리가 적절했는지 평가해줘. 핵심만 3줄 요약."
        )

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional Technical Analyst. Focus only on chart analysis."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]}
            ],
            max_tokens=500
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"분석 실패: {e}"


def create_chart(symbol: str, position_side: str, entry_time_ms, exit_time_ms, order_id: str):
    global exchange

    print(f"📈 차트 생성 시도: {symbol}")
    try:
        if entry_time_ms:
            since_time = entry_time_ms - (15 * 60 * 1000 * 10)
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since_time)
        else:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=200)

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # 오른쪽 여백
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=15), periods=15, freq="15min")
        future_df = pd.DataFrame(index=future_dates, columns=df.columns)
        df_extended = pd.concat([df, future_df])

        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        file_name = f"Trade_{safe_symbol}_{order_id}.png"
        save_path = os.path.join(CHART_DIR, file_name)

        buy_marker = [float("nan")] * len(df)
        sell_marker = [float("nan")] * len(df)
        offset_ratio = 0.008

        # entry
        if entry_time_ms:
            entry_dt = pd.to_datetime(entry_time_ms, unit="ms")
            try:
                entry_idx = df.index.get_indexer([entry_dt], method="nearest")[0]
                if position_side == "LONG":
                    buy_marker[entry_idx] = df["low"].iloc[entry_idx] * (1 - offset_ratio)
                else:
                    sell_marker[entry_idx] = df["high"].iloc[entry_idx] * (1 + offset_ratio)
            except:
                pass

        # exit
        if exit_time_ms:
            exit_dt = pd.to_datetime(exit_time_ms, unit="ms")
            try:
                exit_idx = df.index.get_indexer([exit_dt], method="nearest")[0]
                if position_side == "LONG":
                    sell_marker[exit_idx] = df["high"].iloc[exit_idx] * (1 + offset_ratio)
                else:
                    buy_marker[exit_idx] = df["low"].iloc[exit_idx] * (1 - offset_ratio)
            except:
                pass

        pad_len = len(df_extended) - len(df)
        buy_marker_ext = buy_marker + [float("nan")] * pad_len
        sell_marker_ext = sell_marker + [float("nan")] * pad_len

        add_plots = [
            mpf.make_addplot(buy_marker_ext, type="scatter", markersize=200, marker="^", color="green"),
            mpf.make_addplot(sell_marker_ext, type="scatter", markersize=200, marker="v", color="red"),
        ]

        mc = mpf.make_marketcolors(up="red", down="blue", edge="inherit", wick="inherit", volume="in")
        s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style="yahoo", gridstyle="", facecolor="white")

        mpf.plot(
            df_extended,
            type="candle",
            volume=True,
            style=s,
            addplot=add_plots,
            title=symbol,
            savefig=save_path,
            figscale=1.5,
            tight_layout=True,
        )

        print(f"📸 차트 저장 완료: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ 차트 생성 실패: {e}")
        return None


def save_to_csv(row: dict):
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if (not file_exists) or os.stat(CSV_PATH).st_size == 0:
            w.writerow(["거래시간", "주문ID", "종목", "포지션", "레버리지", "진입수량", "진입가",
                        "청산가", "손익금", "손익률", "승패여부", "AI분석", "차트파일"])
        w.writerow([
            row["time"], row["order_id"], row["symbol"], row["side"], row["leverage"], row["qty"],
            row["entry_price"], row["exit_price"], row["pnl"], row["roi"], row["result"],
            row["ai_analysis"], row["chart_file"]
        ])


def get_leverage(symbol: str) -> int:
    global exchange
    try:
        positions = exchange.fetch_positions([symbol])
        if positions:
            return positions[0].get("leverage", 1) or 1
    except:
        return 1
    return 1


def mask_key(k: str) -> str:
    if not k:
        return "-"
    if len(k) <= 8:
        return k[0:2] + "****"
    return f"{k[:4]}****{k[-4:]}"


# =========================
# 감시 루프 (기존 main()을 스레드로)
# =========================
def monitor_loop():
    global monitor_running, last_status, last_order_id, exchange, recent_records

    with state_lock:
        last_status = "monitor_loop started"

    # 시작 시 최신 주문 id 저장(중복 방지)
    try:
        orders = exchange.fetch_closed_orders(limit=1)
        if orders:
            last_order_id = orders[0]["id"]
    except:
        last_order_id = None

    while True:
        with state_lock:
            if not monitor_running:
                last_status = "stopped"
                break

        try:
            orders = exchange.fetch_closed_orders(limit=1)
            if not orders:
                time.sleep(1)
                continue

            latest_order = orders[0]
            current_id = latest_order["id"]

            if current_id == last_order_id:
                time.sleep(1)
                continue

            symbol = latest_order["symbol"]
            order_side = latest_order["side"]

            # ⚠️ 기존 로직 유지 (MVP): 종료 주문 기준으로 LONG/SHORT 추정
            position_side = "LONG" if order_side.lower() == "sell" else "SHORT"

            with state_lock:
                last_status = f"new order detected: {symbol} {current_id}"

            time.sleep(2)

            leverage = get_leverage(symbol)

            trades = exchange.fetch_my_trades(symbol, limit=100)

            pnl = 0.0
            qty = float(latest_order.get("amount") or 0)
            exit_price = float(latest_order.get("price") or 0)
            entry_price = exit_price

            exit_time_ms = latest_order.get("timestamp")
            entry_time_ms = None

            if trades:
                closing_trade = next((t for t in reversed(trades) if t.get("order") == latest_order["id"]), None)
                if closing_trade:
                    info = closing_trade.get("info", {})
                    if "closedPnl" in info:
                        pnl = float(info["closedPnl"])
                    if "execPrice" in info:
                        exit_price = float(info["execPrice"])
                    if "execQty" in info:
                        qty = float(info["execQty"])
                    exit_time_ms = closing_trade.get("timestamp")

                entry_side = "buy" if position_side == "LONG" else "sell"
                opening_trade = next(
                    (t for t in reversed(trades) if t.get("timestamp", 0) < (exit_time_ms or 0) and t.get("side") == entry_side),
                    None
                )
                if opening_trade:
                    entry_price = float(opening_trade.get("price") or entry_price)
                    entry_time_ms = opening_trade.get("timestamp")

            if entry_time_ms is None and qty > 0:
                if position_side == "LONG":
                    entry_price = exit_price - (pnl / qty)
                else:
                    entry_price = exit_price + (pnl / qty)

            margin = (entry_price * qty) / float(leverage) if leverage else 0
            roi_val = (pnl / margin) * 100 if margin > 0 else 0
            roi = f"{roi_val:.2f}%"
            result_str = "WIN" if pnl > 0 else "LOSE"

            chart_path = create_chart(symbol, position_side, entry_time_ms, exit_time_ms, current_id)
            chart_file = os.path.basename(chart_path) if chart_path else ""

            ai_comment = "분석 생략"
            if chart_path and client is not None:
                ai_comment = analyze_chart_with_gpt(chart_path, symbol, position_side)

            row = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "order_id": current_id,
                "symbol": symbol,
                "side": position_side,
                "leverage": leverage,
                "qty": qty,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "roi": roi,
                "result": result_str,
                "ai_analysis": ai_comment,
                "chart_file": chart_file,
            }

            save_to_csv(row)

            # 최근 10건 캐시
            with state_lock:
                recent_records.insert(0, row)
                recent_records = recent_records[:10]
                last_order_id = current_id
                last_status = f"saved: {symbol} {result_str} {roi}"

        except Exception as e:
            with state_lock:
                last_status = f"error: {e}"
            time.sleep(1)


# =========================
# API 엔드포인트 (Firebase에서 호출)
# =========================
@app.route("/start", methods=["POST"])
def start():
    """
    Firebase 웹 폼에서 action="/start"로 POST하면 여기로 들어온다.
    bybit_api_key, bybit_secret_key를 받아 exchange를 만들고 감시 스레드 시작.
    """
    global exchange, monitor_running, monitor_thread, key_mask, last_status

    api_key = request.form.get("bybit_api_key") or (request.json.get("bybit_api_key") if request.is_json else None)
    secret_key = request.form.get("bybit_secret_key") or (request.json.get("bybit_secret_key") if request.is_json else None)

    if not api_key or not secret_key:
        return "키가 비었습니다.", 400

    with state_lock:
        if monitor_running:
            return "이미 실행 중입니다.", 200

        # exchange 생성
        exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": secret_key,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

        monitor_running = True
        key_mask = mask_key(api_key)
        last_status = "starting..."

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    return "모니터 시작", 200


@app.route("/stop", methods=["POST"])
def stop():
    global monitor_running, last_status
    with state_lock:
        monitor_running = False
        last_status = "stop requested"
    return "모니터 중지", 200


@app.route("/status", methods=["GET"])
def status():
    with state_lock:
        return jsonify({
            "running": monitor_running,
            "key_mask": key_mask,
            "status": last_status,
        })


@app.route("/recent", methods=["GET"])
def recent():
    with state_lock:
        return jsonify(recent_records)


@app.route("/charts/<path:filename>", methods=["GET"])
def charts(filename):
    # chart 폴더의 이미지를 브라우저로 서빙
    return send_from_directory(CHART_DIR, filename)


# =========================
# 실행
# =========================
if __name__ == "__main__":
    # Cloud Run 호환 포트
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
# app.py
import os
import time
import csv
import base64
import threading
from datetime import datetime

import ccxt
import pandas as pd
import mplfinance as mpf
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

# =========================
# 기본 설정
# =========================
TIMEFRAME = "15m"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "chart")
RECORD_DIR = os.path.join(BASE_DIR, "record")
CSV_PATH = os.path.join(RECORD_DIR, "trading_journal.csv")

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(RECORD_DIR, exist_ok=True)

# OpenAI 키는 서버 고정 (env로 받는 걸 추천하지만 MVP로 그대로 둠)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# 전역 상태 (MVP)
# =========================
app = Flask(__name__)

state_lock = threading.Lock()
monitor_thread = None
monitor_running = False

exchange = None
key_mask = "-"
last_status = "idle"
last_order_id = None

recent_records = []  # 최근 10건(웹 표시용)


# =========================
# 유틸 함수들 (원래 코드 기반)
# =========================
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_chart_with_gpt(image_path: str, symbol: str, side: str) -> str:
    if client is None:
        return "분석 생략(OpenAI 미설정)"

    print(f"🤖 AI가 {symbol} 차트를 기술적으로 분석 중...")
    try:
        b64 = encode_image(image_path)
        prompt_text = (
            f"이 차트는 {symbol}의 15분봉 차트다. "
            f"초록색 화살표(▲)는 Buy, 빨간색 화살표(▼)는 Sell 지점이다. "
            f"내 포지션은 {side}였다. "
            f"오직 기술적 분석 관점(캔들 패턴, 지지/저항, 추세선)에서 "
            f"진입과 청산 자리가 적절했는지 평가해줘. 핵심만 3줄 요약."
        )

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional Technical Analyst. Focus only on chart analysis."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]}
            ],
            max_tokens=500
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"분석 실패: {e}"


def create_chart(symbol: str, position_side: str, entry_time_ms, exit_time_ms, order_id: str):
    global exchange

    print(f"📈 차트 생성 시도: {symbol}")
    try:
        if entry_time_ms:
            since_time = entry_time_ms - (15 * 60 * 1000 * 10)
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since_time)
        else:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=200)

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # 오른쪽 여백
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=15), periods=15, freq="15min")
        future_df = pd.DataFrame(index=future_dates, columns=df.columns)
        df_extended = pd.concat([df, future_df])

        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        file_name = f"Trade_{safe_symbol}_{order_id}.png"
        save_path = os.path.join(CHART_DIR, file_name)

        buy_marker = [float("nan")] * len(df)
        sell_marker = [float("nan")] * len(df)
        offset_ratio = 0.008

        # entry
        if entry_time_ms:
            entry_dt = pd.to_datetime(entry_time_ms, unit="ms")
            try:
                entry_idx = df.index.get_indexer([entry_dt], method="nearest")[0]
                if position_side == "LONG":
                    buy_marker[entry_idx] = df["low"].iloc[entry_idx] * (1 - offset_ratio)
                else:
                    sell_marker[entry_idx] = df["high"].iloc[entry_idx] * (1 + offset_ratio)
            except:
                pass

        # exit
        if exit_time_ms:
            exit_dt = pd.to_datetime(exit_time_ms, unit="ms")
            try:
                exit_idx = df.index.get_indexer([exit_dt], method="nearest")[0]
                if position_side == "LONG":
                    sell_marker[exit_idx] = df["high"].iloc[exit_idx] * (1 + offset_ratio)
                else:
                    buy_marker[exit_idx] = df["low"].iloc[exit_idx] * (1 - offset_ratio)
            except:
                pass

        pad_len = len(df_extended) - len(df)
        buy_marker_ext = buy_marker + [float("nan")] * pad_len
        sell_marker_ext = sell_marker + [float("nan")] * pad_len

        add_plots = [
            mpf.make_addplot(buy_marker_ext, type="scatter", markersize=200, marker="^", color="green"),
            mpf.make_addplot(sell_marker_ext, type="scatter", markersize=200, marker="v", color="red"),
        ]

        mc = mpf.make_marketcolors(up="red", down="blue", edge="inherit", wick="inherit", volume="in")
        s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style="yahoo", gridstyle="", facecolor="white")

        mpf.plot(
            df_extended,
            type="candle",
            volume=True,
            style=s,
            addplot=add_plots,
            title=symbol,
            savefig=save_path,
            figscale=1.5,
            tight_layout=True,
        )

        print(f"📸 차트 저장 완료: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ 차트 생성 실패: {e}")
        return None


def save_to_csv(row: dict):
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if (not file_exists) or os.stat(CSV_PATH).st_size == 0:
            w.writerow(["거래시간", "주문ID", "종목", "포지션", "레버리지", "진입수량", "진입가",
                        "청산가", "손익금", "손익률", "승패여부", "AI분석", "차트파일"])
        w.writerow([
            row["time"], row["order_id"], row["symbol"], row["side"], row["leverage"], row["qty"],
            row["entry_price"], row["exit_price"], row["pnl"], row["roi"], row["result"],
            row["ai_analysis"], row["chart_file"]
        ])


def get_leverage(symbol: str) -> int:
    global exchange
    try:
        positions = exchange.fetch_positions([symbol])
        if positions:
            return positions[0].get("leverage", 1) or 1
    except:
        return 1
    return 1


def mask_key(k: str) -> str:
    if not k:
        return "-"
    if len(k) <= 8:
        return k[0:2] + "****"
    return f"{k[:4]}****{k[-4:]}"


# =========================
# 감시 루프 (기존 main()을 스레드로)
# =========================
def monitor_loop():
    global monitor_running, last_status, last_order_id, exchange, recent_records

    with state_lock:
        last_status = "monitor_loop started"

    # 시작 시 최신 주문 id 저장(중복 방지)
    try:
        orders = exchange.fetch_closed_orders(limit=1)
        if orders:
            last_order_id = orders[0]["id"]
    except:
        last_order_id = None

    while True:
        with state_lock:
            if not monitor_running:
                last_status = "stopped"
                break

        try:
            orders = exchange.fetch_closed_orders(limit=1)
            if not orders:
                time.sleep(1)
                continue

            latest_order = orders[0]
            current_id = latest_order["id"]

            if current_id == last_order_id:
                time.sleep(1)
                continue

            symbol = latest_order["symbol"]
            order_side = latest_order["side"]

            # ⚠️ 기존 로직 유지 (MVP): 종료 주문 기준으로 LONG/SHORT 추정
            position_side = "LONG" if order_side.lower() == "sell" else "SHORT"

            with state_lock:
                last_status = f"new order detected: {symbol} {current_id}"

            time.sleep(2)

            leverage = get_leverage(symbol)

            trades = exchange.fetch_my_trades(symbol, limit=100)

            pnl = 0.0
            qty = float(latest_order.get("amount") or 0)
            exit_price = float(latest_order.get("price") or 0)
            entry_price = exit_price

            exit_time_ms = latest_order.get("timestamp")
            entry_time_ms = None

            if trades:
                closing_trade = next((t for t in reversed(trades) if t.get("order") == latest_order["id"]), None)
                if closing_trade:
                    info = closing_trade.get("info", {})
                    if "closedPnl" in info:
                        pnl = float(info["closedPnl"])
                    if "execPrice" in info:
                        exit_price = float(info["execPrice"])
                    if "execQty" in info:
                        qty = float(info["execQty"])
                    exit_time_ms = closing_trade.get("timestamp")

                entry_side = "buy" if position_side == "LONG" else "sell"
                opening_trade = next(
                    (t for t in reversed(trades) if t.get("timestamp", 0) < (exit_time_ms or 0) and t.get("side") == entry_side),
                    None
                )
                if opening_trade:
                    entry_price = float(opening_trade.get("price") or entry_price)
                    entry_time_ms = opening_trade.get("timestamp")

            if entry_time_ms is None and qty > 0:
                if position_side == "LONG":
                    entry_price = exit_price - (pnl / qty)
                else:
                    entry_price = exit_price + (pnl / qty)

            margin = (entry_price * qty) / float(leverage) if leverage else 0
            roi_val = (pnl / margin) * 100 if margin > 0 else 0
            roi = f"{roi_val:.2f}%"
            result_str = "WIN" if pnl > 0 else "LOSE"

            chart_path = create_chart(symbol, position_side, entry_time_ms, exit_time_ms, current_id)
            chart_file = os.path.basename(chart_path) if chart_path else ""

            ai_comment = "분석 생략"
            if chart_path and client is not None:
                ai_comment = analyze_chart_with_gpt(chart_path, symbol, position_side)

            row = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "order_id": current_id,
                "symbol": symbol,
                "side": position_side,
                "leverage": leverage,
                "qty": qty,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "roi": roi,
                "result": result_str,
                "ai_analysis": ai_comment,
                "chart_file": chart_file,
            }

            save_to_csv(row)

            # 최근 10건 캐시
            with state_lock:
                recent_records.insert(0, row)
                recent_records = recent_records[:10]
                last_order_id = current_id
                last_status = f"saved: {symbol} {result_str} {roi}"

        except Exception as e:
            with state_lock:
                last_status = f"error: {e}"
            time.sleep(1)


# =========================
# API 엔드포인트 (Firebase에서 호출)
# =========================
@app.route("/start", methods=["POST"])
def start():
    """
    Firebase 웹 폼에서 action="/start"로 POST하면 여기로 들어온다.
    bybit_api_key, bybit_secret_key를 받아 exchange를 만들고 감시 스레드 시작.
    """
    global exchange, monitor_running, monitor_thread, key_mask, last_status

    api_key = request.form.get("bybit_api_key") or (request.json.get("bybit_api_key") if request.is_json else None)
    secret_key = request.form.get("bybit_secret_key") or (request.json.get("bybit_secret_key") if request.is_json else None)

    if not api_key or not secret_key:
        return "키가 비었습니다.", 400

    with state_lock:
        if monitor_running:
            return "이미 실행 중입니다.", 200

        # exchange 생성
        exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": secret_key,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

        monitor_running = True
        key_mask = mask_key(api_key)
        last_status = "starting..."

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    return "모니터 시작", 200


@app.route("/stop", methods=["POST"])
def stop():
    global monitor_running, last_status
    with state_lock:
        monitor_running = False
        last_status = "stop requested"
    return "모니터 중지", 200


@app.route("/status", methods=["GET"])
def status():
    with state_lock:
        return jsonify({
            "running": monitor_running,
            "key_mask": key_mask,
            "status": last_status,
        })


@app.route("/recent", methods=["GET"])
def recent():
    with state_lock:
        return jsonify(recent_records)


@app.route("/charts/<path:filename>", methods=["GET"])
def charts(filename):
    # chart 폴더의 이미지를 브라우저로 서빙
    return send_from_directory(CHART_DIR, filename)


# =========================
# 실행
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

