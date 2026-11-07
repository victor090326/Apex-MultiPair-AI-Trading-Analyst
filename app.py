import base64
import io
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# --------------------------- App configuration ---------------------------
st.set_page_config(
    page_title="Apex Multi-Pair AI Trading Analyst",
    page_icon="📈",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "prediction_history.db"
DEFAULT_MODEL = "gemini-3.8-flash"  # Stable model listed in Google's Gemini API docs.

PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "NZD/USD", "USD/CAD",
    "EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/AUD", "EUR/CAD", "EUR/NZD",
    "GBP/JPY", "GBP/CHF", "GBP/AUD", "GBP/CAD", "GBP/NZD",
    "AUD/JPY", "AUD/NZD", "AUD/CAD", "AUD/CHF", "NZD/JPY", "NZD/CAD",
    "CAD/JPY", "CHF/JPY", "XAU/USD (Gold)", "BTC/USD", "US30", "NAS100",
]
TIMEFRAMES = [
    "Auto-detect", "1-minute", "2-minute", "3-minute", "5-minute", "10-minute",
    "15-minute", "30-minute", "1-hour", "4-hour", "Daily", "Weekly",
]
HORIZONS = [
    "1 minute", "2 minutes", "3 minutes", "5 minutes", "8 minutes", "10 minutes",
    "15 minutes", "30 minutes", "1 hour", "2 hours", "4 hours", "1 day", "Custom",
]
HORIZON_MINUTES = {
    "1 minute": 1, "2 minutes": 2, "3 minutes": 3, "5 minutes": 5,
    "8 minutes": 8, "10 minutes": 10, "15 minutes": 15, "30 minutes": 30,
    "1 hour": 60, "2 hours": 120, "4 hours": 240, "1 day": 1440,
}

st.title("📈 Apex Multi-Pair AI Trading Analyst")
st.caption("Free-first • Multi-pair chart review • Flexible timeframes • Quick trades • Multi-timeframe evidence")

st.info(
    "Decision-support only. No AI can guarantee a profitable trade or predict markets perfectly. "
    "The app analyses only the screenshots and candle data you provide. It does not execute trades "
    "or independently monitor live prices."
)

# --------------------------- History database ---------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                horizon TEXT NOT NULL,
                mode TEXT NOT NULL,
                decision TEXT NOT NULL,
                confidence TEXT NOT NULL,
                report TEXT NOT NULL,
                outcome TEXT DEFAULT 'Not verified'
            )
        """)
        con.commit()


def save_prediction(pair, timeframe, horizon, mode, decision, confidence, report):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """INSERT INTO predictions
            (created_at, pair, timeframe, horizon, mode, decision, confidence, report)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), pair, timeframe, horizon,
             mode, decision, confidence, report),
        )
        con.commit()


def get_history():
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(
            "SELECT id, created_at, pair, timeframe, horizon, mode, decision, confidence, outcome "
            "FROM predictions ORDER BY id DESC LIMIT 200", con
        )


def update_outcome(record_id, outcome):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("UPDATE predictions SET outcome=? WHERE id=?", (outcome, int(record_id)))
        con.commit()


init_db()

# --------------------------- Image handling ---------------------------
def image_to_part(uploaded_file):
    raw = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image.thumbnail((2200, 2200))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()


# --------------------------- AI analysis ---------------------------
SYSTEM_GUIDE = """
You are a cautious, evidence-led forex and multi-instrument chart analyst.
You are NOT a broker, price feed, or fortune teller. Analyse only the uploaded chart
images and supplied structured candle data. Never invent a price, indicator reading,
news event, volume, timeframe, support/resistance level, or candle that is not visible
or supplied. If something cannot be read, explicitly say "not readable".

For every instrument, distinguish:
- chart timeframe (the duration of each candle),
- prediction/outlook horizon (how far ahead the user asks you to assess),
- higher-timeframe context and lower-timeframe execution evidence.

If several screenshots for one pair are supplied, use their labels to compare timeframes.
If the timeframe is set to Auto-detect, infer it only if a readable chart label supports it;
otherwise state "not confidently identifiable". Never claim multi-timeframe confirmation
from one screenshot. Treat broker OTC feeds as broker-specific and not interchangeable
with regular forex prices.

Use price structure (HH/HL, LH/LL, ranges, breaks), candle behaviour, visible EMA/RSI/MACD/
Bollinger Bands, momentum, and visible support/resistance only when actually readable.
Do not claim that an indicator exists if it is not visible. Do not invent live market news.
If no trusted news data is supplied, say news was not checked.

A screenshot is a static view and may contain a forming candle. Be cautious with quick
trades and short horizons. If evidence is insufficient, screenshots conflict, the chart
is blurry/cropped, or there is no clear setup, choose NO TRADE.

For each instrument return:
1. Screenshot/data quality
2. Instrument and feed identification
3. Timeframe(s) identified and confidence in identification
4. Higher-timeframe direction (if evidence supplied)
5. Immediate/entry-timeframe direction (if evidence supplied)
6. Market structure and momentum
7. Visible indicators and readings (or not readable/not supplied)
8. Key zones (only if visible and defensible)
9. Bullish scenario and confirmation needed
10. Bearish scenario and confirmation needed
11. Invalidation condition
12. FINAL DECISION: exactly BUY, SELL, or NO TRADE
13. Confidence: LOW, MODERATE, or HIGH, with reasons

Do not manufacture a numerical probability or imply that a confidence label is a
statistically calibrated win rate. If BUY or SELL lacks clear confirmation, choose NO TRADE.
Rank multiple instruments only by the evidence visible in the submitted charts.
"""


def call_gemini(api_key, model, prompt, image_items):
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Gemini SDK is missing. Install the requirements.txt packages.") from exc

    client = genai.Client(api_key=api_key)
    contents = [types.Part.from_text(text=prompt)]
    for label, image_bytes in image_items:
        contents.append(types.Part.from_text(text=f"Screenshot label: {label}"))
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=5000,
        ),
    )
    result = getattr(response, "text", None)
    if not result:
        raise RuntimeError("Gemini returned an empty response. Try a clearer/smaller upload.")
    return result


def friendly_api_error(exc):
    message = str(exc)
    lower = message.lower()
    if "404" in lower or "not found" in lower:
        return (
            "Gemini could not find or access the selected model. Confirm that your API key "
            "is from Google AI Studio and that the model is available to your project. "
            "The app uses the documented stable default model; check Google's model list if access changes."
        )
    if "429" in lower or "resource_exhausted" in lower or "quota" in lower:
        return "Gemini free-tier quota or rate limit reached. Wait and retry later; do not keep pressing Analyse."
    if "401" in lower or "403" in lower or "api key" in lower or "permission" in lower:
        return "Gemini rejected the API key or permissions. Check the key in Google AI Studio. Never put the key in GitHub."
    if "timeout" in lower or "deadline" in lower:
        return "The request timed out. Try fewer screenshots or smaller images, then retry."
    return f"Analysis could not be completed: {type(exc).__name__}: {message[:500]}"


# --------------------------- Sidebar controls ---------------------------
with st.sidebar:
    st.header("Analysis settings")
    mode = st.radio("Analysis mode", ["Quick Trade", "Multi-Timeframe", "Multi-Pair Scanner"])
    feed_type = st.selectbox("Market/feed type", ["Regular Forex", "Broker OTC", "CFD", "Crypto", "Other / unsure"])
    default_pair = st.selectbox("Primary instrument", PAIRS, index=0)
    main_tf = st.selectbox("Main chart timeframe", TIMEFRAMES, index=0)
    horizon_choice = st.selectbox("Prediction / outlook duration", HORIZONS, index=5)
    if horizon_choice == "Custom":
        custom_value = st.number_input("Custom duration (minutes)", min_value=1, max_value=10080, value=12, step=1)
        horizon_text = f"{int(custom_value)} minutes"
    else:
        horizon_text = horizon_choice
    st.caption("Candle timeframe and prediction duration are separate settings.")
    model_name = st.text_input("Gemini model (advanced; leave unchanged)", value=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    st.caption("Default is a stable model name from Google's current model documentation.")
    api_key = st.text_input(
        "Gemini API key",
        value="",
        type="password",
        help="Get a key from Google AI Studio. It is used only for this session unless you configure Streamlit secrets.",
    )

# Optional secrets/env key support
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""
api_key = api_key or os.getenv("GEMINI_API_KEY", "")

tab_analysis, tab_data, tab_history, tab_help = st.tabs(
    ["🧠 AI Analysis", "📊 Candle Data", "🗂️ Prediction History", "ℹ️ Help"]
)

# --------------------------- AI analysis interface ---------------------------
with tab_analysis:
    st.subheader("Upload chart screenshots")
    st.write(
        "Choose one pair for detailed analysis, or select several pairs in Multi-Pair Scanner. "
        "For multi-timeframe analysis, label each screenshot with its actual timeframe."
    )

    if mode == "Multi-Pair Scanner":
        selected_pairs = st.multiselect(
            "Pairs/instruments to scan",
            PAIRS,
            default=[default_pair, "GBP/USD"] if default_pair != "GBP/USD" else ["GBP/USD", "EUR/USD"],
            help="Upload at least one chart screenshot for each selected pair. This is screenshot-based scanning, not a live market feed.",
        )
        image_items = []
        for pair in selected_pairs:
            with st.expander(f"📷 {pair} chart", expanded=True):
                pair_tf = st.selectbox(f"{pair} timeframe", TIMEFRAMES, key=f"tf_{pair}")
                pair_files = st.file_uploader(
                    f"Upload one or more {pair} screenshots",
                    type=["png", "jpg", "jpeg", "webp"],
                    accept_multiple_files=True,
                    key=f"files_{pair}",
                )
                for index, file in enumerate(pair_files or []):
                    image_items.append((f"{pair} | {pair_tf} | screenshot {index + 1} | feed: {feed_type}", image_to_part(file)))
        scanner_context = (
            f"MODE: Multi-Pair Scanner. Feed type: {feed_type}. Outlook horizon: {horizon_text}. "
            "Analyse each submitted instrument separately, then rank only those with adequate evidence. "
            "Do not infer missing charts for selected pairs. Explicitly list selected pairs with no upload as NOT ANALYSED."
        )
        selected_label = ", ".join(selected_pairs) if selected_pairs else "No pairs selected"
        if st.button("🔎 Scan uploaded pairs", type="primary", use_container_width=True):
            if not selected_pairs:
                st.error("Select at least one pair.")
            elif not image_items:
                st.error("Upload at least one chart screenshot.")
            elif not api_key:
                st.error("Enter your Gemini API key in the sidebar.")
            else:
                with st.spinner("Gemini is analysing the submitted pair charts..."):
                    try:
                        report = call_gemini(api_key, model_name.strip(), SYSTEM_GUIDE + "\n\n" + scanner_context, image_items)
                        st.session_state["latest_report"] = report
                        st.session_state["latest_meta"] = (selected_label, main_tf, horizon_text, mode)
                        st.success("Scan completed. Only uploaded charts were analysed.")
                    except Exception as exc:
                        st.error(friendly_api_error(exc))
    else:
        pair = default_pair
        st.markdown(f"**Selected instrument:** {pair} · **Feed:** {feed_type}")
        slots = st.number_input("Number of timeframe screenshots", min_value=1, max_value=8, value=3 if mode == "Multi-Timeframe" else 1, step=1)
        image_items = []
        for i in range(int(slots)):
            col1, col2 = st.columns([1, 2])
            with col1:
                label_tf = st.selectbox(
                    f"Screenshot {i + 1} timeframe",
                    TIMEFRAMES,
                    index=0 if i == 0 and main_tf == "Auto-detect" else (TIMEFRAMES.index(main_tf) if main_tf in TIMEFRAMES else 0),
                    key=f"single_tf_{i}",
                )
            with col2:
                uploaded = st.file_uploader(
                    f"Screenshot {i + 1} ({pair})",
                    type=["png", "jpg", "jpeg", "webp"],
                    key=f"single_img_{i}",
                )
            if uploaded is not None:
                image_items.append((f"{pair} | {label_tf} | feed: {feed_type}", image_to_part(uploaded)))

        extra_context = st.text_area(
            "Optional notes",
            placeholder="Example: broker/platform, whether the latest candle is closed, visible indicators, session context.",
            height=85,
        )
        button_label = "⚡ Analyse quick trade" if mode == "Quick Trade" else "🧭 Analyse multiple timeframes"
        if st.button(button_label, type="primary", use_container_width=True):
            if not image_items:
                st.error("Upload at least one chart screenshot.")
            elif not api_key:
                st.error("Enter your Gemini API key in the sidebar.")
            else:
                labels = "\n".join(item[0] for item in image_items)
                prompt = (
                    f"MODE: {mode}. Instrument: {pair}. Feed: {feed_type}. "
                    f"Main timeframe setting: {main_tf}. Prediction horizon: {horizon_text}. "
                    f"Uploaded screenshot labels:\n{labels}\n"
                    f"User notes: {extra_context}\n\n"
                    "Analyse this instrument. Keep the selected horizon distinct from candle timeframe. "
                    "For Quick Trade mode, prioritise immediate structure, momentum and entry confirmation. "
                    "For Multi-Timeframe mode, explain whether higher and lower timeframes align or conflict."
                )
                with st.spinner("Gemini is reviewing your chart evidence..."):
                    try:
                        report = call_gemini(api_key, model_name.strip(), SYSTEM_GUIDE + "\n\n" + prompt, image_items)
                        st.session_state["latest_report"] = report
                        st.session_state["latest_meta"] = (pair, main_tf, horizon_text, mode)
                        st.success("Analysis completed. Review confirmation and invalidation conditions.")
                    except Exception as exc:
                        st.error(friendly_api_error(exc))

    if "latest_report" in st.session_state:
        st.divider()
        st.subheader("Latest AI report")
        st.markdown(st.session_state["latest_report"])
        meta = st.session_state.get("latest_meta", ("Unknown", main_tf, horizon_text, mode))
        st.caption(f"Instrument(s): {meta[0]} · Main timeframe setting: {meta[1]} · Horizon: {meta[2]} · Mode: {meta[3]}")
        st.download_button(
            "Download analysis report",
            data=st.session_state["latest_report"],
            file_name="apex_trading_analysis.txt",
            mime="text/plain",
        )
        c1, c2 = st.columns(2)
        with c1:
            decision = st.selectbox("Record final decision", ["NO TRADE", "BUY", "SELL"], key="record_decision")
        with c2:
            confidence = st.selectbox("Confidence label", ["LOW", "MODERATE", "HIGH"], key="record_confidence")
        if st.button("Save this prediction to history"):
            save_prediction(str(meta[0]), str(meta[1]), str(meta[2]), str(meta[3]), decision, confidence, st.session_state["latest_report"])
            st.success("Prediction saved to local app history.")

# --------------------------- Candle data interface ---------------------------
with tab_data:
    st.subheader("Calculate indicators from OHLC candle data")
    st.write("Upload a CSV exported from your broker/data provider. Required columns: open, high, low, close. A timestamp column is recommended.")
    csv_file = st.file_uploader("OHLC CSV file", type=["csv"], key="ohlc_upload")
    csv_text = st.text_area("Or paste CSV candle data", height=120, placeholder="timestamp,open,high,low,close")
    if st.button("Calculate technical readings"):
        try:
            if csv_file is not None:
                df = pd.read_csv(csv_file)
            elif csv_text.strip():
                df = pd.read_csv(io.StringIO(csv_text))
            else:
                raise ValueError("Upload a CSV or paste candle data.")
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            aliases = {"date": "timestamp", "datetime": "timestamp", "time": "timestamp"}
            df = df.rename(columns=aliases)
            required = {"open", "high", "low", "close"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
            for col in required:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=list(required)).copy()
            valid = (
                (df["open"] > 0)
                & (df["high"] >= df[["open", "close", "low"]].max(axis=1))
                & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
            )
            df = df.loc[valid].reset_index(drop=True)
            if len(df) < 60:
                raise ValueError(f"Only {len(df)} valid candles. At least 60 are required.")
            close = df["close"]
            df["EMA20"] = close.ewm(span=20, adjust=False).mean()
            df["EMA50"] = close.ewm(span=50, adjust=False).mean()
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
            rs = gain / loss.replace(0, pd.NA)
            df["RSI14"] = 100 - (100 / (1 + rs))
            df["RSI14"] = df["RSI14"].fillna(100).where(loss.ne(0), 50)
            df["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
            df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            last = df.iloc[-1]
            summary = {
                "candles": int(len(df)),
                "latest_close": float(last["close"]),
                "EMA20": float(last["EMA20"]),
                "EMA50": float(last["EMA50"]),
                "RSI14": float(last["RSI14"]),
                "MACD": float(last["MACD"]),
                "MACD_signal": float(last["MACD_signal"]),
                "EMA_trend": "BULLISH" if last["EMA20"] > last["EMA50"] else "BEARISH" if last["EMA20"] < last["EMA50"] else "NEUTRAL",
                "recent_20_high": float(df.tail(20)["high"].max()),
                "recent_20_low": float(df.tail(20)["low"].min()),
            }
            st.session_state["ohlc_summary"] = summary
            st.session_state["ohlc_summary_pair"] = default_pair
            st.success("Indicator calculations completed from the uploaded candle data.")
        except Exception as exc:
            st.error(f"Could not calculate indicators: {exc}")

    if "ohlc_summary" in st.session_state:
        st.json(st.session_state["ohlc_summary"])
        st.caption(f"Instrument selected when calculated: {st.session_state.get('ohlc_summary_pair', default_pair)}")

# --------------------------- Prediction history interface ---------------------------
with tab_history:
    st.subheader("Prediction history")
    history = get_history()
    if history.empty:
        st.info("No predictions saved yet. Complete an analysis and use 'Save this prediction to history'.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.markdown("#### Verify an outcome")
        record_id = st.selectbox("Prediction record ID", history["id"].tolist())
        outcome = st.selectbox("Observed outcome", ["Correct", "Incorrect", "Unclear / expired"])
        if st.button("Save outcome"):
            update_outcome(record_id, outcome)
            st.success("Outcome updated. Accuracy is calculated only from records you manually verify.")
        verified = history[history["outcome"].isin(["Correct", "Incorrect"])]
        if not verified.empty:
            accuracy = (verified["outcome"] == "Correct").mean() * 100
            st.metric("Verified historical accuracy", f"{accuracy:.1f}%", help="Based only on manually verified records marked Correct or Incorrect.")
        st.caption("This history is stored in the app's local SQLite database. Some free hosting services reset local files when the app restarts; use persistent storage for long-term records.")

# --------------------------- Help ---------------------------
with tab_help:
    st.subheader("How to use Apex")
    st.markdown(
        """
- **Quick Trade:** Upload the most relevant short-term chart and choose the outlook duration independently.
- **Multi-Timeframe:** Upload several screenshots of the same instrument and label each screenshot with its real timeframe.
- **Multi-Pair Scanner:** Select several instruments and upload screenshots for each. The AI ranks only the charts actually supplied.
- **Candle Data:** Upload broker-exported OHLC CSV data to calculate EMA, RSI and MACD readings from actual values.
- **Prediction History:** Save decisions and manually record outcomes to calculate verified historical accuracy.

**Important limitations**
- This is not a live-price scanner. The multi-pair scanner reviews only screenshots you upload.
- Automatic timeframe detection can be uncertain. Confirm the chart label manually when possible.
- Screenshot analysis can misread small labels and chart details. CSV OHLC data is better for exact calculations.
- A confidence label is not a calibrated probability or guaranteed win rate.
- No orders are placed. Never expose your Gemini API key in source code, GitHub, screenshots or public chat.
"""
    )

st.divider()
st.caption("Apex Multi-Pair AI Trading Analyst • Gemini-powered chart review • No broker connection • No automatic execution")
