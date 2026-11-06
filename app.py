import base64
import io
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(page_title="Multi-Pair AI Trading Analyst", page_icon="📈", layout="wide")

st.title("Multi-Pair AI Trading Analyst")
st.caption("Multi-currency chart review • screenshot vision • optional candle-data analysis")
st.warning(
    "Decision-support only. No AI can reliably know what the next candle will do. "
    "A 6–10 minute outlook is a conditional scenario, not a guaranteed prediction or a trade instruction. "
    "Use NO TRADE when evidence is unclear, conflicting, or incomplete."
)

# ------------------------- Shared helpers -------------------------
REQUIRED = {"open", "high", "low", "close"}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    aliases = {"date": "timestamp", "datetime": "timestamp", "time": "timestamp"}
    return frame.rename(columns=aliases)


def load_ohlc(uploaded_file, pasted_text: str) -> pd.DataFrame:
    if uploaded_file is not None:
        frame = pd.read_csv(uploaded_file)
    elif pasted_text.strip():
        frame = pd.read_csv(io.StringIO(pasted_text.strip()))
    else:
        raise ValueError("Upload a CSV or paste candle data first.")

    frame = normalize_columns(frame)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))

    for col in REQUIRED:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")

    frame = frame.dropna(subset=list(REQUIRED))
    valid = (
        (frame["open"] > 0)
        & (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
    )
    frame = frame.loc[valid].reset_index(drop=True)
    if len(frame) < 60:
        raise ValueError(f"Only {len(frame)} valid candles. At least 60 are needed for the indicator summary.")
    return frame


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    delta = d["close"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi14"] = 100 - (100 / (1 + rs))
    d.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100
    d.loc[(avg_loss == 0) & (avg_gain == 0), "rsi14"] = 50
    prev = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["swing_high"] = False
    d["swing_low"] = False
    for i in range(2, len(d) - 2):
        window_h = d.loc[i - 2:i + 2, "high"].to_numpy()
        window_l = d.loc[i - 2:i + 2, "low"].to_numpy()
        d.loc[i, "swing_high"] = window_h[2] > np.max(np.delete(window_h, 2))
        d.loc[i, "swing_low"] = window_l[2] < np.min(np.delete(window_l, 2))
    d["last_swing_high"] = d["high"].where(d["swing_high"]).ffill().shift(1)
    d["last_swing_low"] = d["low"].where(d["swing_low"]).ffill().shift(1)
    d["bos"] = "NO BOS"
    bull = (d["close"] > d["last_swing_high"]) & (d["close"].shift(1) <= d["last_swing_high"])
    bear = (d["close"] < d["last_swing_low"]) & (d["close"].shift(1) >= d["last_swing_low"])
    d.loc[bull, "bos"] = "BULLISH BOS"
    d.loc[bear, "bos"] = "BEARISH BOS"
    return d


def data_summary(d: pd.DataFrame, candle_minutes: int) -> dict:
    last = d.iloc[-1]
    if pd.isna(last["rsi14"]) or pd.isna(last["atr14"]):
        raise ValueError("Indicator values are not ready. Upload more candles.")
    recent = d.tail(20)
    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())
    direction = "BULLISH" if last["ema20"] > last["ema50"] else "BEARISH" if last["ema20"] < last["ema50"] else "NEUTRAL"
    momentum = "POSITIVE" if last["rsi14"] > 55 else "NEGATIVE" if last["rsi14"] < 45 else "NEUTRAL"
    return {
        "candle_timeframe_minutes": candle_minutes,
        "latest_close": float(last["close"]),
        "ema20": float(last["ema20"]),
        "ema50": float(last["ema50"]),
        "rsi14": float(last["rsi14"]),
        "atr14": float(last["atr14"]),
        "ema_trend": direction,
        "momentum": momentum,
        "latest_bos": str(last["bos"]),
        "recent_20_candle_high": recent_high,
        "recent_20_candle_low": recent_low,
        "candles_supplied": int(len(d)),
        "last_timestamp": str(last["timestamp"]) if "timestamp" in d.columns else "not supplied",
    }


def image_to_data_url(uploaded) -> str:
    raw = uploaded.getvalue()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    # Keep image readable while avoiding unnecessarily huge API payloads.
    image.thumbnail((2400, 2400))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/jpeg;base64," + encoded


ANALYSIS_INSTRUCTIONS = """
You are a cautious multi-instrument chart-reading assistant. Analyse only the instrument identified by the user. This is visual chart analysis, not a price oracle.
The main chart timeframe and conditional outlook window are supplied in USER CONTEXT. The user wants a conditional outlook over the next
the user-selected number of minutes. Do not pretend you can know future candles.

Read only what is visibly supported. Never invent exact prices, indicator values, timeframe labels,
news, volume, broker data, or unseen candles. If a label or indicator value is too small/unclear, say unreadable.
If only one timeframe is supplied, do not claim multi-timeframe confirmation. Treat OTC quotes as
broker-specific and do not assume they match another EUR/USD feed.

Return these headings:
1. SCREENSHOT QUALITY — readable or unclear; note cropped/hidden chart details.
2. MARKET STRUCTURE — visible swing highs/lows, HH/HL or LH/LL only if identifiable; range/trend.
3. INDICATORS — names and readings only when legible; otherwise state not readable.
4. KEY ZONES — possible support/resistance and BOS only if visually identifiable; no invented levels.
5. OUTLOOK SCENARIOS — bullish continuation, bearish continuation, and/or range scenario; explain what must happen.
6. DECISION — choose exactly one: BUY BIAS, SELL BIAS, or NO TRADE. This is a cautious bias, not an instruction.
7. CONFIRMATION — observable price action that would support the bias.
8. INVALIDATION — observable price action that would cancel the bias.
9. CONFIDENCE — LOW, MODERATE, or HIGH, with a short reason. Never use certainty language.

Choose NO TRADE if the screenshot is unclear, conflicting, lacks enough candles, appears to show a forming candle
without context, or does not provide a clear setup. Explain that a single still screenshot cannot validate
what will happen over the next 6–10 minutes. Do not present this as financial advice.
"""


def call_openai(image_data_urls, context: str, api_key: str, model: str, structured_summary=None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    user_content = [{"type": "input_text", "text": ANALYSIS_INSTRUCTIONS + "\n\nUSER CONTEXT:\n" + context}]
    if structured_summary:
        user_content.append({"type": "input_text", "text": "Verified calculations from uploaded OHLC data (use these exact values; do not alter):\n" + json.dumps(structured_summary, indent=2)})
    for label, url in image_data_urls:
        user_content.append({"type": "input_text", "text": f"Chart image: {label}"})
        user_content.append({"type": "input_image", "image_url": url, "detail": "high"})
    response = client.responses.create(model=model, input=[{"role": "user", "content": user_content}])
    return response.output_text or "The model returned no text."


def call_gemini(image_data_urls, context: str, api_key: str, model: str, structured_summary=None) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    parts = [types.Part.from_text(text=ANALYSIS_INSTRUCTIONS + "\n\nUSER CONTEXT:\n" + context)]
    if structured_summary:
        parts.append(types.Part.from_text(text="Verified OHLC calculations:\n" + json.dumps(structured_summary, indent=2)))
    for label, url in image_data_urls:
        raw = base64.b64decode(url.split(",", 1)[1])
        parts.append(types.Part.from_text(text=f"Chart image: {label}"))
        parts.append(types.Part.from_bytes(data=raw, mime_type="image/jpeg"))
    response = client.models.generate_content(model=model, contents=parts)
    return response.text or "The model returned no text."


# ------------------------- Sidebar -------------------------
with st.sidebar:
    st.header("Analysis setup")
    common_pairs = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "NZD/USD", "USD/CAD",
        "EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/AUD", "EUR/CAD", "EUR/NZD",
        "GBP/JPY", "GBP/CHF", "GBP/AUD", "GBP/CAD", "GBP/NZD",
        "AUD/JPY", "AUD/NZD", "AUD/CAD", "AUD/CHF",
        "NZD/JPY", "NZD/CAD", "CAD/JPY", "CHF/JPY",
        "XAU/USD (Gold)", "BTC/USD", "US30", "NAS100", "Custom instrument"
    ]
    instrument_choice = st.selectbox("Currency pair / instrument", common_pairs, index=0)
    if instrument_choice == "Custom instrument":
        instrument = st.text_input("Enter instrument symbol", value="GBP/USD").strip().upper()
    else:
        instrument = instrument_choice
    market_type = st.selectbox("Market/feed type", ["Regular Forex", "Broker OTC", "CFD", "Other / not sure"])
    candle_minutes = st.selectbox("Main chart timeframe", [1, 2, 5, 15, 30, 60], index=1, format_func=lambda x: f"{x}-minute candles")
    horizon = st.selectbox("Outlook window", [2, 4, 6, 8, 10, 15, 30], index=3, format_func=lambda x: f"Next {x} minutes")
    st.caption("Choose the timeframe and outlook that match the uploaded chart. Longer outlooks are not guaranteed predictions.")
    provider = st.selectbox("AI analysis provider", ["OpenAI (ChatGPT models)", "Google Gemini"])
    if provider.startswith("OpenAI"):
        model_name = st.text_input("OpenAI vision model", value="gpt-5.4-mini", help="Must be a vision-capable model enabled for your API account.")
    else:
        model_name = st.text_input("Gemini vision model", value="gemini-2.5-flash")
    st.caption("API usage is billed by the provider separately from a ChatGPT subscription.")

# ------------------------- Main interface -------------------------
tab_screen, tab_data, tab_guide = st.tabs(["🖼️ Screenshot analysis", "📊 Candle-data analysis", "ℹ️ How to use"])

with tab_screen:
    st.subheader("Upload your broker chart screenshot")
    st.write(f"Upload a clear chart screenshot for **{instrument}** ({market_type}). Keep the candles, price scale, timeframe label, and indicators visible where possible.")
    main_chart = st.file_uploader(f"Main {instrument} chart screenshot (required)", type=["png", "jpg", "jpeg", "webp"], key="main_chart")
    c1, c2 = st.columns(2)
    with c1:
        higher_chart_5m = st.file_uploader("Optional 5-minute context screenshot", type=["png", "jpg", "jpeg", "webp"], key="higher5")
    with c2:
        higher_chart_15m = st.file_uploader("Optional 15-minute context screenshot", type=["png", "jpg", "jpeg", "webp"], key="higher15")

    context = st.text_area(
        "Extra context (optional)",
        placeholder="Example: Broker name; instrument; candle timeframe; screenshot taken just after a candle closed; visible indicators.",
        height=90,
    )
    st.markdown("#### Connect an AI model")
    if provider.startswith("OpenAI"):
        api_key = st.text_input("OpenAI API key", type="password", help="Entered for this session only. Do not paste your key into GitHub or share it.")
        st.caption("OpenAI API uses a separate API account and billing. A ChatGPT subscription does not automatically include API credits.")
    else:
        api_key = st.text_input("Gemini API key", type="password", help="Entered for this session only. Do not paste your key into GitHub or share it.")

    include_ohlc = st.checkbox("Also include my latest candle-data analysis if available", value=True)
    run = st.button("Analyse EUR/USD screenshot", type="primary", use_container_width=True)
    if run:
        if main_chart is None:
            st.error("Upload the main chart screenshot first.")
        elif not api_key.strip():
            st.error("Enter the API key for the selected AI provider.")
        else:
            try:
                images = [(f"Main chart for {instrument} ({candle_minutes}-minute timeframe selected by user)", image_to_data_url(main_chart))]
                if higher_chart_5m is not None:
                    images.append(("Optional 5-minute context chart", image_to_data_url(higher_chart_5m)))
                if higher_chart_15m is not None:
                    images.append(("Optional 15-minute context chart", image_to_data_url(higher_chart_15m)))
                summary = st.session_state.get("ohlc_summary") if include_ohlc else None
                full_context = f"Instrument: {instrument}. Market/feed type: {market_type}. User-selected outlook horizon: {horizon} minutes. User-selected main timeframe: {candle_minutes} minutes. Do not assume the instrument or feed from the screenshot if unclear.\n{context}"
                with st.spinner("AI is reviewing the uploaded chart images. This may take a little time..."):
                    if provider.startswith("OpenAI"):
                        answer = call_openai(images, full_context, api_key.strip(), model_name.strip(), summary)
                    else:
                        answer = call_gemini(images, full_context, api_key.strip(), model_name.strip(), summary)
                st.session_state["last_screenshot_analysis"] = answer
                st.session_state["last_analysis_time"] = datetime.now(timezone.utc).isoformat()
                st.success("Chart review completed. Read the conditions and invalidation carefully.")
            except Exception as exc:
                st.error(f"Screenshot analysis failed: {type(exc).__name__}: {exc}")
                st.info("Check the API key, model name, API billing/access, and internet connection. Do not post your API key in screenshots or GitHub.")

    if "last_screenshot_analysis" in st.session_state:
        st.markdown("---")
        st.subheader(f"AI chart-analysis report — {instrument}")
        st.markdown(st.session_state["last_screenshot_analysis"])
        st.caption("Generated at: " + st.session_state.get("last_analysis_time", "unknown"))
        st.download_button(
            "Download screenshot analysis (.txt)",
            st.session_state["last_screenshot_analysis"],
            file_name=f"{instrument.replace('/', '-').replace(' ', '-')}_analysis.txt",
            mime="text/plain",
        )

with tab_data:
    st.subheader(f"Optional: calculate indicators for {instrument} from actual candle data")
    st.write("For more objective calculations, upload a CSV exported from the same broker/feed. Required columns: `open, high, low, close`; `timestamp` is strongly recommended.")
    uploaded_csv = st.file_uploader("Upload OHLC CSV", type=["csv"], key="ohlc_csv")
    pasted_csv = st.text_area("Or paste OHLC CSV", height=110, placeholder="timestamp,open,high,low,close\n2026-11-04 12:00:00,1.1000,1.1010,1.0990,1.1005")
    if st.button("Calculate indicators", use_container_width=True):
        try:
            raw = load_ohlc(uploaded_csv, pasted_csv)
            analyzed = add_indicators(raw)
            summary = data_summary(analyzed, candle_minutes)
            summary["instrument"] = instrument
            summary["market_feed"] = market_type
            st.session_state["ohlc_data"] = analyzed
            st.session_state["ohlc_summary"] = summary
            st.success("Candle-data calculations completed.")
        except Exception as exc:
            st.error(str(exc))

    if "ohlc_summary" in st.session_state:
        summary = st.session_state["ohlc_summary"]
        a, b, c, dcol = st.columns(4)
        a.metric("EMA trend", summary["ema_trend"])
        b.metric("RSI 14", f'{summary["rsi14"]:.1f}')
        c.metric("BOS", summary["latest_bos"])
        dcol.metric("Candles", summary["candles_supplied"])
        st.json(summary)
        data = st.session_state["ohlc_data"]
        x = data["timestamp"] if "timestamp" in data.columns else data.index
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=x, open=data["open"], high=data["high"], low=data["low"], close=data["close"], name="EUR/USD"))
        fig.add_trace(go.Scatter(x=x, y=data["ema20"], name="EMA 20"))
        fig.add_trace(go.Scatter(x=x, y=data["ema50"], name="EMA 50"))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"{instrument} OHLC with EMA 20 / EMA 50")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download calculated readings (.json)", json.dumps(summary, indent=2), file_name=f"{instrument.replace('/', '-').replace(' ', '-')}_candle_readings.json", mime="application/json")

with tab_guide:
    st.subheader("How to get the best screenshot review")
    st.markdown(
        """
1. Open your selected instrument on Quotex or your broker.
2. Select the timeframe chosen in the sidebar and wait for the current candle to close if possible.
3. Zoom so at least 30–50 recent candles are visible, without hiding the price scale.
4. Keep EMA, RSI, support/resistance, and other indicators visible if you use them.
5. Take a screenshot and upload it in the first tab.
6. If possible, add separate 5-minute and 15-minute screenshots for higher-timeframe context.
7. Choose the 6, 8, or 10-minute outlook and run the analysis.
8. Read the confirmation and invalidation conditions. If the report says NO TRADE, do not force a setup.

**Important limits:** A still image cannot reveal future price movement, exact broker order flow, or a reliable probability of the next candles. The AI may misread chart labels, colors, or patterns. A screenshot from one broker may not match another broker's OTC feed. The app does not connect to Quotex or another broker, execute orders, or monitor live prices.
        """
    )
    st.info("For stronger analysis, provide both screenshots and timestamped OHLC candles from the same instrument/feed. A screenshot-only outlook remains visual and uncertain.")

st.divider()
st.caption("Multi-Pair AI Trading Analyst • No broker connection • No automatic trades • No guaranteed outcomes")
