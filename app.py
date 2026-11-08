import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Multi-Pair Candle Analyst",
    page_icon="ðŸ“Š",
    layout="wide",
)

st.title("ðŸ“Š Multi-Pair Candle Analyst")
st.caption("M2 candle analysis â€¢ 8â€“10 minute decision support â€¢ maximum 30 candles")

st.warning(
    "Decision support only. BUY, SELL and WAIT are rule-based technical signals, "
    "not guaranteed predictions. This app does not connect to broker prices or place trades. "
    "Use verified OHLC data from the selected market."
)

# ------------------------- Market selection -------------------------
st.header("1. Select your market")

market = st.radio(
    "Market environment",
    ["Real Forex", "Quotex OTC"],
    horizontal=True,
)

common_pairs = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "AUD/JPY", "EUR/AUD", "GBP/AUD",
]

if market == "Quotex OTC":
    st.caption(
        "OTC symbols and availability depend on your Quotex account. "
        "Choose the matching instrument shown by your broker."
    )
    pair_options = common_pairs + [
        "EUR/USD OTC", "GBP/USD OTC", "USD/JPY OTC", "EUR/JPY OTC",
        "AUD/USD OTC", "GBP/JPY OTC", "Custom OTC instrument",
    ]
else:
    pair_options = common_pairs + ["Custom Forex instrument"]

instrument_choice = st.selectbox("Currency pair / instrument", pair_options)
if instrument_choice.startswith("Custom"):
    instrument = st.text_input("Enter the exact broker instrument name", placeholder="e.g. USD/NGN")
else:
    instrument = instrument_choice

c1, c2 = st.columns(2)
with c1:
    st.metric("Candle timeframe", "M2 Â· 2 minutes")
with c2:
    horizon = st.selectbox("Decision horizon", [8, 10], index=0, format_func=lambda x: f"{x} minutes")

st.info(
    f"Selected environment: **{market}** | Instrument: **{instrument or 'Not entered'}** "
    f"| Horizon: **{horizon} minutes**"
)

# ------------------------- OHLC input -------------------------
st.header("2. Upload or paste M2 OHLC candles")
st.write(
    "Provide candles in chronological order (oldest first). Required columns: "
    "`open, high, low, close`. A timestamp column is recommended."
)

uploaded = st.file_uploader("Upload OHLC CSV", type=["csv"])
pasted = st.text_area(
    "Or paste CSV candle data",
    height=160,
    placeholder="timestamp,open,high,low,close\n2026-11-08 10:00,1.1500,1.1510,1.1490,1.1505",
)

def load_candles(file_obj, pasted_text):
    if file_obj is not None:
        frame = pd.read_csv(file_obj)
    elif pasted_text.strip():
        frame = pd.read_csv(io.StringIO(pasted_text.strip()))
    else:
        raise ValueError("Upload a CSV file or paste OHLC candle data.")

    frame.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in frame.columns
    ]
    aliases = {
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "close_price": "close",
    }
    frame = frame.rename(columns=aliases)

    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError("Missing required OHLC columns: " + ", ".join(missing))

    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=required).copy()
    if frame.empty:
        raise ValueError("No valid numeric OHLC rows were found.")

    if "timestamp" in frame.columns:
        parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
        if parsed.notna().any():
            frame["_sort_time"] = parsed
            frame = frame.sort_values("_sort_time").drop(columns="_sort_time")

    invalid = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} row(s) contain inconsistent OHLC values. "
            "Check that high is the highest price and low is the lowest price."
        )

    original_count = len(frame)
    frame = frame.tail(30).reset_index(drop=True)
    return frame, original_count

def calculate_indicators(frame):
    data = frame.copy()
    close = data["close"]

    data["ema20"] = close.ewm(span=20, adjust=False).mean()
    data["ema50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["rsi14"] = 100 - (100 / (1 + rs))
    data.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100
    data.loc[(avg_gain == 0) & (avg_loss > 0), "rsi14"] = 0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]
    return data

def make_decision(data):
    last = data.iloc[-1]
    previous = data.iloc[-2]
    score = 0
    reasons = []

    # Trend: EMA relationship and short-term EMA slope.
    if last["ema20"] > last["ema50"]:
        score += 1
        reasons.append("EMA 20 is above EMA 50 (bullish trend structure).")
    elif last["ema20"] < last["ema50"]:
        score -= 1
        reasons.append("EMA 20 is below EMA 50 (bearish trend structure).")

    if last["ema20"] > previous["ema20"]:
        score += 1
        reasons.append("EMA 20 is rising.")
    elif last["ema20"] < previous["ema20"]:
        score -= 1
        reasons.append("EMA 20 is falling.")

    # RSI: avoid treating overbought/oversold as an automatic reversal.
    rsi = last["rsi14"]
    if pd.notna(rsi):
        if 52 <= rsi <= 68:
            score += 1
            reasons.append(f"RSI 14 is {rsi:.1f}, supporting bullish momentum.")
        elif 32 <= rsi <= 48:
            score -= 1
            reasons.append(f"RSI 14 is {rsi:.1f}, supporting bearish momentum.")
        elif rsi > 75:
            reasons.append(f"RSI 14 is {rsi:.1f} (high); bullish entries may be stretched.")
        elif rsi < 25:
            reasons.append(f"RSI 14 is {rsi:.1f} (low); bearish entries may be stretched.")
        else:
            reasons.append(f"RSI 14 is {rsi:.1f} (neutral momentum zone).")

    # MACD direction and histogram.
    if last["macd"] > last["macd_signal"] and last["macd_hist"] > 0:
        score += 1
        reasons.append("MACD is above its signal line with a positive histogram.")
    elif last["macd"] < last["macd_signal"] and last["macd_hist"] < 0:
        score -= 1
        reasons.append("MACD is below its signal line with a negative histogram.")

    # Recent candle closes: use last three completed candles.
    recent = data.tail(3)
    bullish = int((recent["close"] > recent["open"]).sum())
    bearish = int((recent["close"] < recent["open"]).sum())
    if bullish >= 2:
        score += 1
        reasons.append(f"{bullish} of the last 3 candles closed bullish.")
    elif bearish >= 2:
        score -= 1
        reasons.append(f"{bearish} of the last 3 candles closed bearish.")

    if score >= 3:
        decision = "BUY"
        explanation = "Bullish evidence is aligned across several indicators."
    elif score <= -3:
        decision = "SELL"
        explanation = "Bearish evidence is aligned across several indicators."
    else:
        decision = "WAIT"
        explanation = "The available signals are mixed or not strong enough."

    return decision, score, explanation, reasons

if st.button("Analyse latest 30 M2 candles", type="primary", use_container_width=True):
    if not instrument.strip():
        st.error("Select or enter the market instrument first.")
    else:
        try:
            candles, supplied_count = load_candles(uploaded, pasted)
            if len(candles) < 26:
                st.error(
                    f"Only {len(candles)} valid candles were provided. "
                    "Please provide at least 26 candles for the MACD calculation. "
                    "The app will still analyse no more than 30 candles."
                )
            else:
                result = calculate_indicators(candles)
                decision, score, explanation, reasons = make_decision(result)
                st.session_state["last_candle_analysis"] = {
                    "market": market,
                    "instrument": instrument,
                    "horizon": horizon,
                    "candles": result,
                    "decision": decision,
                    "score": score,
                    "explanation": explanation,
                    "reasons": reasons,
                    "supplied_count": supplied_count,
                }
        except Exception as exc:
            st.error(f"Could not analyse the CSV: {exc}")

# ------------------------- Results -------------------------
if "last_candle_analysis" in st.session_state:
    report = st.session_state["last_candle_analysis"]
    result = report["candles"]
    last = result.iloc[-1]

    st.header("3. Candle analysis result")
    st.caption(
        f"{report['market']} â€¢ {report['instrument']} â€¢ M2 candles â€¢ "
        f"{len(result)} analysed (maximum 30) â€¢ {report['horizon']}-minute horizon"
    )

    decision = report["decision"]
    if decision == "BUY":
        st.success("DECISION: BUY")
    elif decision == "SELL":
        st.error("DECISION: SELL")
    else:
        st.warning("DECISION: WAIT")

    st.subheader(report["explanation"])
    a, b, c, d = st.columns(4)
    a.metric("Candles analysed", f"{len(result)} / 30")
    b.metric("EMA 20", f"{last['ema20']:.6f}")
    c.metric("EMA 50", f"{last['ema50']:.6f}")
    d.metric("RSI 14", "â€”" if pd.isna(last["rsi14"]) else f"{last['rsi14']:.1f}")

    e, f, g = st.columns(3)
    e.metric("MACD", f"{last['macd']:.6f}")
    f.metric("MACD signal", f"{last['macd_signal']:.6f}")
    g.metric("Signal score", f"{report['score']:+d} (rule-based)")

    st.markdown("**Why this decision was produced**")
    for reason in report["reasons"]:
        st.write("â€¢ " + reason)

    st.subheader("Latest candle readings")
    display_cols = ["open", "high", "low", "close", "ema20", "ema50", "rsi14", "macd", "macd_signal"]
    st.dataframe(result[display_cols].tail(10), use_container_width=True)

    st.subheader("Recent closing prices")
    chart_data = result[["close", "ema20", "ema50"]].copy()
    chart_data.index = range(1, len(chart_data) + 1)
    st.line_chart(chart_data, use_container_width=True)

    st.info(
        "Important limitations: this is a technical bias assessment, not a measured "
        "probability or a guarantee of where price will be in 8â€“10 minutes. "
        "EMA 50 is calculated from the available 30-candle window and is therefore "
        "not fully warmed up; for a more stable EMA 50, supply longer historical data "
        "only if the analysis design is later changed to calculate indicators on history "
        "while limiting the decision window to the latest 30 candles."
    )

    csv_out = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download analysed candle data",
        data=csv_out,
        file_name="m2_candle_analysis.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Multi-Pair Candle Analyst â€¢ Manual OHLC input only â€¢ No live broker feed â€¢ "
    "No automatic order execution"
)
