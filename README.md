# Apex Multi-Pair AI Trading Analyst

A free-first Streamlit application for evidence-led chart review using Google's Gemini API.

## Features
- Multiple forex pairs and selected instruments
- Quick Trade mode with flexible prediction horizons
- Flexible chart timeframes, including automatic-detection option
- Multi-timeframe screenshot analysis
- Multi-pair screenshot scanner (user-uploaded charts only; not a live market feed)
- OHLC CSV calculations for EMA 20/50, RSI 14 and MACD
- BUY / SELL / NO TRADE analysis with confirmation and invalidation conditions
- Local prediction history with manual outcome verification

## Run locally
1. Install Python 3.10 or newer.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`
4. Enter a Gemini API key in the app sidebar.

## Gemini model
The default is `gemini-3.8-flash`, a stable model listed in Google's Gemini API model documentation. Model availability, free-tier eligibility, quotas and pricing depend on the Google AI Studio project and may change. The app does not guarantee unlimited free API usage.

## Deploy
Deploy as a Streamlit app and set `GEMINI_API_KEY` in the hosting platform's secrets/environment settings. Never commit API keys to GitHub.

For Streamlit secrets, add:
```toml
GEMINI_API_KEY = "your-key-here"
```

## Important
This is a decision-support tool, not financial advice. It does not connect to a broker, receive live prices, execute trades or guarantee accuracy. Multi-pair scanning reviews only the screenshots users upload. Prediction history uses local SQLite storage; free hosts may not preserve local files across restarts unless persistent storage is configured.
