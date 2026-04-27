---
name: MarketMind Project Architecture
description: Core stack, data sources, key design decisions, and feature list
type: project
---

MarketMind AI v2.0 is now a **browser-based** FastAPI + vanilla JS web app (was PySide6 desktop).

**Why yfinance was removed:** Returned 404s for TATAMOTORS.NS and delisted symbols. NSE India API is authoritative.

**Stack:**
- Backend: FastAPI (server.py) + uvicorn, all existing Python core logic preserved
- Frontend: static/index.html — single-page app with Chart.js + D3.js (no framework)
- Entry point: main.py — starts uvicorn on port 8000, opens browser automatically
- Data: NSE India API (prices/history), Screener.in (fundamentals), Zerodha Kite Connect (live + trading), Google News RSS

**Key files:**
- server.py — all API routes (/api/*)
- static/index.html — complete SPA frontend
- marketmind/app_controller.py — orchestrates all data/trading
- marketmind/core/price_fetcher.py — NSE + Screener.in
- marketmind/core/options_fetcher.py — NSE options chain, PCR, max pain
- marketmind/core/earnings_calendar.py — NSE corporate actions
- marketmind/core/news_clusterer.py — TF-IDF cosine similarity clustering
- marketmind/core/backtester.py — historical signal backtesting

**All 10 new features implemented:**
1. Intraday heatmap — color grid of Nifty 50 by % change with sector tabs
2. Options chain viewer — OI table, PCR, max pain, OI bar chart, ATM highlighting
3. Earnings calendar — NSE corporate actions API with fallback
4. Screener/filter — PE, ROE, ROCE, momentum, RSI, market cap filters + score ranking
5. Alert system — SQLite-backed price alerts + WebSocket browser notifications
6. Backtester — swing MA / RSI / MACD strategies, equity curve, Sharpe, drawdown
7. F&O sentiment — PCR fear/greed gauge, OI distribution, interpretation text
8. Portfolio P&L chart — equity curve + drawdown, holdings pie
9. News clustering — TF-IDF cosine grouping, unique stories surfaced first
10. Multi-timeframe RL signals — intraday/swing/positional separate columns

**Sector correlation node click:** clicking a node in the D3 network graph calls
/api/sectors/{sector}/recommendations → returns top stocks with fundamentals, technicals,
entry/target/SL levels, score breakdown, reasoning text.

**WebSocket /ws:** broadcasts live index updates every 30s and triggered price alerts.
