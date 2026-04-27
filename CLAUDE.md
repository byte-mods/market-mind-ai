# MarketMind AI

## Graph-First Development
- **Mandatory:** Before starting any task, query `graphify-out/graph.json` to find the relevant community.
- **Token Efficiency:** Never use `grep` or `ls` on the full repo. Use the graph's "God Nodes" and "Community Clusters" to isolate the work.
- **Validation:** If a task touches `AppController`, `KiteClient`, or `PriceFetcher`, check the graph for "Surprising Connections" to ensure no side effects in the RL, news, or portfolio layers.

## Auto-Query Rule (STRICT — follow every time)

Before writing, editing, or planning ANY code change, you MUST automatically run the appropriate `/graphify` command FIRST. Do NOT wait for the user to ask. Do NOT skip this step.

### When the user says "build X" / "add X" / "fix X" / "implement X":
1. Run `/graphify query "what is connected to X"` to find the relevant community, files, and dependencies
2. Run `/graphify query "what depends on X"` to check for side effects
3. If the task touches two concepts, also run `/graphify path "A" "B"` to find the shortest path
4. THEN and ONLY THEN start writing code

### When the user says "how does X work" / "explain X":
1. Run `/graphify explain "X"` or `/graphify query "how does X work"`
2. Answer using graph output + source file references

### When the user says "what's missing" / "what's broken" / "plan next steps":
1. Run `/graphify query "which modules are isolated or disconnected"`
2. Run `/graphify query "what API endpoints have no working implementation"`
3. Use findings to build the response

### After finishing any code change:
1. Run `/graphify . --update` to keep the graph current

### Quick reference
- God Nodes (top hubs): `AppController`, `SectorClassifier`, `KiteClient`, `TradingEnvironment`, `PriceFetcher`
- Community map: see `graphify-out/GRAPH_REPORT.md`
- Active issues: see `plan.md`

## Tech Stack
- **Backend:** Python 3.14, FastAPI, Uvicorn (ASGI)
- **Frontend:** Single-page HTML/JS in `static/index.html` (no framework, no build step)
- **Cache/DB:** MongoDB (`marketmind` DB, 10-min TTL on `fetch_cache`, `news`, `prices`, `rl_signals`); SQLite (`marketmind.db`, `candles.db`)
- **Brokerage:** Zerodha Kite Connect (REST + WebSocket ticker)
- **Data Sources:**
  - **Prices/Indices:** Kite (live) → NSE India API (fallback) → Screener.in (fundamentals)
  - **News:** Google News RSS (sector-aware queries) — replaced DuckDuckGo scraping
  - **Macro:** NSE allIndices (Nifty PE/PB), RBI rate history (manual)
  - **F&O:** NSE option-chain endpoint
  - **FII/DII, Bulk Deals:** NSE API
- **AI:**
  - **RL ensemble:** DQN + PPO + A3C (PyTorch) trained on NSE historical data
  - **Research/Chat:** Anthropic Claude API (`MarketAssistant`, `claude_research`)
  - **News classification:** TF-IDF cosine clusterer + sector keyword matcher
- **Legacy (do not edit):** `marketmind/main_window.py` and `marketmind/ui/*.py` are PySide6 desktop UI from the pre-web era. Not imported by `server.py`. PySide6 isn't even installed in `.venv`.

## Run / Dev
- `./.venv/bin/python main.py` — boots FastAPI on `http://127.0.0.1:8000`, opens browser
- Auth: Kite via `local.json` (`api_key`, `api_secret`, `access_token`); Anthropic via `local.json:anthropic.api_key`
- All API routes in `server.py`; route prefix `/api/`
- WebSocket live updates on `/ws`

## Hard Rules (from prior incidents)
- **Never re-add `yfinance`.** It returns 404 for delisted/changed Indian symbols (e.g. `TATAMOTORS.NS`). NSE+Kite is authoritative.
- **Never put secrets in committed files.** `local.json` is local-only; do not commit.
- **Never bypass the cache.** All NSE/Screener fetchers route through `_get_nse_session()` and the 10-min Mongo TTL — calling NSE directly will trip rate limits and break the whole app.
- **The user is on Indian markets (NSE/BSE).** Sectors, keywords, currency formatting (₹, lakh/crore), market hours (IST 09:15–15:30) all assume India.
