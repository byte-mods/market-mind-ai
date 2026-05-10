# MarketMind AI

> An institutional-grade, AI-native market intelligence platform for **Indian equity markets** (NSE / BSE) — RL-driven signals, calibrated forecasts, multi-agent debate, RAG over filings, and tax-aware portfolio tooling, all wired to a live Zerodha Kite account.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

---

## Table of Contents

1. [What is MarketMind?](#what-is-marketmind)
2. [Feature Matrix](#feature-matrix)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running](#running)
8. [API Reference](#api-reference)
9. [Project Layout](#project-layout)
10. [Knowledge Graph (graphify)](#knowledge-graph-graphify)
11. [Testing](#testing)
12. [Roadmap](#roadmap)
13. [Contributing](#contributing)
14. [Disclaimer](#disclaimer)
15. [License](#license)

---

## What is MarketMind?

MarketMind is a **single-binary FastAPI server + vanilla-JS SPA** that pipes live Indian-market data into a stack of complementary AI models — reinforcement-learning agents, transformer forecasters, GARCH volatility, regime classifiers, multi-agent debate, and RAG-grounded research — and exposes everything as a clean HTTP/WebSocket API plus a dark-editorial dashboard.

It is designed for **the individual quant** who wants the same calibration discipline (uncertainty bands, walk-forward backtests, conformal prediction) that institutional desks use, while staying close to the Indian retail reality (₹/lakh/crore formatting, FY26 tax rules, SEBI margin awareness, market hours 09:15–15:30 IST).

The system is built around three principles:

1. **Honest uncertainty.** Every price prediction carries 80/95% prediction intervals; every signal carries a calibrated probability. No black-box magic numbers.
2. **Provider-agnostic AI.** A single LLM router lets you swap Claude, DeepSeek, Groq, Ollama, or local `claude` CLI without changing a line of feature code — the cheap models do volume work, the smart models do the reasoning.
3. **Graph-first development.** A persistent knowledge graph (`graphify-out/graph.json`, 2,119 nodes, 55 communities) is queried before any non-trivial change, so refactors don't break invisible dependencies.

---

## Feature Matrix

| Layer | Capability | Status |
|---|---|---|
| **Data** | Zerodha Kite live ticker + REST | ✅ |
| | NSE allIndices, F&O option chain, FII/DII, bulk deals, board meetings | ✅ |
| | BSE corporate announcements, insider disclosures | ✅ |
| | Google News RSS (sector-aware queries) | ✅ |
| | r/IndianStockMarket, ValuePickr, SIAM, GST, IIP/CPI, Google Trends | ✅ |
| **AI / ML** | DQN + PPO + A3C reinforcement-learning ensemble | ✅ |
| | PatchTST transformer forecaster (~150 LOC, in-house) | ✅ |
| | GARCH(1,1) volatility + Holt-Winters trend ensemble | ✅ |
| | Split-conformal prediction wrapper (calibrated 80/95% PIs) | ✅ |
| | Meta-stacker (multinomial logistic) → BUY/SELL/HOLD probabilities | ✅ |
| | HMM regime classifier (Trending Bull / Range / Volatile / Crash / Recovery) | ✅ |
| | Multi-agent debate (Technician + Fundamentalist + Macro + Sentiment + Options) | ✅ |
| | RAG over BSE filings + concall transcripts (ChromaDB + BGE) | ✅ |
| | Event-driven trader (NSE corporate-announcements, severity-scored) | ✅ |
| | News clusterer + sector classifier (TF-IDF cosine) | ✅ |
| **Quant** | Anchored walk-forward backtester with bootstrap Sharpe distribution | ✅ |
| | Markowitz mean-variance optimiser (max-Sharpe, min-var, risk-parity) | ✅ |
| | Efficient frontier with highlighted optimal points | ✅ |
| | Black-Scholes options pricing + 5 Greeks | ✅ |
| | 9 strategy templates (covered call, condor, calendar, ratio, …) | ✅ |
| | Vectorised payoff curves, break-evens, signed-additive Greeks | ✅ |
| | **IV rank / percentile** (252-day rolling, Mongo-backed) | ✅ |
| | **Volatility surface** — multi-expiry (strike, expiry, IV) grid | ✅ |
| | **Term structure** — contango / backwardation / flat detection | ✅ |
| | **Skew analytics** — 25Δ risk reversal, put/call skew index, smile shape | ✅ |
| | Historical + parametric VaR, stress scenarios | ✅ |
| **Indian moat** | Tax-aware rebalancer (STCG/LTCG, ₹1.25L exemption, FY tracking) | ✅ |
| | Multi-asset (MCX commodities, USD/INR, INR-paired crypto) | ✅ |
| | SEBI compliance gate (insider window, position limits, audit log) | ✅ |
| **Frontier** | Causal Bayesian network (PC-algorithm + linear-Gaussian counterfactuals) | ✅ |
| | Hierarchical Risk Parity + Black-Litterman views | ✅ |
| **UI** | Editorial-dark dashboard (Instrument Serif + Geist + JetBrains Mono) | ✅ |
| | Live ticker strip, sparklines, sector heatmap (Nifty 500) | ✅ |
| | WebSocket push (30s tick), toast notifications | ✅ |

✅ = shipped · ⏳ = on roadmap · 🔒 = blocked. See [`plan.md`](plan.md) for the canonical roadmap.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Browser SPA                                │
│  static/index.html — vanilla JS, no build step, dark editorial theme    │
└──────────┬─────────────────────────────────────────────────┬────────────┘
           │ HTTP /api/*                                     │ WS /ws (30s)
           ▼                                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FastAPI server (server.py)                       │
│   ╭───────────────────────────────────────────────────────────────╮     │
│   │                    AppController (singleton)                  │     │
│   │  Orchestrates fetchers, RL signals, news, regime, alerts.     │     │
│   ╰────────┬──────────────────────────────────────────────────────╯     │
└────────────┼─────────────────────────────────────────────────────────────┘
             │
   ┌─────────┼──────────────┬──────────────┬──────────────┐
   ▼         ▼              ▼              ▼              ▼
┌──────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐ ┌──────────────┐
│Kite  │ │PriceFetch│ │NewsFetch │ │ Vector DB   │ │ LLM Router   │
│Client│ │ NSE→Kite │ │Google RSS│ │ (ChromaDB)  │ │ Claude/DSeek │
│ REST │ │ →Screener│ │ +sectors │ │ filings/RAG │ │ /Groq/Ollama │
│ +WS  │ │  fallback│ │ clusterer│ └─────────────┘ └──────────────┘
└──────┘ └──────────┘ └──────────┘
                                    ┌──────────────────────────────┐
                                    │       MongoDB                │
                                    │ caches (10m TTL): fetch_cache,│
                                    │   news, prices, rl_signals   │
                                    │ persistent: alt_signals,     │
                                    │   forecast_cache (24h/5m),   │
                                    │   events, lots (W4.1)        │
                                    └──────────────────────────────┘
                                    ┌──────────────────────────────┐
                                    │        SQLite                │
                                    │  marketmind.db, candles.db   │
                                    └──────────────────────────────┘

ML Pipeline (offline + online)
─────────────────────────────────────────────────────────────────────────
  raw OHLCV ──► features ──┬─► RL ensemble  (DQN + PPO + A3C)   ┐
                            ├─► PatchTST ──┐                      │
                            ├─► Holt-Winters ┼─► EnsembleForecaster─► Conformal
                            └─► GARCH ────┘                      │      │
                                                                  │      ▼
                            regime ──► gating ──────────────────►─┴► MetaStacker
                                                                  │      │
                            sentiment ───────────────────────────►─┘      │
                                                                          ▼
                                                            BUY/SELL/HOLD probs
                                                            + 95% return CI
```

### Why this shape?
- **Single FastAPI process** keeps deploy trivial (`./.venv/bin/python main.py`) and avoids the ops cost of a microservice fan-out for a one-developer system.
- **Singleton `AppController`** holds heavy fetchers/models so HTTP routes stay thin.
- **All NSE/Screener fetchers** route through `_get_nse_session()` and a shared 10-min Mongo TTL — calling NSE directly trips rate-limits and breaks every downstream feature.
- **WebSocket push every 30 s** instead of per-tick, because Zerodha's WebSocket tick rate is too noisy for a UI and would saturate browser memory.
- **Vanilla JS SPA** has zero build step, ships straight from `static/index.html`, and is friendly to any contributor who can read 2010-era JavaScript.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.14** | Modern type-checking, ergonomic async |
| Web | **FastAPI + Uvicorn** | ASGI, Pydantic-typed contracts, OpenAPI for free |
| Frontend | **Single-page HTML/JS** | No build step, no framework lock-in |
| Cache | **MongoDB** (10-min TTL on volatile collections) | Schema-flexible, TTL primitives |
| Storage | **SQLite** (`marketmind.db`, `candles.db`) | Zero-ops local persistence |
| Vector DB | **ChromaDB** | Local, free, BGE-large embeddings |
| Brokerage | **Zerodha Kite Connect** | REST + WebSocket ticker, India-native |
| RL | **PyTorch** (DQN + PPO + A3C, in-house) | Full control, no opaque libraries |
| Forecast | **statsmodels + arch + custom PatchTST** | Calibrated, not magic |
| LLM | **Anthropic Claude** (default) via swappable router | Best reasoning for debate/RAG |
| News | **Google News RSS** (sector-aware queries) | Free, no auth, replaces DDG scraping |

### Hard rules (encoded from prior incidents)

- **Never re-add `yfinance`.** It returns 404 for delisted/changed Indian symbols. NSE+Kite is authoritative.
- **Never put secrets in committed files.** `local.json` is local-only. `.env` is `.gitignore`d.
- **Never bypass the cache.** Direct NSE calls trip rate limits and cascade-fail every feature.
- **Never strip the brotli fix.** NSE returns brotli-encoded JSON if `Accept-Encoding` advertises `br`. Removing `br` from the header (one character) is what unblocked heatmap, sectors, options chain, bulk deals, fundamentals, and historicals.
- **Indian-market assumptions are pervasive.** Sectors, currency formatting (₹, lakh, crore), market hours (IST 09:15–15:30), tax regime (FY Apr–Mar). Don't generalise without thinking.

---

## Installation

### Prerequisites

- **Python 3.14** (3.12+ works; 3.14 is the development target)
- **MongoDB** running locally (or set `MONGO_URI` to a remote instance)
- **Zerodha Kite Connect** account + API key (for live data; offline-only mode works without)
- **Anthropic API key** (for chat/research/debate/RAG; optional if `LLM_BACKEND=claude_cli`)

### Clone + venv

```bash
git clone https://github.com/<your-fork>/marketmind.git
cd marketmind
python3.14 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -r requirements-dev.txt   # for tests
```

### MongoDB

```bash
# macOS (Homebrew)
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community

# Linux (Docker)
docker run -d -p 27017:27017 --name mongo mongo:7
```

The default URI `mongodb://localhost:27017` and database name `marketmind` are wired in — no schema setup required, collections are created lazily.

---

## Configuration

You have two equivalent paths. **Env vars take precedence over `local.json`.**

### Path A — `.env` (recommended)

```bash
cp .env.example .env
# edit .env, fill in real keys
```

### Path B — `local.json` (legacy, also gitignored)

```json
{
  "api_key": "your_kite_api_key",
  "api_secret": "your_kite_api_secret",
  "access_token": "your_kite_access_token_for_today",
  "anthropic": { "api_key": "sk-ant-..." }
}
```

### Kite access token

Kite tokens expire daily at 06:00 IST. The login flow is:

1. Hit `GET /api/kite/login-url` → opens Zerodha login page.
2. After login, Zerodha redirects with `request_token=...` in the URL.
3. POST that token to `/api/kite/session` — server exchanges it for an `access_token` and persists to `local.json`.

You'll need to repeat this once per trading day.

---

## Running

```bash
./.venv/bin/python main.py
```

This:

1. Boots FastAPI on `http://127.0.0.1:8000` (configurable via `HOST`/`PORT`).
2. Starts a 30-second WebSocket broadcast loop pushing live prices, RL signals, news, regime.
3. Opens your default browser to the dashboard.

For development with auto-reload:

```bash
./.venv/bin/uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

For production (single-machine deployment):

```bash
./.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Don't run multiple workers.** AppController is a process-singleton holding model state and the Mongo connection pool — multi-worker mode duplicates everything and breaks the WS broadcast invariant. If you need horizontal scale, you need a real architecture redesign first.

---

## API Reference

The server exposes ~80 routes under `/api/*`. The full list is auto-documented at [`/docs`](http://127.0.0.1:8000/docs) (Swagger) and [`/redoc`](http://127.0.0.1:8000/redoc) (ReDoc) when the server is running.

Highlight tour:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/portfolio` | Live holdings + day positions |
| `GET` | `/api/portfolio/equity-curve` | Historical NAV curve |
| `POST` | `/api/optimize` | Portfolio optimisation (max_sharpe, min_variance, risk_parity, equal_weight, hrp, black_litterman) |
| `POST` | `/api/optimize/frontier` | Efficient frontier (20 points) |
| `POST` | `/api/optimize/black-litterman` | Black-Litterman with user views + market prior |
| `POST` | `/api/risk/portfolio` | VaR + stress + sector betas |
| `POST` | `/api/backtest/walkforward` | Anchored walk-forward Sharpe distribution |
| `GET` | `/api/forecast/{sym}?horizon=N&model=ensemble` | Calibrated price forecast + 80/95 PI |
| `GET` | `/api/signal/{sym}/calibrated?horizon=N` | BUY/SELL/HOLD probabilities + 95% return CI |
| `GET` | `/api/regime` | Current market regime + transition probabilities |
| `POST` | `/api/debate` | Multi-agent stock verdict |
| `POST` | `/api/research/{sym}/grounded` | RAG-cited research answer |
| `GET` | `/api/events?since=...&min_severity=60` | Material corporate events |
| `GET` | `/api/altdata` | Reddit + ValuePickr + SIAM + GST + IIP + Trends |
| `POST` | `/api/options/strategy` | Build & analyse a 9-template options strategy |
| `GET` | `/api/options/iv-rank/{sym}` | IV rank [0,100] + percentile from 252-day history |
| `GET` | `/api/options/vol/{sym}` | Full vol analytics: surface + term structure + skew + IV rank |
| `GET` | `/api/causal/nodes` | Causal network nodes + current values + DAG edges |
| `POST` | `/api/causal/whatif` | Counterfactual estimate: ``do(node=value) → target`` |
| `GET` | `/api/market/heatmap` | 499-stock Nifty 500 heatmap |
| `GET` | `/api/market/indices` | Nifty 500 + sectoral indices |
| `GET` | `/api/sectors/correlations` | 9-sector correlation matrix |
| `GET` | `/api/rl/signals` | Multi-timeframe RL BUY/SELL signals |
| `GET` | `/api/bulk-deals` | NSE bulk + block deals |
| `GET` | `/api/earnings-calendar` | Upcoming board meetings |
| `GET` | `/api/fo-sentiment` | PCR-based fear/greed gauge |
| `WS` | `/ws` | 30-second push: prices, RL, news, regime |

Many more — see Swagger.

---

## Project Layout

```
marketmind/
├── main.py                      # entry point: boots FastAPI + opens browser
├── server.py                    # all HTTP / WebSocket routes (~2,650 LOC)
├── plan.md                      # canonical roadmap (Wave 0 → Wave 5)
├── CLAUDE.md                    # graph-first dev rules for AI assistants
├── pyproject.toml               # pytest config
├── requirements.txt             # runtime deps
├── requirements-dev.txt         # test deps
├── local.json                   # secrets (gitignored; or use .env)
├── .env.example                 # env template (committed)
├── .gitignore
├── LICENSE                      # Apache 2.0
├── NOTICE                       # third-party attributions
├── README.md                    # this file
│
├── marketmind/
│   ├── app_controller.py        # singleton orchestrator
│   ├── core/
│   │   ├── kite_client.py       # Zerodha REST + WebSocket
│   │   ├── price_fetcher.py     # NSE → Kite → Screener fallback chain
│   │   └── kite_candles.py      # historical OHLCV cache
│   ├── analysis/
│   │   ├── portfolio_optimizer.py     # Markowitz max-Sharpe / min-var / RP
│   │   ├── risk_engine.py             # VaR, stress, sector betas
│   │   ├── correlations.py            # rolling correlation matrix
│   │   ├── portfolio_simulator.py     # Monte Carlo
│   │   └── market_predictor.py        # heuristic + ML composite
│   ├── ml/
│   │   ├── trading_env.py             # OpenAI-Gym-style RL environment
│   │   ├── rl_agent.py                # DQN/PPO/A3C wrappers
│   │   ├── ensemble.py                # signal blender
│   │   ├── forecast/                  # PatchTST + GARCH + Holt-Winters
│   │   │   ├── ensemble.py
│   │   │   ├── conformal.py           # split-conformal calibration
│   │   │   ├── meta_stacker.py        # softmax BUY/SELL/HOLD
│   │   │   ├── evaluator.py           # PI coverage harness
│   │   │   └── cache.py               # Mongo forecast_cache (24h/5m TTL)
│   │   └── options/                   # BS pricing + 9 strategy templates + vol analytics
│   │       ├── pricing.py
│   │       ├── strategies.py
│   │       ├── builder.py             # vectorised payoff curves
│   │       └── vol_analytics.py       # IV rank, surface, term structure, skew
│   ├── vectordb/                      # ChromaDB persistence
│   └── models/                        # PyTorch checkpoints (gitignored)
│
├── static/
│   └── index.html               # vanilla JS dashboard (single page)
│
├── tests/                       # 27 test files, ~500+ tests
│   ├── conftest.py
│   ├── test_harness.py
│   ├── test_server_wiring.py
│   ├── altdata/                 # alt-data fetcher tests
│   ├── forecast/                # PatchTST / GARCH / conformal tests
│   ├── options/                 # BS / strategies / builder tests
│   ├── test_api_signal_calibrated.py
│   ├── test_api_options_strategy.py
│   ├── test_api_forecast.py
│   └── test_api_altdata.py
│
└── graphify-out/                # knowledge-graph artifacts (committed)
    ├── graph.json               # 2,119 nodes, 4,838 edges, 55 communities
    ├── graph.html               # interactive viewer (open in browser)
    └── GRAPH_REPORT.md          # community labels + god nodes + audit
```

---

## Knowledge Graph (graphify)

This repo is built **graph-first**. Before touching any subsystem we query a persistent knowledge graph to find the blast radius — what depends on what, which "god nodes" sit at the centre, which surprising edges hide between modules.

Quick reference:

```bash
/graphify .                                          # full rebuild
/graphify . --update                                 # incremental (after code changes)
/graphify query "what is connected to X"             # BFS dependency map
/graphify query "what depends on X"                  # reverse dependency
/graphify path "AppController" "TradingEnvironment"  # shortest path between two concepts
/graphify explain "PortfolioOptimizer"               # plain-language node explanation
```

**God nodes** (most connected): `AppController` (125), `SectorClassifier` (107), `KiteClient` (103), `EnsembleForecaster` (83), `PriceFetcher` (62).

Open `graphify-out/graph.html` in any browser for the interactive viewer (no server needed). Read `graphify-out/GRAPH_REPORT.md` for the human-readable audit.

The strict rule (also in `CLAUDE.md`): **never** `grep` or `ls` the full repo for orientation; always start from a graphify query, then read targeted files.

---

## Testing

```bash
./.venv/bin/pytest                    # full suite
./.venv/bin/pytest tests/forecast/    # single subsystem
./.venv/bin/pytest -k "conformal"     # by keyword
./.venv/bin/pytest -x --ff            # stop on first fail, fail-fast first
```

Coverage targets:

- **Unit tests** for every pure module (forecaster, conformal, meta-stacker, options pricing, regime, etc.).
- **Integration tests** with stand-in FastAPI apps that mirror `server.py` logic without booting Mongo / Kite / Anthropic.
- **Property-style invariants** where they exist (e.g. tax-saved ≤ naive tax for any random seed; PI coverage ≥ 0.75 OOS).

A typical PR runs **all** tests in <60 s on an M-series Mac. CI (when set up) runs the same.

---

## Roadmap

The full roadmap lives in [`plan.md`](plan.md). Current state at a glance:

- **Wave 0 — Foundation** ✅ LLM router + Vector DB
- **Wave 1 — AI feel** ✅ Multi-agent debate + Regime classifier + Walk-forward backtest
- **Wave 2 — Information edge** ✅ RAG + Event-driven trader + Alt-data
- **Wave 3 — Quant-grade** ✅ Forecasting models + Conformal stacking + Options builder
- **Wave 4 — Indian moat** ⏳ Tax-aware rebalancer (in progress) + Multi-asset
- **Wave 5 — Frontier** ⏳ Causal Bayes net + HRP + Black-Litterman + SEBI compliance

> Time to "noticeably more intelligent": end of Wave 1 (~10 days). Time to "institutional-grade": end of Wave 3 (✅ shipped). Time to "Indian-moat-defensible": end of Wave 4.

---

## Contributing

This is currently a single-author project. PRs and issue reports are welcome on GitHub. If you contribute:

1. **Run `/graphify query` before any non-trivial change** — discover the blast radius. Don't grep the whole repo.
2. **Tests in the same commit as code.** Tests-later is tests-never.
3. **Match the surrounding style.** No imported style guide; surrounding code is the canon.
4. **Run the suite before opening a PR**: `./.venv/bin/pytest`.
5. **Run `/graphify . --update` after your change** — the graph is checked in and stays current.
6. **No `unwrap`/`panic`/`expect`-style bombs on production paths.** Errors must be explicit.
7. **Don't commit secrets.** Pre-commit hook recommended (e.g. `gitleaks`).

---

## Disclaimer

> **MarketMind is research software, not investment advice.**
>
> Outputs from this system — RL signals, forecasts, debate verdicts, tax suggestions, rebalance trades — are informational. They are produced by statistical models with stated uncertainty bands and known failure modes. They do **not** constitute financial, investment, tax, or legal advice.
>
> You are solely responsible for any trading decisions you make. Markets are uncertain; models can be wrong; data can be stale; bugs exist. Past performance, including any reported backtest, is not indicative of future results.
>
> The author and contributors disclaim all liability for any loss arising from use of this software, to the maximum extent permitted by law (see Apache License §7–§8).
>
> If you are managing other people's money, you must independently comply with **SEBI** regulations (registration, PDA, PMS/RIA licensing, insider-trading windows, position limits, audit logs). The (planned) `W5.3 SEBI compliance layer` is a starting point — not a substitute for legal counsel.

---

## License

Copyright © 2026 Sudeep Dasgupta.

Licensed under the **Apache License, Version 2.0** (the "License"); you may not use this software except in compliance with the License. You may obtain a copy of the License at:

> http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, **WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND**, either express or implied. See [`LICENSE`](LICENSE) for the full text and [`NOTICE`](NOTICE) for third-party attributions.
