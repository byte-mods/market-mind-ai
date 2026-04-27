# MarketMind — Roadmap

> Forward build plan to evolve from "working trading dashboard" to **super-AI-intelligent market intelligence platform**.
> Repair plan completed 2026-04-27 (15/15 issues). See archive at bottom.

Status: ⏳ pending · 🟡 in progress · ✅ done · 🔒 blocked.

Effort scale: **XS** (≤2h) · **S** (½ day) · **M** (1–2 days) · **L** (3–7 days) · **XL** (>1 week).

---

## Wave 0 — Foundation (build these first; everything else depends)

### F1. LLM router (configurable backend) ✅ M
A single `llm.py` interface so every AI feature is provider-agnostic.

```
config: {
  llm_backend: "claude_cli" | "claude_api" | "deepseek" | "groq" | "ollama" | "openai",
  models: { research: "claude-opus-4-7", debate: "claude-sonnet-4-6", classify: "haiku-4-5" }
}
```

- Adapter classes implement `chat(messages, model, temperature, json_mode) → text`.
- `claude_cli` adapter shells out to local `claude` CLI for zero-cost dev iteration.
- **Why:** unblocks every Tier-1 AI feature; lets cheap providers handle high-volume agents.
- **Acceptance:** `/api/chat` and `/api/research` work identically across all 6 backends; one env-var flips them.
- **Risks:** non-Claude providers don't have web-search tool; route web-search-required prompts to Claude only.

### F2. Vector DB foundation ✅ S
Local Chroma or Qdrant with one collection per concept (filings, concalls, news, broker reports).
- Embedding model: `BAAI/bge-large-en` (free, runs on CPU) or OpenAI `text-embedding-3-small` if budget allows.
- **Acceptance:** can `add(doc, metadata)`, `query(text, k=5, filter=...)`.
- **Why:** prerequisite for RAG (W2.1) and citation-grounded research.

---

## Wave 1 — AI feel (1–2 weeks total)

### W1.1. Multi-agent stock debate ⭐ ✅ L
Five specialist agents argue about a stock; moderator synthesises a verdict.

| Agent | Tool access | Reads |
|---|---|---|
| **Technician** | `pf.calculate_technical_indicators` | RSI, MACD, MAs, MFI, ADX |
| **Fundamentalist** | `pf._get_screener_fundamentals` + `kite.get_holdings` | PE, ROE, debt/equity, growth |
| **Macro hawk** | `macro_fetcher.get_all` + `fii_dii_fetcher` | VIX, USD/INR, FII flows, repo rate |
| **Sentiment** | `news_clusterer.cluster_news` + `sector_classifier.get_sector_sentiment` | News tilt, sector mood |
| **Options** | `options_fetcher.get_option_chain` | PCR, max-pain, IV-skew |

**Pipeline:**
1. User asks "should I buy RELIANCE?"
2. Moderator dispatches 5 agents in parallel via LLM router; each returns `{stance, confidence, reasoning, evidence: [tool calls]}`.
3. Moderator runs a 2nd-pass *adversarial review* (each agent critiques the others' weakest claim).
4. Final ranked verdict: BUY/SELL/HOLD with confidence band + dissent log.

- **API:** `POST /api/debate` `{symbol}` → `{verdict, confidence, agents:[...], dissent:[...]}`
- **UI:** new `Debate` section with collapsible cards per agent + dissent panel.
- **Acceptance:** verdict produced in <20s with citations to specific tool outputs.
- **Cost knob:** technician/fundamentalist run on cheap models; macro/sentiment on smart models. ~$0.04/debate on Claude, ~$0.005 on DeepSeek.
- **Risks:** prompt-injection from news content; sandbox tool-call results.

### W1.2. Regime classifier ✅ S
HMM (or change-point) over `(nifty_returns, vix_level, breadth_a/d, sector_dispersion)` → labels current market as `Trending Bull / Range / Volatile / Crash / Recovery`.

- Library: `hmmlearn` (5 states) or `ruptures` (changepoint detection).
- **API:** `GET /api/regime` → `{state, confidence, days_in_state, transition_probs}`
- **Hook:** all RL/strategy signals gated by regime — e.g. trend-following silenced in `Range`, mean-reversion silenced in `Trending Bull`.
- **Acceptance:** regime labels match human-eyeball historical chart.
- **Risks:** small N for Indian regimes; supplement with rules.

### W1.3. Walk-forward backtester ✅ M
Replace `core/backtester.py` single-shot with anchored-walk-forward.

- Train on `[t-W, t]`, test on `[t, t+H]`, advance `t += H`, repeat.
- Bootstrap Sharpe distribution from 1000 in-sample re-samples.
- Realistic costs: bid/ask spread from Kite L1 + STT + brokerage + slippage curve.
- **API:** `POST /api/backtest/walkforward` returns CDF of out-of-sample Sharpe + drawdown distribution.
- **Acceptance:** in-sample vs out-of-sample Sharpe gap reported (overfit detector).
- **Why:** current backtester reports survivor-biased optimistic numbers.

---

## Wave 2 — Information edge (2–3 weeks)

### W2.1. RAG over filings + concalls ✅ L
Index BSE/NSE filings + concall transcripts + broker initiation notes in vector DB. AI Research now cites actual sentences.

- **Sources:**
  - BSE annual reports & quarterly results (PDF; chunk + embed)
  - NSE corporate announcements feed (already integrated for #8 — point ingestion to vector DB)
  - Concall transcripts (Trendlyne / Bloomberg Quint / company IR pages)
  - SEBI insider trading disclosures
- **Pipeline:** nightly cron → fetch new docs → embed → upsert to Chroma → `/api/research/{sym}` queries top-k → injects as context to LLM.
- **API:** `POST /api/research/{sym}/grounded` — returns answer with cited sentences + source URLs.
- **Acceptance:** answer to "is RELIANCE's debt position improving?" cites at least 2 specific quarters from concalls.
- **Storage:** ~5GB for BSE-500 5y history.

### W2.2. Event-driven trader ✅ M
Poll NSE `corporate-announcements` API every 60s. Classify each announcement (results, dividend, insider, M&A, governance, profit warning) via LLM. If material → toast + draft order.

- **Pipeline:** NSE feed → LLM classify → severity score (0-100) → if >60, push WS event to UI + log to `events` collection.
- **API:** `GET /api/events?since=...&min_severity=60`
- **UI:** sliding event drawer on right edge, click to expand; severe events flash the chat-toggle button.
- **Acceptance:** detects e.g. an insider buy ≥₹1 Cr within 2 minutes of NSE publication.
- **Risks:** LLM cost on every announcement — pre-filter on `bm_purpose` keywords first.

### W2.3. Alternative data (India-flavored) ✅ M
- **r/IndianStockMarket + IndiaInvestments** weekly retail sentiment scrape — public top.json, no auth
- **ValuePickr** Discourse `/top/weekly.json` — thread velocity + ticker mentions
- **SIAM monthly auto sales** → auto-sector signal (maintained table; YoY + 3m avg)
- **GST collections** (monthly) → broad economic activity proxy (maintained table; YoY + 3m avg)
- **IIP / CPI** (MOSPI/RBI bulletin) → macro stance (Stable/Inflationary/Disinflationary/Stagflation)
- **Google Trends** for top tickers → retail-interest spikes (optional pytrends, safe-degrade)
- Persisted to Mongo `alt_signals` (TTL 7d, keyed `{source}:{key}`); 6h warming loop on startup.
- **API:** `GET /api/altdata` → flat `{source: {key: {value, unit, confidence, as_of}}}` + `_meta`.
- **Tests:** 60 unit + integration (full suite green, no live HTTP).

---

## Wave 3 — Quant-grade (3–4 weeks)

### W3.1. Forecasting models layer ✅ L
Train + serve uncertainty-aware forecasters alongside RL.

- **PatchTST** (in-house ~150 LOC torch implementation): patchify → embed → 2-layer transformer encoder → linear head; bootstrap CI from in-sample residuals.
- **GARCH(1,1)** (`arch` package): conditional variance over horizon; price-level bands via log-normal projection.
- **Holt-Winters trend** (`statsmodels` `ExponentialSmoothing`): replaced original NeuralProphet pick (NeuralProphet has no Python 3.14 build); same role in ensemble; PI bands from residual stdev.
- **Ensemble:** point = 0.6·PatchTST + 0.4·Trend; bands re-anchored from GARCH; regime-conditional bull/bear via heuristic drift; per-component fallback flags drive re-weighting.
- **Cache:** Mongo `forecast_cache`, TTL 24h daily / 5min intraday, keyed `{sym}:{horizon}:{model}:{interval}`.
- **Evaluator:** `evaluate_pi_coverage()` — anchored walk-forward harness; populates `calibration.pi80_oos_coverage` (W3.1 acceptance gate at ≥0.75).
- **API:** `GET /api/forecast/{sym}?horizon=N&model=ensemble` → full ForecastResult JSON.
- **UI:** ribbon chart on stock detail with confidence cones. *(Frontend wave; deferred.)*
- **Tests:** 57 (model unit + ensemble + cache + evaluator + API stand-in + live wiring).

### W3.2. Conformal prediction + meta-stacking ✅ M
Wrap RL + forecaster + sentiment in a meta-learner that emits *calibrated* probability bands.

- **Split conformal wrapper** (`marketmind/ml/forecast/conformal.py`): wraps any inner Forecaster, splits 80/20 train/calibration, computes nonconformity residuals, returns calibrated 90/95 PIs with marginal-coverage guarantee. Exposes `recalibrate(df_new)` for weekly refresh under regime shift.
- **Meta-stacker** (`meta_stacker.py`): scikit-learn multinomial LogisticRegression over a 9-dim feature vector (forecast_return, forecast_vol, rl_signal, regime one-hot, sentiment_tilt) → softmax(BUY/SELL/HOLD). Bootstrapped with deterministic synthetic-rule defaults; offline retrain via `fit_from_history(X, y)`.
- **API:** `GET /api/signal/{sym}/calibrated?horizon=N` → `{p_buy, p_sell, p_hold, expected_return, return_95ci, forecast, features}` — wires conformal-wrapped ensemble + RL + regime + sector sentiment through the meta-stacker.
- **Tests:** 24 new (7 conformal coverage + 11 meta-stacker + 6 API stand-in) plus live wiring assertion.

### W3.3. Options strategy builder ✅ L
Move from chain display → strategy assembly.

- **Strategies:** covered call, cash-secured put, bull call spread, bear put spread, straddle, strangle, iron condor, calendar, ratio spread (9 templates).
- **Pricing core** (`marketmind/ml/options/pricing.py`): Black-Scholes call/put + 5 Greeks (Δ Γ Θ ν ρ); IV converted from NSE-percent to decimal at the boundary; degenerate inputs (T=0, σ=0) collapse to intrinsic.
- **Strategy templates** (`strategies.py`): `build_default_legs(name, chain, expiry_days, lots, lot_size)` seeds legs from option-chain ATM ± k strikes; premium pulled from `ltp`, IV from `iv` field; ratio spread is 1×2 (long 1 ATM, short 2 OTM).
- **Analytics** (`builder.py`): vectorised numpy payoff curve (200 points, ±30% range), max P/L, linear-interpolated break-evens, signed-additive net Greeks, theoretical BS value, conservative margin proxy (`|max_loss| × 1.2`, flagged `margin_is_proxy: true`).
- **API:** `POST /api/options/strategy` `{symbol, strategy, expiry_days?, lots?, lot_size?, legs?, underlying?, back_expiry_days?}` → full analytics; calendar_spread requires `back_expiry_days`; markets-closed chain returns `{unavailable: true}` short-circuit.
- **Limitations documented:** BS assumes European exercise (exact for NIFTY/BANKNIFTY/FINNIFTY index options; close approximation for equities); IV-rank hint deferred to a dedicated wave; UI deferred (frontend wave); SPAN margin deferred (regulator-driven).
- **Tests:** 58 (15 pricing + 18 strategies + 17 builder + 7 API + 1 wiring); full suite 201 green.

---

## Wave 4 — Indian moat (2–3 weeks)

### W4.1. Tax-aware rebalancer ⏳ M
Optimise rebalance to minimise India-specific tax drag.

- **Indian tax regime (FY26):** STCG 15% (≤1y), LTCG 12.5% above ₹1.25L exemption (>1y).
- Read holdings + cost basis + acquisition dates from Kite.
- Solver: minimise `(realized_gain × tax_rate) + tracking_error_to_target`.
- Suggest tax-loss harvesting against current realized gains.
- **API:** `POST /api/portfolio/rebalance/tax-optimal` → `{trades:[...], tax_saved_inr, tracking_error}`.
- **Acceptance:** in a worked test case beats naive rebalance by ≥3% on after-tax return.
- **Why:** completely undifferentiated by US tools; uniquely valuable here.

### W4.2. Multi-asset panel ⏳ M
Beyond NSE equities.

- **MCX commodities** (gold/silver/crude/zinc) via Kite (`MCX:GOLDM23APRFUT`).
- **USD/INR + EUR/INR** forex via NSE currency derivatives.
- **INR-paired crypto** correlation panel (CoinGecko free tier).
- New `Macro / Cross-asset` section showing rolling 30/90-day correlations.

---

## Wave 5 — Frontier (research-grade)

### W5.1. Causal Bayesian network ⏳ XL
Nodes: Repo rate, USD/INR, crude oil, FII flows, GDP growth, sector indices. Learn structure from 10y data via PC-algorithm (`pgmpy`); ID effects via `dowhy`.

- **API:** `POST /api/causal/whatif` `{intervention: {repo_rate: -0.5}, target: "BANKNIFTY"}` → counterfactual estimate + confidence.
- **Acceptance:** explain *why* recommendations exist via causal paths.

### W5.2. Hierarchical Risk Parity + Black-Litterman ⏳ L
- HRP allocates by clustering correlation tree (no inverse covariance — robust to noise).
- Black-Litterman lets users inject *views* ("I think TCS will outperform sector by 3%") and blend with market priors.
- Replace mean-variance default in optimiser.

### W5.3. SEBI compliance layer ⏳ M
- Insider trading window enforcement (closed periods around results — block self-trades on tracked tickers).
- Position limits per stock (Kite enforces but UI should warn pre-trade).
- Algo PDA: pre-trade self-checks + post-trade audit log per SEBI regulation.
- **Why:** anyone running this for a fund needs this. Cheap insurance.

---

## Sequencing & dependencies

```
F1 (LLM router) ─┬─→ W1.1 (multi-agent debate)
                 ├─→ W2.2 (event classifier)
                 └─→ W3.2 (meta-stacker; LLM as feature)
F2 (vector DB) ──┬─→ W2.1 (RAG)
                 └─→ W2.3 (alt-data search)
W1.2 (regime) ─────→ gates everything in W3
W1.3 (walk-fwd) ───→ honest numbers feed W3.2 calibration
```

Recommended order: **F1 → F2 → W1.2 → W1.3 → W1.1 → W2.1 → W2.2 → W2.3 → W3 → W4 → W5.**

Time to "noticeably more intelligent": end of Wave 1 (~10 days). Time to "institutional-grade": end of Wave 3 (~6 weeks).

---

## Tracking columns (when work begins)

| ID | Title | Owner | State | Started | Done | PR |
|---|---|---|---|---|---|---|
| F1 | LLM router | – | ✅ | 2026-04-27 | 2026-04-27 | – |
| F2 | Vector DB | – | ✅ | 2026-04-27 | 2026-04-27 | – |
| W1.1 | Multi-agent debate | – | ✅ | 2026-04-27 | 2026-04-27 | – |
| W1.2 | Regime classifier | – | ✅ | 2026-04-27 | 2026-04-27 | – |
| W1.3 | Walk-forward backtest | – | ✅ | 2026-04-27 | 2026-04-27 | – |
| W2.1 | Filings RAG | – | ✅ | 2026-04-27 | 2026-04-27 | – |
| W2.2 | Event-driven layer | – | ✅ | 2026-04-27 | 2026-04-27 | – |
| W2.3 | Alt-data | – | ✅ | 2026-04-27 | 2026-04-27 | – |
| W3.1 | Forecasting models | – | ✅ | 2026-04-27 | 2026-04-27 | – |
| W3.2 | Conformal stacking | – | ✅ | 2026-04-27 | 2026-04-27 | – |
| W3.3 | Options strategies | – | ✅ | 2026-04-27 | 2026-04-27 | – |
| W4.1 | Tax rebalancer | – | ⏳ | – | – | – |
| W4.2 | Multi-asset | – | ⏳ | – | – | – |
| W5.1 | Causal Bayes net | – | ⏳ | – | – | – |
| W5.2 | HRP + B-L | – | ⏳ | – | – | – |
| W5.3 | SEBI compliance | – | ⏳ | – | – | – |

---

## What's intentionally NOT in scope

- **Twitter/X sentiment** — API closed; signal mostly gone in India.
- **Satellite imagery** — fascinating but no liquid-stock alpha.
- **Pure ML black-box price prediction** without uncertainty — we'll lose money.
- **High-frequency / sub-second strategies** — Kite tick rate insufficient; needs colocation.

---

## Archive — Completed repair plan (2026-04-27)

All 15 audit items completed in one day. Final regression all green:

```
/api/orders                  ✅ 200  3,430 B  real Kite orders + ISO timestamps
/api/screener?limit=5        ✅ 200  1,067 B  parallelized, real fundamentals
/api/bulk-deals              ✅ 200 34,782 B  140 bulk + 2 block (snapshot-largedeal)
/api/earnings-calendar       ✅ 200 55,495 B  100 board meetings (corporate-board-meetings)
/api/sectors/correlations    ✅ 200  2,198 B  9-sector correlation matrix
/api/rl/signals              ✅ 200    825 B  5 BUY signals via multiframe fallback
/api/risk/stock/RELIANCE     ✅ 200    380 B  real VaR (-₹2,172 95%, -₹3,788 99%)
/api/stocks/RELIANCE         ✅ 200  5,673 B  PE 22.9, ROE 9.25%, mcap ₹18.48L Cr
/api/market/heatmap          ✅ 200 59,436 B  499 stocks (was 177 fallback)
/api/market/indices          ✅ 200    455 B  Nifty 500 + 3 others
```

**Keystone fix:** removed `br` from NSE Accept-Encoding header (one character) — `requests` can't decode brotli without the `brotli` package, NSE returned brotli-encoded JSON, the JSON-shape guard rejected it as garbage. This single change unblocked the heatmap, sector correlations, options chain (when markets open), bulk deals, earnings calendar, fundamentals, and historical bars.

**Bonus delivered:**
- Modern editorial-dark UI redesign (Instrument Serif + Geist + JetBrains Mono, saffron accent, glass cards)
- Hero ticker strip + sparklines in metric cards
- Nifty 50 → Nifty 500 swap across every screen
- Knowledge graph: 1,047 nodes, 53 communities (`graphify-out/graph.html`)
- `CLAUDE.md` graph-first workflow; `.gitignore` with secrets exclusion
- env-var override for all secrets
- PySide6 desktop UI quarantined to `_legacy/` (12 files, ~6,500 lines)
