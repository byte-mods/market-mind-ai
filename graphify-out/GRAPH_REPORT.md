# Graph Report - .  (2026-04-28)

## Corpus Check
- 147 files · ~134,402 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3050 nodes · 6373 edges · 83 communities detected
- Extraction: 68% EXTRACTED · 32% INFERRED · 0% AMBIGUOUS · INFERRED: 2065 edges (avg confidence: 0.66)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]

## God Nodes (most connected - your core abstractions)
1. `TaxLot` - 102 edges
2. `EnsembleForecaster` - 91 edges
3. `SplitConformalWrapper` - 81 edges
4. `ColorPalette` - 77 edges
5. `MetricCard` - 77 edges
6. `AppController` - 77 edges
7. `SectorClassifier` - 76 edges
8. `ForecastResult` - 75 edges
9. `_run()` - 63 edges
10. `KiteClient` - 63 edges

## Surprising Connections (you probably didn't know these)
- `/api/sectors/{sector}/recommendations endpoint` --implements--> `SectorsView`  [INFERRED]
  memory/project_marketmind.md → _legacy/ui/sectors.py
- `AppController (singleton)` --semantically_similar_to--> `God node: AppController (245 edges)`  [INFERRED] [semantically similar]
  README.md → graphify-out/GRAPH_REPORT.md
- `KiteClient (REST + WS)` --semantically_similar_to--> `God node: KiteClient (146 edges)`  [INFERRED] [semantically similar]
  README.md → graphify-out/GRAPH_REPORT.md
- `Split-conformal prediction wrapper` --semantically_similar_to--> `God node: SplitConformalWrapper (81 edges)`  [INFERRED] [semantically similar]
  README.md → graphify-out/GRAPH_REPORT.md
- `Backtester (10 strategies)` --references--> `Data sources: NSE, Screener, Kite, Google News`  [EXTRACTED]
  marketmind/core/backtester.py → memory/project_marketmind.md

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (130): AnalysisView, AnalysisWorker, _panel(), MarketMind AI - Professional Analysis View Wall-Street-grade market analysis: se, Professional McKinsey/Wall-Street-style analysis view.     - Sector prediction p, Runs heavy analysis in background, emits results, _table_style(), BaseHTTPRequestHandler (+122 more)

### Community 1 - "Community 1"
Cohesion: 0.02
Nodes (250): BaseModel, _finite_sample_quantile(), SplitConformalWrapper, _detect_sector(), Best-effort sector detection from news + heuristics., EnsembleForecaster, get_ensemble_forecaster(), _is_fallback() (+242 more)

### Community 2 - "Community 2"
Cohesion: 0.02
Nodes (131): BulkDealsFetcher, get_bulk_deals_fetcher(), MarketMind AI - Bulk & Block Deals Tracker Tracks large institutional trades fro, Get bulk + block deals combined with analytics., Fetch the combined bulk/block/short deals snapshot. Cached 15 min., _doc_to_result(), ForecastCache, get_forecast_cache() (+123 more)

### Community 3 - "Community 3"
Cohesion: 0.02
Nodes (148): ABC, AltDataAggregator, _default_sources(), get_aggregator(), Alt-data aggregator: parallel fan-out + Mongo persist + flat-dict surface.  Arch, Trampoline for the executor — keeps the closure simple., Process-wide singleton. Pass mongo_col on first call only., _run_source() (+140 more)

### Community 4 - "Community 4"
Cohesion: 0.03
Nodes (174): classify_holding_period(), compute_tax(), MarketMind AI — Tax Engine (W4.1)  Pure-function FY26 Indian listed-equity capit, Materialise a single lot sale into a RealizedGain record.      Quantity must be, Aggregate `gains` and apply FY26 STCG/LTCG rates.      `ltcg_used_inr` lets the, A single (partial or whole) lot disposition.      `gain_inr` is signed: negative, Aggregated tax summary for a set of realised gains.      `*_realized_inr` are ne, Return (holding_period_days, gain_type).      Boundary: holding period of *exact (+166 more)

### Community 5 - "Community 5"
Cohesion: 0.01
Nodes (174): fetch_news_via_claude (CLI+SDK), get_recent_news_from_mongo, _NEWS_PROMPT (Indian market query), run_claude_news_pipeline, store_news_in_mongo, Community: AppController + News Pipeline, Community: Audit Log + Order Hooks, Community: Compliance API Tests (+166 more)

### Community 6 - "Community 6"
Cohesion: 0.02
Nodes (78): AppController, MarketMind AI - App Controller Main controller for managing data and business lo, Compute insider-window status for a symbol (regardless of designation)., Replace the designated-symbols list. Persists to Mongo + invalidates cache., Get all orders for the day, Get all executed trades, Place an order via Kite. Writes one compliance-audit row per call         (sourc, Start automated buyer loop (+70 more)

### Community 7 - "Community 7"
Cohesion: 0.03
Nodes (107): _Band, Forecaster, ForecastResult, make_band(), Forecasting primitives: ``ForecastResult`` record + ``Forecaster`` protocol.  Sc, Structural protocol every concrete forecaster honours.      Lifecycle:         f, Point + 80/95 prediction interval. All fields in price units., One forecast record.      Fields:         symbol:            "RELIANCE" (+99 more)

### Community 8 - "Community 8"
Cohesion: 0.03
Nodes (85): Exception, KiteClient, KiteConfig, Full Kite Connect API client wrapper.      Handles:     - Authentication & sessi, Initialize KiteConnect REST client, Return Kite login URL for browser-based authentication, Exchange request_token for access_token. Call after user logs in., Log out and clear access token (+77 more)

### Community 9 - "Community 9"
Cohesion: 0.04
Nodes (76): AuditLogEntry, AuditLogStore, Compliance audit-log store (W5.3 T1).  Append-only Mongo-backed record of every, Return entries newest-first. ``limit`` capped to ``MAX_QUERY_LIMIT``., One audit row. Frozen so callers can't mutate after handing in., Thin Mongo wrapper. Pass ``mongo_col=None`` to disable persistence., Persist one entry; returns the assigned ``_id`` or None on no-op/error., AuditLogStore (+68 more)

### Community 10 - "Community 10"
Cohesion: 0.03
Nodes (68): Rationale: ATR-based SL/TP adapts to volatility, Backtester._load_data (Kite -> NSE fallback), Backtester._precompute (shared indicators), Backtester.run, Backtester._simulate (ATR SL/TP + trailing), _fetch_from_kite, KiteCandles.get_candles, KiteCandles.get_candles_df (+60 more)

### Community 11 - "Community 11"
Cohesion: 0.04
Nodes (42): Backtester (10 strategies), get_backtester singleton, MarketMind AI - Strategy Backtester (Professional Grade) =======================, ADX Trend-Following: ADX > 22 + MA10 > MA20 > MA50 + RSI 45-65.         One of t, RSI Pullback in Uptrend: RSI dips to 40-48 while price above MA50.         High, MACD Histogram reversal: histogram makes higher low (bullish divergence), Volume-confirmed Donchian breakout (Turtle system enhanced).         Breakout ab, Triple MA Ribbon: 10/20/50 all aligned upward + RSI > 50 + volume.         When (+34 more)

### Community 12 - "Community 12"
Cohesion: 0.04
Nodes (41): CorrelationAnalyzer, get_correlation_analyzer(), MarketMind AI - Correlations Analysis Module Computes sector and stock correlati, Compute beta of a stock relative to market         Beta = Cov(stock, market) / V, Analyzes correlations between sectors and stocks, Compute correlation matrix from returns DataFrame, Get description of correlation strength, Get description of correlation direction (+33 more)

### Community 13 - "Community 13"
Cohesion: 0.05
Nodes (46): _accuracy(), _adx_approx(), _atr(), compute_features (rich indicator pipeline), _ema(), from_dict(), get_combined_signal (RL+ML+Confluence), _LogisticReg (numpy LR) (+38 more)

### Community 14 - "Community 14"
Cohesion: 0.06
Nodes (61): analyse(), _find_break_evens(), _intrinsic(), _leg_payoff(), _net_greeks(), _net_premium(), Strategy analytics — legs → payoff curve, max P/L, break-evens, net Greeks.  Pur, Net premium paid (positive) or received (negative) at entry. (+53 more)

### Community 15 - "Community 15"
Cohesion: 0.05
Nodes (30): fake_mongo_alt_signals(), fake_mongo_col(), FakeMongoCol, frozen_now(), Shared pytest fixtures for the MarketMind test suite.  Design notes: - `fake_mon, `alt_signals` collection — keyed by composite `source:key`., Pin `datetime.now()` and `datetime.utcnow()` to a fixed point.      Useful for t, In-memory dict-backed pymongo.Collection stand-in.      Indexes documents by the (+22 more)

### Community 16 - "Community 16"
Cohesion: 0.06
Nodes (26): _category_counts(), _chunk(), _classify(), FilingsIngester, _normalise_date(), MarketMind AI - Filings & Concall Ingester (W2.1)  Pulls NSE corporate-announcem, Pull raw rows from NSE corporate-announcements for the symbol., Fetch, chunk, embed. Idempotent thanks to deterministic IDs. (+18 more)

### Community 17 - "Community 17"
Cohesion: 0.06
Nodes (29): generate_research_report(), get_assistant(), _get_client(), _load_api_key(), MarketAssistant, MarketMind AI - Claude-Powered Research Engine Generates institutional-grade inv, Streaming conversational assistant with market context.     Uses Claude API with, Send a message and get response (non-streaming). (+21 more)

### Community 18 - "Community 18"
Cohesion: 0.09
Nodes (37): _LotPayload, _make_app(), _ok_envelope(), W4.1 verification: POST /api/portfolio/rebalance/tax-optimal wire contract.  Sta, 200 + full envelope + all canonical keys present., 200 + full envelope + all canonical keys present., Weight-sum drift must surface in `error` (envelope, not 500)., Weight-sum drift must surface in `error` (envelope, not 500). (+29 more)

### Community 19 - "Community 19"
Cohesion: 0.08
Nodes (29): build_features(), feature_matrix(), _macd_hist(), OHLCV → multivariate feature tensor for forecasters.  Input  (pandas DataFrame):, Return df with engineered columns appended.      Raises ValueError if required O, Drop NaN rows from a feature-built DF and return ``(N, F)`` numpy array.      Us, Wilder's RSI — exponentially smoothed gains/losses., MACD histogram, normalised by close to keep cross-symbol comparable. (+21 more)

### Community 20 - "Community 20"
Cohesion: 0.08
Nodes (18): get_optimizer singleton, PortfolioOptimizer, MarketMind AI - Portfolio Optimizer Markowitz mean-variance optimization, effici, Generate efficient frontier points for risk-return chart., Compare all four strategies side by side., Implements:     - Markowitz Mean-Variance (maximum Sharpe, minimum variance), Equal risk contribution (risk parity)., Return (annual_return, annual_vol, sharpe). (+10 more)

### Community 21 - "Community 21"
Cohesion: 0.13
Nodes (31): build_default_legs(), _ladder(), _leg(), list_strategies(), Named option strategies → leg lists, optionally seeded from an option chain.  A, Return sorted unique strikes from chain rows., Estimate strike step (difference between adjacent strikes near ATM)., Construct a default leg list for `name` from `chain`.      `chain` is the dict r (+23 more)

### Community 22 - "Community 22"
Cohesion: 0.12
Nodes (26): _build_app(), DesignatedSymbolsRequest, _ok_audit_envelope(), _ok_designated_envelope(), _ok_insider_envelope(), _ok_pretrade_envelope(), PretradeRequest, W5.3 T5 verification: SEBI compliance route mirror tests.  Builds a tiny FastAPI (+18 more)

### Community 23 - "Community 23"
Cohesion: 0.12
Nodes (26): feature_vector(), get_meta_stacker(), MetaStacker, Meta-stacker: feature dict → calibrated softmax(P_buy, P_sell, P_hold).  Inputs, Deterministic feature/label corpus used to bootstrap default weights.      Embed, Return the singleton meta-stacker. Bootstrapped on first call from     the deter, Test hook — wipes the singleton so each test gets a fresh fit if needed., Map the feature dict into a fixed-shape numpy vector.      Missing keys → 0.0. U (+18 more)

### Community 24 - "Community 24"
Cohesion: 0.14
Nodes (24): check_position_limits(), _holding_value(), PositionLimitStatus, Position-limit checks (W5.3 T3).  Concentration-only this section. Computes post, Outcome of the concentration check., Best-effort current value: last_price × quantity., Decide whether the proposed trade exceeds concentration limits.      Errors: zer, _h() (+16 more)

### Community 25 - "Community 25"
Cohesion: 0.11
Nodes (18): _ann(), W5.3 T2 verification: insider-window pure compute over NSE announcements., Convenience: build a minimal announcement row., Today == quarter-end day → that quarter-end is the most-recent     boundary; wit, Today = quarter-end + 1 day → most-recent quarter-end is still     the same cale, Same wall-clock instant produces different decisions depending on     whether ``, test_insider_window_accepts_datetime_today(), test_insider_window_closed_when_awaiting_current_quarter_results() (+10 more)

### Community 26 - "Community 26"
Cohesion: 0.1
Nodes (12): Predict direction/magnitude for all sectors.         Returns dict: sector -> pre, Predict a single sector, get_sentiment_analyzer(), MarketMind AI - Sentiment Analyzer Module Analyzes sentiment of financial news u, Analyze sentiment of text         Returns dict with score, label, and confidence, Financial news sentiment analyzer     Uses lexicon-based approach with financial, Quick classification of headline, Analyze a batch of news items (+4 more)

### Community 27 - "Community 27"
Cohesion: 0.1
Nodes (13): get_portfolio_simulator(), RLEnhancedEnsemble.make_investment_decision, PortfolioSimulator (Monte Carlo), Simulate portfolio with given allocations to sectors, Portfolio simulation engine using Monte Carlo methods, Combines RL agent actions with other signals for ensemble decisions, Make complete investment decision combining all signals, Get signal from technical indicators (+5 more)

### Community 28 - "Community 28"
Cohesion: 0.1
Nodes (13): VaR / CVaR calculator, Singleton getter, PortfolioSimulator (Monte Carlo), MarketMind AI - Portfolio Simulator Monte Carlo portfolio simulations, Generate bull/base/bear scenario analysis, Calculate Value at Risk and Conditional VaR, Monte Carlo simulation engine for portfolio analysis, Calculate summary statistics (+5 more)

### Community 29 - "Community 29"
Cohesion: 0.16
Nodes (18): _ok_envelope(), W4.1 deferred MINOR: real-server route test.  Mirror tests in `test_api_rebalanc, Malformed `as_of` → real route returns 400 + flat envelope (NOT a     Pydantic 4, Missing required `target_weights` → Pydantic 422 from FastAPI before     the rou, `lots_override` body field is parsed by Pydantic and forwarded into     the cont, A valid ISO `as_of` is parsed to a `datetime.date` instance before     the contr, Controller returns `authenticated=False` → real route echoes the     envelope th, Records call args, returns a configured envelope. Mirrors the keyword     signat (+10 more)

### Community 30 - "Community 30"
Cohesion: 0.12
Nodes (8): SQLite Database, MarketMind AI - Database Module SQLite database for storing news and cache, Store stock price data, Get historical stock prices, Store RL trading signal, Initialize database tables, Get recent RL signals, Store portfolio simulation result

### Community 31 - "Community 31"
Cohesion: 0.24
Nodes (16): _chain(), _Leg, _make_app(), W3.3 verification: POST /api/options/strategy wire contract.  Stand-in pattern (, No NaN/Inf leaks through — _sanitize must scrub them., Mirror server.py _sanitize for NaN/Inf scrubbing., _Req, _sanitize() (+8 more)

### Community 32 - "Community 32"
Cohesion: 0.17
Nodes (10): W5.3 T5 real-server route tests.  Boots the actual FastAPI app from `server.py`, Missing required body field → 422, controller not invoked., All four W5.3 routes are present on the actual FastAPI app., _StubController, test_real_server_audit_log_wires_to_controller(), test_real_server_compliance_routes_registered_on_live_app(), test_real_server_designated_symbols_wires_to_controller(), test_real_server_insider_window_wires_to_controller() (+2 more)

### Community 33 - "Community 33"
Cohesion: 0.14
Nodes (16): Auto-Query Rule (run /graphify before code change), God Nodes: AppController, SectorClassifier, KiteClient, TradingEnvironment, PriceFetcher, Graph-First Development rule, Hard Rule: never bypass NSE cache (10-min Mongo TTL), Hard Rule: Indian market context (NSE/BSE, IST, ₹), Hard Rule: never commit secrets (local.json), Hard Rule: never re-add yfinance (404 on Indian symbols), Legacy: PySide6 desktop UI quarantined (+8 more)

### Community 34 - "Community 34"
Cohesion: 0.43
Nodes (6): cluster_news (TF-IDF cosine clustering), _cosine(), MarketMind AI - News Clusterer Groups similar headlines to remove duplicates and, Cluster news items by headline similarity.     Returns one representative item p, TF-IDF vector builder, _tokenize()

### Community 35 - "Community 35"
Cohesion: 0.33
Nodes (1): T11: e2e wiring check.  Imports the real ``server`` module and asserts that W2.3

### Community 36 - "Community 36"
Cohesion: 0.5
Nodes (5): PortfolioOptimizer.efficient_frontier, _max_sharpe SLSQP solver, _min_variance solver, PortfolioOptimizer.optimize, _risk_parity solver

### Community 37 - "Community 37"
Cohesion: 0.5
Nodes (4): Fetch news across all sectors, SECTOR_QUERIES dict (12 sectors), Keyword sector classifier (legacy), SECTOR_KEYWORDS classification dict

### Community 38 - "Community 38"
Cohesion: 0.5
Nodes (4): SectorPredictor._predict_sector composite scoring, SentimentAnalyzer.aggregate_sentiment, SentimentAnalyzer.analyze, Financial positive/negative lexicons

### Community 39 - "Community 39"
Cohesion: 0.67
Nodes (3): SECTOR_BETAS table, STRESS_SCENARIOS dict (India-relevant), RiskEngine.stress_test

### Community 40 - "Community 40"
Cohesion: 0.67
Nodes (3): ACTIONS BUY/HOLD/SELL/HEDGE/REBALANCE, TradingEnvironment._get_state (50-d feature vec), TradingEnvironment.step (5 actions)

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (2): InvestmentAdvisor.analyze_stock entry/exit/SL, FundamentalAnalyzer.score_stock

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (2): news SQLite schema, Database.store_news

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (2): rl_signals SQLite schema, Database.store_rl_signal

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (2): W3.3 Options strategy builder, Black-Scholes options pricing + Greeks

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (0): 

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (0): 

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (0): 

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (0): 

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Correlated sector simulation (Cholesky)

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): MarketMind memory index

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): RiskEngine.stock_var

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): RiskEngine.portfolio_var

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): WebSocket /ws live updates + alerts

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): torch>=2.0.0

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): scikit-learn dependency

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): compute_beta (Cov/Var)

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): compute_sector_correlations

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Fetch the latest signals. Must never raise — decorate with safe_fetch.

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (0): 

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (0): 

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (0): 

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (0): 

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (0): 

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (0): 

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Effort Scale (XS/S/M/L/XL)

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (0): 

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): pandas>=2.1.0

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): numpy>=1.25.0

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): scikit-learn>=1.3.0

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): beautifulsoup4>=4.12.0

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): requests>=2.31.0

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): lxml>=4.9.0

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (1): sqlalchemy>=2.0.0

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): httpx>=0.25.0

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): ta>=0.10.0

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): pydantic>=2.0.0

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): python-multipart>=0.0.9

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (1): aiofiles>=23.2.1

### Community 79 - "Community 79"
Cohesion: 1.0
Nodes (1): arch>=6.3 (W3.1 forecasting)

### Community 80 - "Community 80"
Cohesion: 1.0
Nodes (1): statsmodels>=0.14 (Holt-Winters)

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (1): pytest-asyncio>=0.23

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (1): requests-mock>=1.12

## Knowledge Gaps
- **693 isolated node(s):** `Wait for the server to start then open the browser.`, `MarketMind AI - Google News Fetcher Fetches financial news for all Indian market`, `Fetches financial news from Google News RSS for all sectors.`, `Fetch news for every sector; deduplicate by URL.`, `Fetch news for a single sector (with caching).` (+688 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 41`** (2 nodes): `InvestmentAdvisor.analyze_stock entry/exit/SL`, `FundamentalAnalyzer.score_stock`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (2 nodes): `news SQLite schema`, `Database.store_news`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (2 nodes): `rl_signals SQLite schema`, `Database.store_rl_signal`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (2 nodes): `W3.3 Options strategy builder`, `Black-Scholes options pricing + Greeks`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Correlated sector simulation (Cholesky)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `MarketMind memory index`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `RiskEngine.stock_var`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `RiskEngine.portfolio_var`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `WebSocket /ws live updates + alerts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `torch>=2.0.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `scikit-learn dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `compute_beta (Cov/Var)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `compute_sector_correlations`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Fetch the latest signals. Must never raise — decorate with safe_fetch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Effort Scale (XS/S/M/L/XL)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `pandas>=2.1.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `numpy>=1.25.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `scikit-learn>=1.3.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `beautifulsoup4>=4.12.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `requests>=2.31.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `lxml>=4.9.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `sqlalchemy>=2.0.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `httpx>=0.25.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `ta>=0.10.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `pydantic>=2.0.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `python-multipart>=0.0.9`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (1 nodes): `aiofiles>=23.2.1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 79`** (1 nodes): `arch>=6.3 (W3.1 forecasting)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 80`** (1 nodes): `statsmodels>=0.14 (Holt-Winters)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (1 nodes): `pytest-asyncio>=0.23`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `requests-mock>=1.12`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_get_nifty500_constituents()` connect `Community 1` to `Community 9`, `Community 2`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Why does `KiteClient` connect `Community 8` to `Community 9`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `Backtester (10 strategies)` connect `Community 11` to `Community 2`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 99 inferred relationships involving `TaxLot` (e.g. with `RealizedGain` and `CurrentHolding`) actually correct?**
  _`TaxLot` has 99 INFERRED edges - model-reasoned connections that need verification._
- **Are the 82 inferred relationships involving `EnsembleForecaster` (e.g. with `PatchTSTForecaster` and `TrendForecaster`) actually correct?**
  _`EnsembleForecaster` has 82 INFERRED edges - model-reasoned connections that need verification._
- **Are the 75 inferred relationships involving `SplitConformalWrapper` (e.g. with `ForecastResult` and `Forecaster`) actually correct?**
  _`SplitConformalWrapper` has 75 INFERRED edges - model-reasoned connections that need verification._
- **Are the 75 inferred relationships involving `ColorPalette` (e.g. with `_WorkerSignals` and `MainWindow`) actually correct?**
  _`ColorPalette` has 75 INFERRED edges - model-reasoned connections that need verification._