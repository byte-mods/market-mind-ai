# Graph Report - .  (2026-04-27)

## Corpus Check
- 0 files · ~99,999 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2119 nodes · 4838 edges · 55 communities detected
- Extraction: 62% EXTRACTED · 38% INFERRED · 0% AMBIGUOUS · INFERRED: 1822 edges (avg confidence: 0.64)
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

## God Nodes (most connected - your core abstractions)
1. `AppController` - 125 edges
2. `SectorClassifier` - 107 edges
3. `KiteClient` - 103 edges
4. `EnsembleForecaster` - 83 edges
5. `ColorPalette` - 77 edges
6. `MetricCard` - 77 edges
7. `ForecastResult` - 75 edges
8. `SplitConformalWrapper` - 73 edges
9. `PriceFetcher` - 62 edges
10. `_run()` - 57 edges

## Surprising Connections (you probably didn't know these)
- `Data sources: NSE, Screener, Kite, Google News` --references--> `Backtester`  [EXTRACTED]
  memory/project_marketmind.md → marketmind/core/backtester.py
- `uvicorn server:app launcher` --implements--> `FastAPI + uvicorn + vanilla JS SPA`  [INFERRED]
  main.py → memory/project_marketmind.md
- `C3 verification: GET /api/signal/{sym}/calibrated wire contract.  Stand-in patte` --uses--> `ForecastResult`  [INFERRED]
  tests/test_api_signal_calibrated.py → marketmind/ml/forecast/base.py
- `Build a minimal app whose calibrated route mirrors server.py logic.` --uses--> `ForecastResult`  [INFERRED]
  tests/test_api_signal_calibrated.py → marketmind/ml/forecast/base.py
- `80-row threshold must be enforced.` --uses--> `ForecastResult`  [INFERRED]
  tests/test_api_signal_calibrated.py → marketmind/ml/forecast/base.py

## Hyperedges (group relationships)
- **PPO training pipeline (rollout, GAE, update, eval)** — ppo_trainer_train_ppo_agent, ppo_trainer_actorcriticnet, ppo_trainer_rolloutbuffer, ppo_trainer__ppo_update, ppo_trainer__simulate_deterministic [EXTRACTED 1.00]
- **Sector keyword/query corpora used for news classification** — sector_classifier_sectors, google_news_fetcher_sector_queries, news_fetcher_sector_keywords [INFERRED 0.85]
- **Browser runtime: FastAPI server + AppController + WS broadcast** — server_app, server_controller, server_connectionmanager, server__background_loop, server_startup [EXTRACTED 1.00]
- **Kite-backed historical data pipeline (config -> client -> candles cache -> backtester)** — kite_client_kiteconfig, kite_client_kiteclient, kite_candles_kitecandles, backtester_backtester [EXTRACTED 0.90]
- **Auto buyer-loop flow (settings UI -> config -> AutoTrader -> bracket order)** — kite_settings_kitesettingsview, kite_client_kiteconfig, kite_client_autotrader, kite_client_place_bracket_trailing [EXTRACTED 0.90]
- **Portfolio analytics: optimization + risk + stress scenarios sharing returns/sector data** — portfolio_optimizer_portfoliooptimizer, risk_engine_riskengine, risk_engine_stress_scenarios, risk_engine_sector_betas [INFERRED 0.80]
- **End-to-end RL signal pipeline (price -> features -> state -> action -> persisted signal -> UI)** — price_fetcher_calculate_technical_indicators, app_controller_build_state_vector, app_controller_update_rl_signals, database_store_rl_signal, dashboard_update_rl_signals [EXTRACTED 0.90]
- **RL + ML + Confluence combined trading signal** — rl_trainer_predict_rl, rl_trainer_predict_ml, rl_trainer_score_confluence, rl_trainer_get_combined_signal [EXTRACTED 0.95]
- **News fetch -> sentiment annotation -> persistence flow** — claude_news_fetcher_pipeline, app_controller_fetch_news, sentiment_analyzer_analyze, database_store_news, dashboard_update_news [EXTRACTED 0.90]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (173): AppController, MarketMind AI - App Controller Main controller for managing data and business lo, Return a MongoDB collection or None if not connected., Return True if data is stale (>10 min) or not yet in MongoDB., Record a successful fetch timestamp in MongoDB., Upsert news items into MongoDB (deduplicate by URL)., Upsert watchlist price data into MongoDB., Write RL signals to MongoDB. (+165 more)

### Community 1 - "Community 1"
Cohesion: 0.02
Nodes (114): AnalysisView, AnalysisWorker, _panel(), MarketMind AI - Professional Analysis View Wall-Street-grade market analysis: se, Professional McKinsey/Wall-Street-style analysis view.     - Sector prediction p, Runs heavy analysis in background, emits results, _table_style(), BaseHTTPRequestHandler (+106 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (131): ABC, AltDataAggregator, _default_sources(), get_aggregator(), Alt-data aggregator: parallel fan-out + Mongo persist + flat-dict surface.  Arch, Trampoline for the executor — keeps the closure simple., Process-wide singleton. Pass mongo_col on first call only., _run_source() (+123 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (123): _Band, Forecaster, ForecastResult, make_band(), Structural protocol every concrete forecaster honours.      Lifecycle:         f, Point + 80/95 prediction interval. All fields in price units., One forecast record.      Fields:         symbol:            "RELIANCE", _finite_sample_quantile() (+115 more)

### Community 4 - "Community 4"
Cohesion: 0.03
Nodes (96): BulkDealsFetcher, get_bulk_deals_fetcher(), MarketMind AI - Bulk & Block Deals Tracker Tracks large institutional trades fro, Get bulk + block deals combined with analytics., Fetch the combined bulk/block/short deals snapshot. Cached 15 min., _doc_to_result(), ForecastCache, get_forecast_cache() (+88 more)

### Community 5 - "Community 5"
Cohesion: 0.03
Nodes (68): Rationale: ATR-based SL/TP adapts to volatility, Backtester._load_data (Kite -> NSE fallback), Backtester._precompute (shared indicators), Backtester.run, Backtester._simulate (ATR SL/TP + trailing), _fetch_from_kite, KiteCandles.get_candles, KiteCandles.get_candles_df (+60 more)

### Community 6 - "Community 6"
Cohesion: 0.04
Nodes (87): EventPoller, get_event_poller(), MarketMind AI - Event Poller (W2.2)  Async background task that polls the NSE co, Blocking NSE call — must be run inside an executor., One pass over the watchlist. Returns a tally summary., get_filings_ingester(), get_price_fetcher(), Get or create global price fetcher instance. (+79 more)

### Community 7 - "Community 7"
Cohesion: 0.04
Nodes (44): Backtester, get_backtester(), MarketMind AI - Strategy Backtester (Professional Grade) =======================, ADX Trend-Following: ADX > 22 + MA10 > MA20 > MA50 + RSI 45-65.         One of t, RSI Pullback in Uptrend: RSI dips to 40-48 while price above MA50.         High, MACD Histogram reversal: histogram makes higher low (bullish divergence), Volume-confirmed Donchian breakout (Turtle system enhanced).         Breakout ab, Triple MA Ribbon: 10/20/50 all aligned upward + RSI > 50 + volume.         When (+36 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (46): _accuracy(), _adx_approx(), _atr(), compute_features(), _ema(), from_dict(), get_combined_signal(), _LogisticReg (+38 more)

### Community 9 - "Community 9"
Cohesion: 0.05
Nodes (47): build_features(), feature_matrix(), _macd_hist(), OHLCV → multivariate feature tensor for forecasters.  Input  (pandas DataFrame):, Return df with engineered columns appended.      Raises ValueError if required O, Drop NaN rows from a feature-built DF and return ``(N, F)`` numpy array.      Us, Wilder's RSI — exponentially smoothed gains/losses., MACD histogram, normalised by close to keep cross-symbol comparable. (+39 more)

### Community 10 - "Community 10"
Cohesion: 0.04
Nodes (38): CorrelationAnalyzer, get_correlation_analyzer(), MarketMind AI - Correlations Analysis Module Computes sector and stock correlati, Compute beta of a stock relative to market         Beta = Cov(stock, market) / V, Analyzes correlations between sectors and stocks, Compute correlation matrix from returns DataFrame, Get description of correlation strength, Get description of correlation direction (+30 more)

### Community 11 - "Community 11"
Cohesion: 0.05
Nodes (40): generate_research_report(), get_assistant(), _get_client(), _load_api_key(), MarketAssistant, MarketMind AI - Claude-Powered Research Engine Generates institutional-grade inv, Streaming conversational assistant with market context.     Uses Claude API with, Send a message and get response (non-streaming). (+32 more)

### Community 12 - "Community 12"
Cohesion: 0.06
Nodes (11): Get long-term portfolio holdings (equity delivery positions).         Returns li, Get current intraday/F&O positions.         Returns {'day': [...], 'net': [...]}, Get account margins.         Returns {'equity': {'available': {...}, 'utilised':, Get available cash balance., Get all orders for the day.         Each order: order_id, tradingsymbol, exchang, Universal order placement method.         Returns order_id string on success, No, Place a market order., Place a stop-loss (SL) limit order. (+3 more)

### Community 13 - "Community 13"
Cohesion: 0.06
Nodes (27): _build_evidence_pack(), Run all evidence fetchers in parallel; return one dict for all agents., FIIDIIFetcher, get_fii_dii_fetcher(), MarketMind AI - FII/DII Flow Tracker Fetches Foreign & Domestic Institutional In, Get summary stats: rolling net flows, trend, signal., Return synthetic recent data when API is down., Fetch FII/DII equity trade data for the last N trading days. (+19 more)

### Community 14 - "Community 14"
Cohesion: 0.05
Nodes (29): FundamentalAnalyzer, InvestmentAdvisor, NewsStockCorrelator, MarketMind AI - Market Predictor & Analysis Engine Automatic sector/index predic, Predicts sector/index direction and magnitude using:     - News sentiment moment, Predict direction/magnitude for all sectors.         Returns dict: sector -> pre, Predict a single sector, Fetches and analyzes fundamental data for stocks.     Uses NSE India API + Scree (+21 more)

### Community 15 - "Community 15"
Cohesion: 0.07
Nodes (25): _category_counts(), _chunk(), _classify(), FilingsIngester, _normalise_date(), MarketMind AI - Filings & Concall Ingester (W2.1)  Pulls NSE corporate-announcem, Pull raw rows from NSE corporate-announcements for the symbol., Fetch, chunk, embed. Idempotent thanks to deterministic IDs. (+17 more)

### Community 16 - "Community 16"
Cohesion: 0.08
Nodes (18): get_optimizer(), PortfolioOptimizer, MarketMind AI - Portfolio Optimizer Markowitz mean-variance optimization, effici, Generate efficient frontier points for risk-return chart., Compare all four strategies side by side., Implements:     - Markowitz Mean-Variance (maximum Sharpe, minimum variance), Equal risk contribution (risk parity)., Return (annual_return, annual_vol, sharpe). (+10 more)

### Community 17 - "Community 17"
Cohesion: 0.11
Nodes (27): feature_vector(), get_meta_stacker(), MetaStacker, Meta-stacker: feature dict → calibrated softmax(P_buy, P_sell, P_hold).  Inputs, Deterministic feature/label corpus used to bootstrap default weights.      Embed, Return the singleton meta-stacker. Bootstrapped on first call from     the deter, Test hook — wipes the singleton so each test gets a fresh fit if needed., Map the feature dict into a fixed-shape numpy vector.      Missing keys → 0.0. U (+19 more)

### Community 18 - "Community 18"
Cohesion: 0.1
Nodes (16): fake_mongo_alt_signals(), fake_mongo_col(), FakeMongoCol, frozen_now(), Shared pytest fixtures for the MarketMind test suite.  Design notes: - `fake_mon, `alt_signals` collection — keyed by composite `source:key`., Pin `datetime.now()` and `datetime.utcnow()` to a fixed point.      Useful for t, In-memory dict-backed pymongo.Collection stand-in.      Indexes documents by the (+8 more)

### Community 19 - "Community 19"
Cohesion: 0.1
Nodes (13): get_portfolio_simulator(), RLEnhancedEnsemble.make_investment_decision, PortfolioSimulator, Simulate portfolio with given allocations to sectors, Portfolio simulation engine using Monte Carlo methods, Combines RL agent actions with other signals for ensemble decisions, Make complete investment decision combining all signals, Get signal from technical indicators (+5 more)

### Community 20 - "Community 20"
Cohesion: 0.1
Nodes (13): VaR / CVaR calculator, get_portfolio_simulator(), PortfolioSimulator, MarketMind AI - Portfolio Simulator Monte Carlo portfolio simulations, Generate bull/base/bear scenario analysis, Calculate Value at Risk and Conditional VaR, Monte Carlo simulation engine for portfolio analysis, Calculate summary statistics (+5 more)

### Community 21 - "Community 21"
Cohesion: 0.12
Nodes (8): Database, MarketMind AI - Database Module SQLite database for storing news and cache, Store stock price data, Get historical stock prices, Store RL trading signal, Initialize database tables, Get recent RL signals, Store portfolio simulation result

### Community 22 - "Community 22"
Cohesion: 0.43
Nodes (6): cluster_news(), _cosine(), MarketMind AI - News Clusterer Groups similar headlines to remove duplicates and, Cluster news items by headline similarity.     Returns one representative item p, _tfidf_vector(), _tokenize()

### Community 23 - "Community 23"
Cohesion: 0.33
Nodes (6): fetch_news_via_claude (CLI+SDK), get_recent_news_from_mongo, _NEWS_PROMPT (Indian market query), run_claude_news_pipeline, store_news_in_mongo, anthropic SDK dependency

### Community 24 - "Community 24"
Cohesion: 0.5
Nodes (5): PortfolioOptimizer.efficient_frontier, _max_sharpe SLSQP solver, _min_variance solver, PortfolioOptimizer.optimize, _risk_parity solver

### Community 25 - "Community 25"
Cohesion: 0.4
Nodes (1): T11: e2e wiring check.  Imports the real ``server`` module and asserts that W2.3

### Community 26 - "Community 26"
Cohesion: 0.5
Nodes (4): Fetch news across all sectors, SECTOR_QUERIES dict (12 sectors), Keyword sector classifier (legacy), SECTOR_KEYWORDS classification dict

### Community 27 - "Community 27"
Cohesion: 0.5
Nodes (4): SectorPredictor._predict_sector composite scoring, SentimentAnalyzer.aggregate_sentiment, SentimentAnalyzer.analyze, Financial positive/negative lexicons

### Community 28 - "Community 28"
Cohesion: 0.67
Nodes (3): SECTOR_BETAS table, STRESS_SCENARIOS dict (India-relevant), RiskEngine.stress_test

### Community 29 - "Community 29"
Cohesion: 0.67
Nodes (3): ACTIONS BUY/HOLD/SELL/HEDGE/REBALANCE, TradingEnvironment._get_state (50-d feature vec), TradingEnvironment.step (5 actions)

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (2): InvestmentAdvisor.analyze_stock entry/exit/SL, FundamentalAnalyzer.score_stock

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (2): news SQLite schema, Database.store_news

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (2): rl_signals SQLite schema, Database.store_rl_signal

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (0): 

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Correlated sector simulation (Cholesky)

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): MarketMind memory index

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): RiskEngine.stock_var

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): RiskEngine.portfolio_var

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): WebSocket /ws live updates + alerts

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): /api/sectors/{sector}/recommendations endpoint

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): pymongo dependency

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): kiteconnect dependency

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): torch dependency

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): fastapi dependency

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): scikit-learn dependency

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): compute_beta (Cov/Var)

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): compute_sector_correlations

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Fetch the latest signals. Must never raise — decorate with safe_fetch.

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (0): 

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (0): 

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (0): 

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **363 isolated node(s):** `Wait for the server to start then open the browser.`, `MarketMind AI - Google News Fetcher Fetches financial news for all Indian market`, `Fetches financial news from Google News RSS for all sectors.`, `Fetch news for every sector; deduplicate by URL.`, `Fetch news for a single sector (with caching).` (+358 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 30`** (2 nodes): `InvestmentAdvisor.analyze_stock entry/exit/SL`, `FundamentalAnalyzer.score_stock`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (2 nodes): `news SQLite schema`, `Database.store_news`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (2 nodes): `rl_signals SQLite schema`, `Database.store_rl_signal`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Correlated sector simulation (Cholesky)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `MarketMind memory index`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `RiskEngine.stock_var`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `RiskEngine.portfolio_var`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `WebSocket /ws live updates + alerts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `/api/sectors/{sector}/recommendations endpoint`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `pymongo dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `kiteconnect dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `torch dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `fastapi dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `scikit-learn dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `compute_beta (Cov/Var)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `compute_sector_correlations`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Fetch the latest signals. Must never raise — decorate with safe_fetch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AppController` connect `Community 0` to `Community 1`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `KiteClient` connect `Community 0` to `Community 1`, `Community 4`, `Community 12`, `Community 6`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Why does `EnsembleForecaster` connect `Community 0` to `Community 3`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Are the 66 inferred relationships involving `AppController` (e.g. with `ConnectionManager` and `OrderRequest`) actually correct?**
  _`AppController` has 66 INFERRED edges - model-reasoned connections that need verification._
- **Are the 94 inferred relationships involving `SectorClassifier` (e.g. with `ConnectionManager` and `sector_recommendations()`) actually correct?**
  _`SectorClassifier` has 94 INFERRED edges - model-reasoned connections that need verification._
- **Are the 59 inferred relationships involving `KiteClient` (e.g. with `AppController` and `.__init__()`) actually correct?**
  _`KiteClient` has 59 INFERRED edges - model-reasoned connections that need verification._
- **Are the 74 inferred relationships involving `EnsembleForecaster` (e.g. with `ConnectionManager` and `OrderRequest`) actually correct?**
  _`EnsembleForecaster` has 74 INFERRED edges - model-reasoned connections that need verification._