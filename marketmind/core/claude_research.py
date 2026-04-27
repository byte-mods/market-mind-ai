"""
MarketMind AI - Claude-Powered Research Engine
Generates institutional-grade investment research reports and answers
natural language queries about the market using Claude API.
"""
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_api_key() -> Optional[str]:
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), '..', '..', 'local.json')
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get('anthropic', {}).get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
    except Exception:
        return os.environ.get('ANTHROPIC_API_KEY')


def _get_client():
    api_key = _load_api_key()
    if not api_key:
        raise ValueError("Anthropic API key not found in local.json or ANTHROPIC_API_KEY env var")
    from anthropic import Anthropic
    return Anthropic(api_key=api_key)


# ── Research Report ────────────────────────────────────────────────────────────

def generate_research_report(symbol: str, stock_data: Dict, sector_data: Dict,
                              rl_signal: Optional[Dict] = None,
                              fii_summary: Optional[Dict] = None) -> Dict:
    """
    Generate a full investment research report for a stock using Claude.
    Returns {'report': str (markdown), 'verdict': str, 'target': float}
    """
    # Note: LLM backend is resolved through the router below — no direct client.

    # Build context
    fund = {
        'pe': stock_data.get('pe_ratio'),
        'pb': stock_data.get('pb_ratio'),
        'roe': stock_data.get('roe'),
        'roce': stock_data.get('roce'),
        'market_cap': stock_data.get('market_cap'),
        'debt_equity': stock_data.get('debt_equity'),
        'revenue_growth': stock_data.get('revenue_growth'),
        'profit_growth': stock_data.get('profit_growth'),
        'eps': stock_data.get('eps'),
        'div_yield': stock_data.get('dividend_yield'),
    }
    tech = stock_data.get('technical_indicators', {})
    news = stock_data.get('related_news', [])[:5]
    price = stock_data.get('current_price', 0)
    change_pct = stock_data.get('change_pct', 0)

    news_titles = '\n'.join(f"- {n.get('title', '')}" for n in news) or 'No recent news'

    fii_ctx = ''
    if fii_summary:
        fii_ctx = (
            f"\n**FII/DII Flows (last {fii_summary.get('days',5)} days):**\n"
            f"- FII Net: ₹{fii_summary.get('fii_net_total',0):.0f}Cr ({fii_summary.get('fii_signal','—')})\n"
            f"- DII Net: ₹{fii_summary.get('dii_net_total',0):.0f}Cr\n"
            f"- Trend: {fii_summary.get('fii_trend','—')}"
        )

    rl_ctx = ''
    if rl_signal:
        rl_ctx = (
            f"\n**RL Trading Signal:** {rl_signal.get('action','—')} "
            f"(confidence: {rl_signal.get('confidence',0)*100:.0f}%)\n"
            f"Entry: ₹{rl_signal.get('entry_price',0):.2f}, "
            f"Target: ₹{rl_signal.get('exit_price',0):.2f}, "
            f"SL: ₹{rl_signal.get('stop_loss',0):.2f}"
        )

    prompt = f"""You are a senior equity research analyst at a top-tier investment bank.
Generate a comprehensive, data-driven investment research report for **{symbol}** in professional McKinsey/Goldman Sachs style.

## Available Data

**Current Price:** ₹{price:.2f} ({'+' if change_pct >= 0 else ''}{change_pct*100 if abs(change_pct)<1 else change_pct:.2f}%)

**Fundamental Metrics:**
- P/E Ratio: {fund['pe'] or '—'}
- P/B Ratio: {fund['pb'] or '—'}
- ROE: {fund['roe'] or '—'}%
- ROCE: {fund['roce'] or '—'}%
- Market Cap: ₹{(fund['market_cap'] or 0)/1e7:.0f}Cr
- Debt/Equity: {fund['debt_equity'] or '—'}
- Revenue Growth: {fund['revenue_growth'] or '—'}%
- Profit Growth: {fund['profit_growth'] or '—'}%
- EPS: {fund['eps'] or '—'}
- Dividend Yield: {fund['div_yield'] or '—'}%

**Technical Indicators:**
- RSI (14): {tech.get('rsi', '—')}
- MACD: {tech.get('macd', '—')} | Signal: {tech.get('macd_signal', '—')}
- 20-day MA: ₹{tech.get('ma_20', '—')} | 50-day MA: ₹{tech.get('ma_50', '—')}
- 20-day Momentum: {(tech.get('momentum_20', 0) or 0)*100:.2f}%
- Above MA50: {tech.get('above_ma_50', '—')} | Above MA200: {tech.get('above_ma_200', '—')}
- Volume Ratio: {tech.get('volume_ratio', '—')}x

**Recent News:**
{news_titles}
{fii_ctx}
{rl_ctx}

---

Write a complete research report with these exact sections:

## Executive Summary
[2-3 sentences: verdict, key reasons, target price]

## Investment Thesis
[3-4 bullet points: why buy/hold/avoid — be specific with numbers]

## Fundamental Analysis
[Analyse PE vs sector average, ROE quality, debt sustainability, growth trajectory. Call out any red flags.]

## Technical Picture
[Current trend, key support/resistance, momentum, whether RSI/MACD supports entry timing]

## Key Risks
[3-4 specific risks: regulatory, competitive, macro, management]

## Valuation & Price Target
[Use at least 2 methods: PE-based target, ROE-based intrinsic value, or technical target. Give 12-month price target with bull/base/bear scenarios.]

## Verdict
**Rating:** [BUY / HOLD / SELL / AVOID]
**Conviction:** [High / Medium / Low]
**12-Month Price Target:** ₹[number]
**Stop Loss:** ₹[number]
**Horizon:** [Intraday / Swing / Positional]

Be direct, specific, use numbers. No generic disclaimers. Write like you're presenting to a portfolio manager."""

    try:
        from marketmind.core.llm import get_router
        router = get_router()
        report_text = router.chat(
            [{"role": "user", "content": prompt}],
            role="research",
            max_tokens=2000,
            temperature=0.7,
            timeout=240.0,   # heavy research prompt; CLI/API may take 60–180s
        )

        # Extract verdict and target from report
        verdict = 'HOLD'
        target = price
        for line in report_text.split('\n'):
            line_lower = line.lower()
            if '**rating:**' in line_lower or 'rating:' in line_lower:
                if 'strong buy' in line_lower: verdict = 'STRONG BUY'
                elif 'buy' in line_lower: verdict = 'BUY'
                elif 'sell' in line_lower: verdict = 'SELL'
                elif 'avoid' in line_lower: verdict = 'AVOID'
            if 'price target' in line_lower and '₹' in line:
                try:
                    parts = line.split('₹')
                    for p in parts[1:]:
                        num = ''.join(c for c in p[:10] if c.isdigit() or c == '.')
                        if num:
                            target = float(num)
                            break
                except Exception:
                    pass

        return {
            'symbol': symbol,
            'report': report_text,
            'verdict': verdict,
            'target_price': round(target, 2),
            'current_price': price,
            'upside_pct': round((target / price - 1) * 100, 1) if price else 0,
        }
    except Exception as e:
        err_str = str(e)
        if 'authentication_error' in err_str or '401' in err_str or 'invalid x-api-key' in err_str or 'API key' in err_str:
            err = "Research engine unavailable: API key is missing or invalid for the configured LLM backend."
        else:
            err = f"Research engine temporarily unavailable ({type(e).__name__})."
        logger.error(f"Research report error for {symbol}: {e}")
        return {'error': err, 'symbol': symbol}


# ── Conversational Assistant ───────────────────────────────────────────────────

class MarketAssistant:
    """
    Streaming conversational assistant with market context.
    Uses Claude API with conversation history.
    """

    SYSTEM_PROMPT = """You are MarketMind AI — an expert Indian equity market analyst and financial advisor.
You have deep knowledge of:
- NSE/BSE listed companies, Nifty500 universe (large-cap, mid-cap, small-cap)
- Indian macroeconomics (RBI policy, FII flows, budget impact)
- Technical analysis (candlestick patterns, indicators, chart patterns)
- Fundamental analysis (DCF, PE/PB/ROE analysis, accounting quality)
- F&O strategies (options, futures, hedging)
- Sector rotation, market cycles, momentum strategies

Rules:
- Always give specific, actionable answers with numbers
- For stock queries: mention current price context, key levels, and rationale
- For macro queries: connect to specific sector/stock impact
- Never give generic advice — be precise and direct
- Add risk caveats but keep them brief
- Use Indian context: mention Sensex/Nifty levels, RBI rates, INR, FII activity
- Format with bold headers and bullet points for clarity"""

    def __init__(self):
        self._history: List[Dict] = []
        # No direct Anthropic client — backend is resolved via the LLM router.

    def chat(self, user_message: str, market_context: Optional[Dict] = None) -> str:
        """Send a message and get response (non-streaming)."""
        # Inject market context into first message if provided
        augmented_msg = user_message
        if market_context and not self._history:
            ctx_lines = []
            if market_context.get('nifty500') or market_context.get('nifty'):
                ctx_lines.append(f"Nifty500: {market_context.get('nifty500') or market_context.get('nifty')}")
            if market_context.get('vix'):
                ctx_lines.append(f"India VIX: {market_context['vix']}")
            if market_context.get('fii_signal'):
                ctx_lines.append(f"FII Signal: {market_context['fii_signal']}")
            if ctx_lines:
                augmented_msg = f"[Market Context: {', '.join(ctx_lines)}]\n\n{user_message}"

        self._history.append({"role": "user", "content": augmented_msg})

        try:
            from marketmind.core.llm import get_router
            router = get_router()
            assistant_msg = router.chat(
                self._history[-20:],
                role="debate",
                system=self.SYSTEM_PROMPT,
                max_tokens=1500,
                temperature=0.7,
                timeout=180.0,
            )
            self._history.append({"role": "assistant", "content": assistant_msg})
            return assistant_msg
        except Exception as e:
            err_str = str(e)
            if 'authentication_error' in err_str or '401' in err_str or 'invalid x-api-key' in err_str or 'API key' in err_str:
                error_msg = "AI assistant is unavailable: API key is missing or invalid. Update local.json or set the env var for the configured backend."
            else:
                error_msg = f"AI assistant is temporarily unavailable ({type(e).__name__})."
            logger.error(f"Chat error: {e}")
            self._history.append({"role": "assistant", "content": error_msg})
            return error_msg

    def reset(self):
        self._history = []


# Session-based assistant instances (per WebSocket connection)
_assistants: Dict[str, MarketAssistant] = {}

def get_assistant(session_id: str = 'default') -> MarketAssistant:
    if session_id not in _assistants:
        _assistants[session_id] = MarketAssistant()
    return _assistants[session_id]
