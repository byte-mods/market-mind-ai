"""Tests for RiskEngine.holdings_from_portfolio + sector_for_symbol.

Both feed the new /api/risk/load-holdings endpoint, which exists so the Risk
Analytics page can pre-fill from real Kite holdings instead of forcing the
user to retype every position (the cause of the 'feels like a demo' bug)."""
from marketmind.analysis.risk_engine import RiskEngine


def test_sector_for_symbol_resolves_known_stocks():
    e = RiskEngine()
    # SectorClassifier groundtruth — RELIANCE is Energy, TCS is IT, HDFCBANK is Banking.
    assert e.sector_for_symbol('RELIANCE') == 'Energy'
    assert e.sector_for_symbol('TCS') == 'IT'
    assert e.sector_for_symbol('HDFCBANK') == 'Banking'


def test_sector_for_symbol_falls_back_to_default_for_unknown():
    e = RiskEngine()
    # Unknown symbol → Banking (median beta) so stress shocks still apply
    # rather than silently dropping the position.
    assert e.sector_for_symbol('TOTALLYMADEUPSYM') == 'Banking'
    assert e.sector_for_symbol('TOTALLYMADEUPSYM', default='IT') == 'IT'


def test_sector_for_symbol_is_case_insensitive():
    e = RiskEngine()
    assert e.sector_for_symbol('reliance') == 'Energy'
    assert e.sector_for_symbol('Tcs') == 'IT'


def test_holdings_from_portfolio_marks_to_market_with_last_price():
    e = RiskEngine()
    summary = {
        'authenticated': True,
        'holdings': [
            {'tradingsymbol': 'RELIANCE', 'quantity': 10, 'last_price': 2500.0,
             'close_price': 2480.0, 'average_price': 2400.0},
        ],
    }
    out = e.holdings_from_portfolio(summary)
    assert len(out) == 1
    assert out[0]['symbol'] == 'RELIANCE'
    assert out[0]['value'] == 25000.0  # qty * last_price
    assert out[0]['sector'] == 'Energy'


def test_holdings_from_portfolio_uses_close_price_after_hours():
    """Kite returns last_price=0 outside market hours; we must fall back to
    close_price (or average_price) so the value isn't silently dropped."""
    e = RiskEngine()
    summary = {
        'holdings': [
            {'tradingsymbol': 'TCS', 'quantity': 5, 'last_price': 0,
             'close_price': 3800.0, 'average_price': 3500.0},
        ],
    }
    out = e.holdings_from_portfolio(summary)
    assert len(out) == 1
    assert out[0]['value'] == 19000.0  # 5 * close_price
    assert out[0]['sector'] == 'IT'


def test_holdings_from_portfolio_skips_zero_value_rows():
    """A holding with no usable price would otherwise contribute zero to the
    portfolio total and divide-by-zero in the VaR calc. Drop such rows."""
    e = RiskEngine()
    summary = {
        'holdings': [
            {'tradingsymbol': 'GOODSTOCK', 'quantity': 10, 'last_price': 100,
             'close_price': 0, 'average_price': 0},
            {'tradingsymbol': 'BADSTOCK', 'quantity': 0, 'last_price': 0,
             'close_price': 0, 'average_price': 0},
        ],
    }
    out = e.holdings_from_portfolio(summary)
    assert [h['symbol'] for h in out] == ['GOODSTOCK']


def test_holdings_from_portfolio_empty_input_returns_empty_list():
    e = RiskEngine()
    assert e.holdings_from_portfolio({}) == []
    assert e.holdings_from_portfolio({'holdings': []}) == []
    assert e.holdings_from_portfolio(None) == []


def test_holdings_from_portfolio_classifies_unknown_symbol_with_default_sector():
    e = RiskEngine()
    summary = {
        'holdings': [
            {'tradingsymbol': 'NEWIPO2026', 'quantity': 100, 'last_price': 50,
             'close_price': 50, 'average_price': 45},
        ],
    }
    out = e.holdings_from_portfolio(summary)
    assert out[0]['sector'] == 'Banking'  # default for unknown
    assert out[0]['value'] == 5000.0
