"""SEBI compliance layer (W5.3).

Pre-trade gating + post-trade audit log for personal & algorithmic trading
on Indian markets. Approximates SEBI (Prohibition of Insider Trading)
Regulations 2015 — designated-person trading-window enforcement, plus
per-portfolio concentration warnings. Persists every pre-trade decision
and order attempt to MongoDB ``compliance_audit_log`` (no TTL — regulatory
artefact, not cache).

Modules:
    audit_log       — Mongo-backed append-only store
    insider_window  — pure compute over pre-fetched NSE corp-announcements
    position_limits — pure compute (concentration % this section)
    pretrade_check  — orchestrator composing the three above

All compute modules are deterministic and side-effect-free. The store and
orchestrator degrade gracefully when ``mongo_col`` is None (no-op writes,
empty reads), matching the ``forecast_cache`` precedent.
"""
