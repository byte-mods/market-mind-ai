"""Pre-trade compliance orchestrator (W5.3 T3).

Composes the three pure compute layers:
    insider_window  — only enforced for designated symbols
    position_limits — always enforced (concentration warning)

Writes one ``audit_log`` entry per check (source=pretrade). The result
maps:
    any errors        → DECISION_BLOCK
    any warnings only → DECISION_WARN
    nothing           → DECISION_ALLOW

Designated-symbols list is supplied by the caller as a ``Callable[[],
Iterable[str]]`` so the orchestrator stays free of state and the
controller can persist the list however it likes (Mongo, in-memory,
config file). An empty list means no insider-window enforcement happens
for any symbol — the safe default.
"""
from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from marketmind.compliance.audit_log import (
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_WARN,
    SOURCE_PRETRADE,
    AuditLogEntry,
    AuditLogStore,
)
from marketmind.compliance.insider_window import compute_insider_window
from marketmind.compliance.position_limits import (
    DEFAULT_MAX_CONCENTRATION_PCT,
    check_position_limits,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PretradeDecision:
    """Result of a pre-trade compliance check."""

    decision: str
    reasons: List[str] = field(default_factory=list)
    ts: Optional[_dt.datetime] = None
    audit_id: Optional[str] = None
    insider_window_open: Optional[bool] = None
    insider_window_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if isinstance(self.ts, _dt.datetime):
            d["ts"] = self.ts.isoformat()
        return d


class PretradeChecker:
    """Stateless orchestrator. Audit store + designated-symbols are injected."""

    def __init__(
        self,
        audit_log: AuditLogStore,
        designated_symbols_provider: Callable[[], Iterable[str]],
        max_concentration_pct: float = DEFAULT_MAX_CONCENTRATION_PCT,
    ) -> None:
        self.audit_log = audit_log
        self._designated = designated_symbols_provider
        self.max_concentration_pct = max_concentration_pct

    def check(
        self,
        symbol: str,
        transaction_type: str,
        quantity: float,
        price: float,
        announcements: List[Dict[str, Any]],
        holdings: List[Dict[str, Any]],
        today: Optional[_dt.date] = None,
    ) -> PretradeDecision:
        """Run the full pre-trade gate. Always returns; never raises on data."""
        ts = _dt.datetime.now(_dt.timezone.utc)
        today = today or ts.date()
        sym_norm = (symbol or "").upper().strip()

        reasons: List[str] = []
        insider_open: Optional[bool] = None
        insider_reason: Optional[str] = None
        # Structured flag — never gate the decision on substring matches against
        # human-readable reason text; that coupling silently breaks if the
        # message wording is ever changed for logging readability.
        insider_blocked = False

        # Insider window — only for designated symbols.
        try:
            designated = {s.upper().strip() for s in (self._designated() or [])}
        except Exception as e:  # noqa: BLE001
            logger.warning("designated_symbols_provider failed: %s", e)
            designated = set()
        if sym_norm and sym_norm in designated:
            try:
                window = compute_insider_window(sym_norm, announcements or [], today)
                insider_open = window.is_open
                insider_reason = window.reason
                if not window.is_open:
                    insider_blocked = True
                    reasons.append(f"insider window closed: {window.reason}")
            except Exception as e:  # noqa: BLE001
                logger.warning("insider_window check failed: %s", e)
                # Fail-open with a WARN — a hard fail-closed would block every
                # trade whenever the NSE feed flakes. Caller surfaces in audit.
                reasons.append(f"insider window check unavailable: {e}")

        # Position limits — always.
        limit_status = check_position_limits(
            symbol=sym_norm,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
            holdings=holdings or [],
            max_concentration_pct=self.max_concentration_pct,
        )
        reasons.extend(limit_status.errors)
        reasons.extend(limit_status.warnings)

        if limit_status.errors or insider_blocked:
            decision = DECISION_BLOCK
        elif reasons:
            decision = DECISION_WARN
        else:
            decision = DECISION_ALLOW

        # Audit entry — best-effort; failures here never block the decision.
        audit_id: Optional[str] = None
        try:
            entry = AuditLogEntry(
                ts=ts,
                symbol=sym_norm or "?",
                transaction_type=(transaction_type or "?").upper(),
                quantity=float(quantity or 0),
                price=float(price or 0),
                decision=decision,
                reasons=reasons,
                source=SOURCE_PRETRADE,
            )
            audit_id = self.audit_log.append(entry)
        except Exception as e:  # noqa: BLE001
            logger.warning("pretrade audit write failed: %s", e)

        return PretradeDecision(
            decision=decision,
            reasons=reasons,
            ts=ts,
            audit_id=audit_id,
            insider_window_open=insider_open,
            insider_window_reason=insider_reason,
        )
