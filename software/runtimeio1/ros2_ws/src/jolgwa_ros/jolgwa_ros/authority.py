from dataclasses import dataclass
from enum import Enum


class ControlAuthority(str, Enum):
    NONE = "NONE"
    LLM_ROUTE = "LLM_ROUTE"
    JETSON_EVENT_CAPTURE = "JETSON_EVENT_CAPTURE"
    JETSON_SAFETY = "JETSON_SAFETY"
    HUMAN = "HUMAN"


class JetsonSafetyState(str, Enum):
    CLEAR = "CLEAR"
    SLOW = "SLOW"
    HOLD = "HOLD"
    EVADE = "EVADE"
    STALE = "STALE"


@dataclass(frozen=True)
class AuthorityDecision:
    authority: ControlAuthority
    safety_state: JetsonSafetyState | None = None


def resolve_authority(
    *,
    human_active: bool,
    route_active: bool,
    jetson_event_active: bool = False,
    jetson_state: JetsonSafetyState | None,
    jetson_fresh: bool,
    require_jetson: bool,
) -> AuthorityDecision:
    """Resolve human > safety > event capture > LLM route."""

    if human_active:
        return AuthorityDecision(ControlAuthority.HUMAN)

    if require_jetson and not jetson_fresh:
        return AuthorityDecision(
            ControlAuthority.JETSON_SAFETY, JetsonSafetyState.STALE
        )

    if route_active and jetson_state is JetsonSafetyState.STALE:
        return AuthorityDecision(ControlAuthority.JETSON_SAFETY, jetson_state)

    if route_active and jetson_fresh and jetson_state not in (None, JetsonSafetyState.CLEAR):
        return AuthorityDecision(ControlAuthority.JETSON_SAFETY, jetson_state)

    if jetson_event_active:
        return AuthorityDecision(ControlAuthority.JETSON_EVENT_CAPTURE)

    if route_active:
        return AuthorityDecision(ControlAuthority.LLM_ROUTE)

    return AuthorityDecision(ControlAuthority.NONE)
