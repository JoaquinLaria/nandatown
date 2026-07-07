# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateUsage=false
"""Unit tests for the CheckoutFrontier negotiation plugin.

The commerce risk model is the test emphasis: hard wallet budgets and vendor
cost floors are inviolable (never accept, echo, or propose across your own
cap), a generous early quote stays live until it can be accepted, and a
checkout never closes without both signatures.
"""

from __future__ import annotations

import asyncio

from nest_core.types import AgentId, Money, NegotiationStatus, Terms
from nest_plugins_reference.negotiation.checkout_frontier import CheckoutFrontier


def _terms(price: int, deadline: int) -> Terms:
    return Terms(price=Money(amount=price), conditions={"deadline_days": deadline})


def _pd(terms: Terms) -> tuple[int, int]:
    price = terms.price.amount if terms.price is not None else 0
    return price, int(terms.conditions.get("deadline_days", 0))


def _buyer(**overrides: object) -> CheckoutFrontier:
    kwargs: dict[str, object] = {
        "weights": {"price": 0.5, "deadline": 0.5},
        "price_range": (40, 60),
        "deadline_range": (1, 11),
        "side": "buyer",
        "patience": 0.8,
    }
    kwargs.update(overrides)
    return CheckoutFrontier(AgentId("shopper"), **kwargs)  # type: ignore[arg-type]


def _vendor(**overrides: object) -> CheckoutFrontier:
    kwargs: dict[str, object] = {
        "weights": {"price": 0.5, "deadline": 0.5},
        "price_range": (40, 60),
        "deadline_range": (1, 11),
        "side": "seller",
        "patience": 0.8,
    }
    kwargs.update(overrides)
    return CheckoutFrontier(AgentId("vendor"), **kwargs)  # type: ignore[arg-type]


async def _respond_to(
    agent: CheckoutFrontier, offers: list[tuple[int, int]]
) -> list[tuple[bool, tuple[int, int] | None]]:
    """Feed a sequence of counterparty offers; collect (accepted, counter) pairs."""
    session = await agent.open(AgentId("counterparty"), _terms(*_pd(_terms(40, 1))))
    out: list[tuple[bool, tuple[int, int] | None]] = []
    for price, deadline in offers:
        await agent.offer(session, _terms(price, deadline))
        resp = await agent.respond(session)
        counter = _pd(resp.counter_terms) if resp.counter_terms is not None else None
        out.append((resp.accepted, counter))
    return out


# Utility


def test_utility_directional_buyer() -> None:
    """The buyer prefers a lower price and a shorter deadline."""
    buyer = _buyer()
    assert buyer.utility(_terms(45, 3)) > buyer.utility(_terms(55, 3))
    assert buyer.utility(_terms(45, 3)) > buyer.utility(_terms(45, 9))


def test_utility_directional_vendor() -> None:
    """The vendor prefers a higher price and a longer deadline."""
    vendor = _vendor()
    assert vendor.utility(_terms(55, 3)) > vendor.utility(_terms(45, 3))
    assert vendor.utility(_terms(45, 9)) > vendor.utility(_terms(45, 3))


def test_ideal_bundle_scores_one() -> None:
    """Each side's ideal corner scores exactly 1.0."""
    assert _buyer().utility(_terms(40, 1)) == 1.0
    assert _vendor().utility(_terms(60, 11)) == 1.0


# Weights from seed


def test_weights_from_seed_deterministic() -> None:
    """The same (agent_id, seed) always derives the same weights."""
    a = CheckoutFrontier(AgentId("shopper"), seed=7)
    b = CheckoutFrontier(AgentId("shopper"), seed=7)
    assert a._weights == b._weights


def test_weights_from_seed_vary_by_agent() -> None:
    """Different agent ids draw different preferences from the same seed."""
    a = CheckoutFrontier(AgentId("shopper-a"), seed=7)
    b = CheckoutFrontier(AgentId("shopper-b"), seed=7)
    assert a._weights != b._weights


def test_weights_from_seed_well_formed() -> None:
    """Derived weights are 6-dp rounded, in range, and sum to 1."""
    weights = CheckoutFrontier(AgentId("shopper"), seed=42)._weights
    for value in weights.values():
        assert 0.0 <= value <= 1.0
        assert value == round(value, 6)
    assert abs(weights["price"] + weights["deadline"] - 1.0) < 1e-9


# Acceptance and the standing-quote echo


def test_accepts_ideal_offer_immediately() -> None:
    """An offer at the agent's ideal corner is accepted in round one."""
    result = asyncio.run(_respond_to(_buyer(), [(40, 1)]))
    assert result[0][0] is True


def test_echoes_best_standing_quote_when_aspiration_reached() -> None:
    """A generous early quote is re-proposed verbatim once aspiration decays.

    Round 1: (44, 2) is worth 0.85 to the buyer, below aspiration 1.0, so it
    is not accepted, but it stays live. Round 2: the vendor regresses to its
    own ideal (60, 11); aspiration is now 0.8 <= 0.85, so instead of accepting
    the bad current offer or conceding fresh ground, the buyer re-proposes
    the standing (44, 2). Round 3: the vendor re-offers it; the buyer accepts.
    """
    result = asyncio.run(_respond_to(_buyer(), [(44, 2), (60, 11), (44, 2)]))
    assert result[0] == (False, (40, 1))
    assert result[1] == (False, (44, 2))
    assert result[2][0] is True


def test_earliest_quote_wins_exact_utility_ties() -> None:
    """Among utility-tied standing quotes the earliest is echoed.

    With equal weights and matched spans, (46, 2) and (44, 4) tie exactly for
    the buyer (0.85 each). The earlier one is the counterparty's least
    conceded, so echoing it never trades away counterparty surplus for
    nothing.
    """
    buyer = _buyer(deadline_range=(1, 21))

    async def go() -> tuple[int, int] | None:
        session = await buyer.open(AgentId("vendor"), _terms(40, 1))
        await buyer.offer(session, _terms(46, 2))
        await buyer.respond(session)
        await buyer.offer(session, _terms(44, 4))
        resp = await buyer.respond(session)
        return _pd(resp.counter_terms) if resp.counter_terms is not None else None

    # u(46,2) = 0.5*(14/20) + 0.5*(19/20) = 0.825; u(44,4) = 0.5*(16/20) + 0.5*(17/20) = 0.825
    assert asyncio.run(go()) == (46, 2)


# Hard money caps


def test_buyer_never_accepts_or_counters_above_budget() -> None:
    """Attractive but unaffordable quotes are refused; counters respect the wallet."""
    buyer = _buyer(
        weights={"price": 0.1, "deadline": 0.9},
        price_range=(40, 120),
        deadline_range=(1, 21),
        patience=0.9,
        budget=100,
    )
    result = asyncio.run(_respond_to(buyer, [(105, 1), (110, 1), (108, 1), (110, 1)]))
    for accepted, counter in result:
        assert accepted is False
        assert counter is not None
        assert counter[0] <= 100


def test_buyer_accepts_rush_quote_within_budget() -> None:
    """The same rush logroll is accepted the moment it fits the wallet."""
    buyer = _buyer(
        weights={"price": 0.1, "deadline": 0.9},
        price_range=(40, 120),
        deadline_range=(1, 21),
        patience=0.9,
        budget=100,
    )
    result = asyncio.run(_respond_to(buyer, [(105, 1), (100, 1)]))
    assert result[0][0] is False
    assert result[1][0] is True


def test_vendor_never_goes_below_floor() -> None:
    """A vendor with a cost floor never proposes or accepts beneath it."""
    vendor = _vendor(
        price_range=(40, 120),
        deadline_range=(1, 21),
        patience=0.9,
        floor=80,
    )
    result = asyncio.run(_respond_to(vendor, [(50, 21), (60, 21), (70, 21), (75, 21)]))
    for accepted, counter in result:
        assert accepted is False
        assert counter is not None
        assert counter[0] >= 80


def test_no_eligible_bundle_declines_to_counter() -> None:
    """A wallet below every feasible price yields refusal, not a fantasy counter."""
    buyer = _buyer(budget=30)
    result = asyncio.run(_respond_to(buyer, [(40, 1)]))
    assert result[0] == (False, None)


# Patience-horizon termination


def test_declines_fresh_counters_past_horizon() -> None:
    """Past max_rounds responds, the agent stops conceding and declines to counter.

    Aspiration is frozen at the horizon, so a fresh counter would repeat the
    same deterministic exchange forever; the plugin stands down instead of
    relying on a driver's round cap to end the session.
    """
    buyer = _buyer(max_rounds=3)
    result = asyncio.run(_respond_to(buyer, [(60, 11)] * 5))
    for accepted, counter in result[:3]:
        assert accepted is False
        assert counter is not None
    assert result[3] == (False, None)
    assert result[4] == (False, None)


def test_echoes_at_most_once_past_horizon_then_accepts_or_declines() -> None:
    """Past the horizon a clearing standing quote is re-proposed once, then never again.

    The counterparty's decision on a repeated echo under frozen aspiration is
    deterministic, so repeating it is provably futile: the second attempt
    declines. A clearing offer that arrives on the table is still accepted,
    at any round.
    """
    buyer = _buyer(max_rounds=2)
    result = asyncio.run(_respond_to(buyer, [(44, 2), (60, 11), (60, 11), (60, 11), (44, 2)]))
    assert result[0] == (False, (40, 1))  # aspiration 1.0: fresh counter
    assert result[1] == (False, (44, 2))  # pre-horizon echo of the standing quote
    assert result[2] == (False, (44, 2))  # the one allowed past-horizon echo
    assert result[3] == (False, None)  # repeat echo is futile: decline
    assert result[4][0] is True  # a clearing current offer is accepted even now


# Direction reading


def test_direction_weights_favor_held_attribute() -> None:
    """The attribute the counterparty holds fixed gets the heavier coefficient."""
    buyer = _buyer()

    async def observe(offers: list[tuple[int, int]]) -> tuple[float, float]:
        session = await buyer.open(AgentId("vendor"), _terms(40, 1))
        for price, deadline in offers:
            await buyer.offer(session, _terms(price, deadline))
        return buyer._direction_weights(session.id)

    # Price conceded, deadline held: they value the deadline.
    assert asyncio.run(observe([(60, 11), (56, 11)])) == (1.0, 2.0)
    # Deadline conceded, price held: they value the price.
    assert asyncio.run(observe([(60, 11), (60, 8)])) == (2.0, 1.0)
    # One quote only: no signal yet.
    assert asyncio.run(observe([(60, 11)])) == (1.0, 1.0)


# Close semantics


def test_close_returns_none_and_rejects_unagreed_session() -> None:
    """No agreement without both signatures: unagreed sessions break down."""
    buyer = _buyer()

    async def go() -> tuple[object, NegotiationStatus]:
        session = await buyer.open(AgentId("vendor"), _terms(40, 1))
        agreement = await buyer.close(session)
        return agreement, session.status

    agreement, status = asyncio.run(go())
    assert agreement is None
    assert status == NegotiationStatus.REJECTED


def test_close_returns_agreement_when_agreed() -> None:
    """An AGREED session closes into an Agreement carrying the final terms."""
    buyer = _buyer()

    async def go() -> tuple[int, int]:
        session = await buyer.open(AgentId("vendor"), _terms(40, 1))
        await buyer.offer(session, _terms(44, 2))
        session.status = NegotiationStatus.AGREED
        agreement = await buyer.close(session)
        assert agreement is not None
        return _pd(agreement.terms)

    assert asyncio.run(go()) == (44, 2)


def test_empty_terms_accepted_as_reference_fallback() -> None:
    """Terms without a price fall back to acceptance, matching the reference plugins."""
    buyer = _buyer()

    async def go() -> bool:
        session = await buyer.open(AgentId("vendor"), Terms())
        resp = await buyer.respond(session)
        return resp.accepted

    assert asyncio.run(go()) is True
