# SPDX-License-Identifier: Apache-2.0
"""Property tests for CheckoutFrontier (Hypothesis).

Four property families:

* **Determinism**: identical construction plus an identical offer sequence
  yields an identical response transcript (no wall-clock, no unseeded RNG).
* **Monotonic concession**: an agent's *fresh* counters (bundles that are not
  echoes of standing counterparty quotes) have non-increasing own utility.
* **No dominated self-play** (the flagship): two CheckoutFrontier agents never
  settle on an agreement Pareto-dominated by any cap-feasible bundle either
  side exchanged. This is exactly the guarantee the merged ``pareto`` plugin
  documents as unattainable for pure trade-off concession in
  ``test_fsj_tradeoff_does_not_guarantee_pareto_optimality`` (a generous early
  bundle is ephemeral there; here every standing quote stays live), asserted
  green here including on that test's pinned counterexample configuration.
* **Hard caps and rationality**: settled deals land inside
  ``[floor, budget]`` with both utilities at or above reservation, and
  mutually exclusive caps always break down instead of dealing.

Weights are drawn on the six-decimal lattice (``k / 1_000_000``) with bundle
spans capped at 15, so every nonzero utility difference on the integer grid
exceeds ``1/(10**6 * 15 * 15) ~ 4.4e-9``, comfortably above the ``1e-9``
comparison epsilon: near-ties are exact ties and the dominance check cannot
be gamed by float noise.
"""

from __future__ import annotations

import asyncio
from typing import NamedTuple

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn
from nest_core.types import AgentId, Money, Terms
from nest_plugins_reference.negotiation.checkout_frontier import CheckoutFrontier

_EPS = 1e-9


def _terms(price: int, deadline: int) -> Terms:
    return Terms(price=Money(amount=price), conditions={"deadline_days": deadline})


def _pd(terms: Terms) -> tuple[int, int]:
    price = terms.price.amount if terms.price is not None else 0
    return price, int(terms.conditions.get("deadline_days", 0))


def _dominates(ub_x: float, us_x: float, ub_y: float, us_y: float) -> bool:
    """X dominates Y: no worse for either party, strictly better for one (eps-guarded)."""
    no_worse = ub_x >= ub_y - _EPS and us_x >= us_y - _EPS
    strictly_better = ub_x > ub_y + _EPS or us_x > us_y + _EPS
    return no_worse and strictly_better


class _Cfg(NamedTuple):
    plo: int
    phi: int
    dlo: int
    dhi: int
    patience: float
    buyer_wp: float
    seller_wd: float
    reservation: float
    budget: int | None
    floor: int | None


@st.composite
def _cfgs(draw: DrawFn) -> _Cfg:
    plo = draw(st.integers(10, 50))
    phi = plo + draw(st.integers(5, 15))
    dlo = draw(st.integers(1, 5))
    dhi = dlo + draw(st.integers(5, 15))
    patience = draw(st.floats(0.7, 0.95, allow_nan=False, allow_infinity=False))
    # Weights on the 6-dp lattice keep utility gaps far above the epsilon.
    buyer_wp = draw(st.integers(100_000, 900_000)) / 1_000_000
    seller_wd = draw(st.integers(100_000, 900_000)) / 1_000_000
    reservation = draw(st.floats(0.0, 0.3, allow_nan=False, allow_infinity=False))
    budget = draw(st.one_of(st.none(), st.integers(plo, phi)))
    floor = draw(st.one_of(st.none(), st.integers(plo, phi)))
    return _Cfg(plo, phi, dlo, dhi, patience, buyer_wp, seller_wd, reservation, budget, floor)


@st.composite
def _cfg_and_offers(draw: DrawFn) -> tuple[_Cfg, list[tuple[int, int]]]:
    cfg = draw(_cfgs())
    offers = draw(
        st.lists(
            st.tuples(st.integers(cfg.plo, cfg.phi), st.integers(cfg.dlo, cfg.dhi)),
            min_size=1,
            max_size=8,
        )
    )
    return cfg, offers


def _make_buyer(cfg: _Cfg, max_rounds: int = 12) -> CheckoutFrontier:
    return CheckoutFrontier(
        AgentId("shopper"),
        weights={"price": cfg.buyer_wp, "deadline": round(1.0 - cfg.buyer_wp, 6)},
        price_range=(cfg.plo, cfg.phi),
        deadline_range=(cfg.dlo, cfg.dhi),
        side="buyer",
        patience=cfg.patience,
        reservation=cfg.reservation,
        budget=cfg.budget,
        max_rounds=max_rounds,
    )


def _make_vendor(cfg: _Cfg, max_rounds: int = 12) -> CheckoutFrontier:
    return CheckoutFrontier(
        AgentId("vendor"),
        weights={"price": round(1.0 - cfg.seller_wd, 6), "deadline": cfg.seller_wd},
        price_range=(cfg.plo, cfg.phi),
        deadline_range=(cfg.dlo, cfg.dhi),
        side="seller",
        patience=cfg.patience,
        reservation=cfg.reservation,
        floor=cfg.floor,
        max_rounds=max_rounds,
    )


async def _self_play(
    buyer: CheckoutFrontier, vendor: CheckoutFrontier, bounds: tuple[int, int, int, int]
) -> tuple[tuple[int, int] | None, list[tuple[int, int]]]:
    """Run two agents to settlement; return the agreement (if any) and exchanged bundles.

    Each side opens from its best-for-self position, then alternately responds
    to the other's latest offer. An agent either accepts (the offer becomes
    the agreement), declines without a counter (breakdown), or counters; every
    counter is recorded and forwarded.
    """
    plo, phi, dlo, dhi = bounds
    buyer_open = _terms(plo, dlo)
    vendor_open = _terms(phi, dhi)
    bs = await buyer.open(AgentId("vendor"), buyer_open)
    vs = await vendor.open(AgentId("shopper"), vendor_open)
    exchanged: list[tuple[int, int]] = [(plo, dlo), (phi, dhi)]
    buyer_last, vendor_last = buyer_open, vendor_open
    for _ in range(30):
        await buyer.offer(bs, vendor_last)
        br = await buyer.respond(bs)
        if br.accepted:
            return _pd(vendor_last), exchanged
        if br.counter_terms is None:
            return None, exchanged
        buyer_last = br.counter_terms
        exchanged.append(_pd(buyer_last))

        await vendor.offer(vs, buyer_last)
        vr = await vendor.respond(vs)
        if vr.accepted:
            return _pd(buyer_last), exchanged
        if vr.counter_terms is None:
            return None, exchanged
        vendor_last = vr.counter_terms
        exchanged.append(_pd(vendor_last))
    return None, exchanged


@given(payload=_cfg_and_offers())
@settings(max_examples=200)
def test_determinism(payload: tuple[_Cfg, list[tuple[int, int]]]) -> None:
    """Two identically-built agents fed the same offers produce identical transcripts."""
    cfg, offers = payload

    async def drive(agent: CheckoutFrontier) -> list[tuple[bool, tuple[int, int] | None]]:
        session = await agent.open(AgentId("opp"), _terms(cfg.plo, cfg.dlo))
        out: list[tuple[bool, tuple[int, int] | None]] = []
        for price, deadline in offers:
            await agent.offer(session, _terms(price, deadline))
            resp = await agent.respond(session)
            counter = _pd(resp.counter_terms) if resp.counter_terms is not None else None
            out.append((resp.accepted, counter))
        return out

    assert asyncio.run(drive(_make_vendor(cfg))) == asyncio.run(drive(_make_vendor(cfg)))


@given(cfg=_cfgs())
@settings(max_examples=200)
def test_monotonic_concession_of_fresh_counters(cfg: _Cfg) -> None:
    """A vendor's fresh counters have non-increasing own utility (MCP / Zeuthen).

    Fed its worst bundle repeatedly (utility 0, always below aspiration), the
    vendor never accepts and never echoes, so every counter is fresh and must
    sit on a non-increasing demand schedule.
    """
    vendor = _make_vendor(cfg)
    worst_for_vendor = _terms(cfg.plo, cfg.dlo)

    async def collect() -> list[float]:
        session = await vendor.open(AgentId("opp"), worst_for_vendor)
        utils: list[float] = []
        for _ in range(15):
            await vendor.offer(session, worst_for_vendor)
            resp = await vendor.respond(session)
            if resp.accepted or resp.counter_terms is None:
                break
            utils.append(vendor.utility(resp.counter_terms))
        return utils

    utils = asyncio.run(collect())
    if cfg.floor is not None and cfg.floor > cfg.phi:
        return  # no eligible bundle at all: nothing to assert
    assert len(utils) >= 2
    for earlier, later in zip(utils, utils[1:], strict=False):
        assert later <= earlier + _EPS, f"concession not monotonic: {utils}"


@given(cfg=_cfgs())
@settings(max_examples=200, deadline=None)
def test_selfplay_agreement_never_dominated_by_exchanged_bundle(cfg: _Cfg) -> None:
    """No settled deal is Pareto-dominated by a cap-feasible exchanged bundle.

    The merged ``pareto`` plugin pins this exact property as out of reach for
    pure trade-off concession (its good early bundles are ephemeral). Standing
    quotes make it hold: an agent accepts only the best cap-feasible bundle
    its counterpart ever offered, and echoes recover anything generous that
    aspiration once refused. Feasibility matters: a bundle no wallet can pay
    for is not evidence of a better deal, so dominance is judged over bundles
    inside both parties' declared caps, exactly as the offline trace validator
    judges it.
    """
    buyer = _make_buyer(cfg, max_rounds=25)
    vendor = _make_vendor(cfg, max_rounds=25)
    agreement, exchanged = asyncio.run(
        _self_play(buyer, vendor, (cfg.plo, cfg.phi, cfg.dlo, cfg.dhi))
    )
    if agreement is None:
        return  # breakdown: the dominance claim is about settled deals only

    ap, ad = agreement
    ub_star = buyer.utility(_terms(ap, ad))
    us_star = vendor.utility(_terms(ap, ad))

    def feasible(bundle: tuple[int, int]) -> bool:
        price = bundle[0]
        within_budget = cfg.budget is None or price <= cfg.budget
        above_floor = cfg.floor is None or price >= cfg.floor
        return within_budget and above_floor

    dominators = [
        bundle
        for bundle in exchanged
        if bundle != (ap, ad)
        and feasible(bundle)
        and _dominates(
            buyer.utility(_terms(*bundle)),
            vendor.utility(_terms(*bundle)),
            ub_star,
            us_star,
        )
    ]
    assert not dominators, f"agreement {agreement} dominated by {dominators}; exchanged={exchanged}"


@given(cfg=_cfgs())
@settings(max_examples=100, deadline=None)
def test_selfplay_agreement_respects_caps_and_reservation(cfg: _Cfg) -> None:
    """Settled deals price inside [floor, budget] and clear both reservations."""
    buyer = _make_buyer(cfg, max_rounds=25)
    vendor = _make_vendor(cfg, max_rounds=25)
    agreement, _exchanged = asyncio.run(
        _self_play(buyer, vendor, (cfg.plo, cfg.phi, cfg.dlo, cfg.dhi))
    )
    if agreement is None:
        return

    ap, ad = agreement
    if cfg.budget is not None:
        assert ap <= cfg.budget
    if cfg.floor is not None:
        assert ap >= cfg.floor
    assert buyer.utility(_terms(ap, ad)) >= cfg.reservation - _EPS
    assert vendor.utility(_terms(ap, ad)) >= cfg.reservation - _EPS


def test_disjoint_caps_always_break_down() -> None:
    """A wallet strictly below the vendor's floor can never produce a deal."""
    cfg = _Cfg(
        plo=40,
        phi=60,
        dlo=1,
        dhi=11,
        patience=0.8,
        buyer_wp=0.5,
        seller_wd=0.5,
        reservation=0.0,
        budget=45,
        floor=50,
    )
    buyer = _make_buyer(cfg, max_rounds=25)
    vendor = _make_vendor(cfg, max_rounds=25)
    agreement, _exchanged = asyncio.run(
        _self_play(buyer, vendor, (cfg.plo, cfg.phi, cfg.dlo, cfg.dhi))
    )
    assert agreement is None


def test_flagship_holds_on_the_pinned_pareto_counterexample() -> None:
    """The exact configuration that defeats the merged plugin settles clean here.

    ``test_fsj_tradeoff_does_not_guarantee_pareto_optimality`` pins a
    configuration whose self-play agreement under ``pareto`` IS dominated by
    an exchanged bundle. Under CheckoutFrontier the same preferences settle on
    a deal no exchanged bundle dominates, because the dominating bundle is a
    standing quote and gets accepted instead of expiring.
    """
    cfg = _Cfg(
        plo=10,
        phi=30,
        dlo=1,
        dhi=9,
        patience=0.890625,
        buyer_wp=0.75,
        seller_wd=0.90625,
        reservation=0.0,
        budget=None,
        floor=None,
    )
    buyer = _make_buyer(cfg, max_rounds=25)
    vendor = _make_vendor(cfg, max_rounds=25)
    agreement, exchanged = asyncio.run(
        _self_play(buyer, vendor, (cfg.plo, cfg.phi, cfg.dlo, cfg.dhi))
    )
    assert agreement is not None, "expected this configuration to settle"

    ap, ad = agreement
    ub_star = buyer.utility(_terms(ap, ad))
    us_star = vendor.utility(_terms(ap, ad))
    dominators = [
        bundle
        for bundle in exchanged
        if bundle != (ap, ad)
        and _dominates(
            buyer.utility(_terms(*bundle)),
            vendor.utility(_terms(*bundle)),
            ub_star,
            us_star,
        )
    ]
    assert not dominators, f"agreement {agreement} dominated by {dominators}"
