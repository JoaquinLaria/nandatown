# SPDX-License-Identifier: Apache-2.0
"""Checkout market scenario: buyer-side evaluation under budget caps.

Ten agentic checkouts, each between a *scripted* margin-only vendor and the
*configured* negotiation plugin in the buyer seat. This inverts the merged
``multi_attribute_market`` evaluation (scripted buyer, configured seller):
here the shopping agent is the artifact under test, because in agent-to-retail
commerce the buyer is the deployed software and the vendor side is a quoting
engine.

The vendor is deliberately **margin-only** (``w_deadline = 0``): fulfillment
slack costs an in-stock vendor nothing, so expedited shipping is free for it
to give away. Its quote schedule concedes price monotonically toward a
clearance target, offering the **expedited lane twice, early** (the rush
rounds) and a slow lane otherwise. A deadline-aware, budget-disciplined buyer
grabs the rush logroll the moment it fits both its aspiration and its wallet;
a deadline-blind, timeout-driven buyer (the reference ``alternating_offers``,
which never reads ``conditions['deadline_days']`` and only accepts at its
round limit) holds out and closes late on a cheaper slow-lane bundle that the
early rush quote Pareto-dominates for *both* sides.

Why the dominance is seed-independent: the buyer's deadline weight is drawn
from ``[0.85, 0.95]``, so the rush bundle beats any slow-lane bundle for the
buyer by at least ``0.85 * (slow - rush) / d_span - w_p * price_gap / p_span``
which is positive for every draw here, and the vendor is margin-only, so the
rush round's higher price beats every later round's lower price outright.
The rush price also sits at or below every drawn budget by construction. The
buyer's aspiration ``patience ** t`` is 0.9 in round 2 and 0.81 in round 3;
the rush bundle's buyer utility is at least 0.8725 for every drawn weight, so
round 3 guarantees acceptance even when round 2's price momentarily exceeds a
low budget draw (the cap visibly bites, then the cheaper rush re-offer lands).

Every exchanged bundle, both parties' private utility parameters, and their
hard money caps are written to the trace so the offline validators can
reconstruct utilities, compute cap-feasible dominance, and audit budget and
floor discipline. Frame grammar (floats to 6 dp for byte-determinism; the
cap is the buyer's budget or the vendor's floor)::

    checkoutcfg:<agent>:<side>:<w_price>:<w_deadline>:<plo>:<phi>:<dlo>:<dhi>:<reservation>:<cap>
    quote:<sid>:<agent>:<side>:<round>:<price>:<deadline>
    deal:<sid>:<price>:<deadline>:<accepting_agent>
    no_deal:<sid>:<rounds>

The vendor drives the whole exchange synchronously inside ``on_start`` (no
cross-agent event interleaving to go non-deterministic), emitting authentic
``ctx.send`` frames. All randomness is drawn in the factory, before any agent
runs, from a generator seeded only by ``(config.seed, pair)`` with a fixed
draw order (buyer weight, budget, floor, then the slow-lane deadlines); that
order is load-bearing for byte-identical replays.

Example::

    from nest_core.runner import ScenarioRunner
    runner = ScenarioRunner(ScenarioConfig.from_yaml("scenarios/checkout_market.yaml"))
    await runner.run()
"""

from __future__ import annotations

import inspect
import random
from typing import Any

from nest_core.scenario import ScenarioConfig
from nest_core.sim.agent import AgentContext, StateMachineAgent
from nest_core.types import AgentId, Money, NegotiationStatus, Terms

N_PAIRS = 10
"""Number of independent shopper-vendor checkouts."""

PRICE_RANGE = (40, 120)
"""Feasible price interval (credits), shared by every pair."""

DEADLINE_RANGE = (1, 21)
"""Feasible delivery-deadline interval (days), shared by every pair."""

PATIENCE = 0.9
"""Concession discount per round for the buyer plugin under test."""

RESERVATION = 0.0
"""Walk-away utility floor for every agent."""

MAX_ROUNDS = 10
"""Maximum bargaining rounds before a checkout is declared a no-deal."""

VENDOR_WEIGHTS = {"price": 1.0, "deadline": 0.0}
"""The scripted vendor: pure margin, indifferent to fulfillment slack."""

BUYER_WEIGHT_LOW = 0.85
"""Lower bound of the shopper's dominant (deadline) weight."""

BUYER_WEIGHT_HIGH = 0.95
"""Upper bound of the shopper's dominant (deadline) weight."""

BUDGET_RANGE = (110, 120)
"""Interval the shopper's hard wallet budget is drawn from.

The lower bound sits between the round-2 rush price (112) and the round-3
rush price (108), so some seeds see the budget refuse round 2's rush quote
before round 3's cheaper re-offer lands: the cap visibly bites in-trace.
"""

FLOOR_RANGE = (60, 75)
"""Interval the vendor's unit-cost floor is drawn from (below every quote)."""

RUSH_ROUNDS = (2, 3)
"""Rounds where the vendor offers the expedited lane, the integrative gift.

Two rounds, not one: the buyer's aspiration ``patience ** (r - 1)`` is 0.9 at
round 2, which the lowest drawn deadline weight (0.85) does not clear at the
round-2 rush price, and 0.81 at round 3, which every drawn weight clears.
Offering the rush lane in both rounds also covers the budget-bite case where
round 2's price exceeds a low budget draw.
"""

CLEARANCE_PRICE = 80
"""Price the vendor's schedule descends toward by the final round."""

SLOW_LANE_SPREAD = 4
"""Non-rush deadlines are drawn from the top of the range, this far down.

Late rounds carry a slow lane, so whichever late bundle a timeout-driven
buyer closes on is dominated by the early rush quote: strictly better for
the margin-only vendor (higher price) and strictly better for every drawn
deadline-sensitive shopper (rush delivery outweighs the price gap).
"""


def _checkoutcfg_frame(
    agent_id: AgentId,
    side: str,
    w_price: float,
    w_deadline: float,
    plo: int,
    phi: int,
    dlo: int,
    dhi: int,
    reservation: float,
    cap: int,
) -> str:
    """Build the once-per-agent frame revealing private utility parameters and cap."""
    return (
        f"checkoutcfg:{agent_id}:{side}:{w_price:.6f}:{w_deadline:.6f}"
        f":{plo}:{phi}:{dlo}:{dhi}:{reservation:.6f}:{cap}"
    )


def _quote_frame(
    sid: str, agent_id: AgentId, side: str, rnd: int, price: int, deadline: int
) -> str:
    """Build the frame recording one quoted (price, deadline) bundle."""
    return f"quote:{sid}:{agent_id}:{side}:{rnd}:{price}:{deadline}"


def _deal_frame(sid: str, price: int, deadline: int, accepting: AgentId) -> str:
    """Build the frame recording an accepted checkout."""
    return f"deal:{sid}:{price}:{deadline}:{accepting}"


def _no_deal_frame(sid: str, rounds: int) -> str:
    """Build the frame recording a checkout that broke down."""
    return f"no_deal:{sid}:{rounds}"


def _terms_pd(terms: Terms) -> tuple[int, int]:
    """Extract the (price, deadline) integer pair carried by ``terms``."""
    price = terms.price.amount if terms.price is not None else 0
    deadline = int(terms.conditions.get("deadline_days", 0))
    return price, deadline


def _construct_negotiator(neg_cls: Any, agent_id: AgentId, candidate: dict[str, Any]) -> Any:
    """Instantiate any Negotiation plugin, passing only the kwargs it accepts.

    The Negotiation protocol does not define ``__init__``, so plugins have
    different constructor signatures: ``CheckoutFrontier`` wants the full
    checkout config while the reference ``AlternatingOffers`` takes only
    ``patience``. Forwarding just the kwargs that name real parameters lets
    the YAML swap the ``negotiation:`` layer without a ``TypeError``.
    Deliberately duplicated from the merged ``multi_attribute_market``
    builder rather than imported: that module is a merged submission the
    charter forbids modifying, and its helper is private.

    Example::

        neg = _construct_negotiator(CheckoutFrontier, AgentId("shopper-0"), candidate)
    """
    params = inspect.signature(neg_cls.__init__).parameters
    accepted = {key: value for key, value in candidate.items() if key in params}
    return neg_cls(agent_id, **accepted)


def _vendor_schedule(
    rng: random.Random, bounds: tuple[int, int, int, int], max_rounds: int
) -> list[tuple[int, int]]:
    """Build the vendor's deterministic, price-monotonic quote schedule.

    Price descends from the top of the range toward the clearance target (the
    margin-only vendor conceding on the only issue it values). The deadline is
    the expedited lane on the rush rounds and a seeded slow lane otherwise.
    The vendor gives the rush lane away early because slack costs it nothing.

    Example::

        schedule = _vendor_schedule(random.Random("42:0"), (40, 120, 1, 21), 10)
    """
    _plo, phi, dlo, dhi = bounds
    slow_lo = max(dlo, dhi - SLOW_LANE_SPREAD)
    schedule: list[tuple[int, int]] = []
    for r in range(1, max_rounds + 1):
        price = round(phi - (phi - CLEARANCE_PRICE) * r / max_rounds)
        deadline = dlo if r in RUSH_ROUNDS else rng.randint(slow_lo, dhi)
        schedule.append((price, deadline))
    return schedule


class CheckoutShopperAgent(StateMachineAgent):
    """Passive counterparty node: the buyer plugin is driven by the vendor's loop.

    The shopper agent does no work itself; it only needs to be a real
    addressable node so the vendor's ``ctx.send`` frames have a destination.

    Example::

        agent = CheckoutShopperAgent(AgentId("shopper-0"))
    """

    def __init__(self, agent_id: AgentId) -> None:
        self._id = agent_id


class CheckoutVendorAgent(StateMachineAgent):
    """The scripted vendor that drives one checkout against the buyer plugin.

    Holds the buyer's configured negotiation-plugin instance and a precomputed
    price-monotonic quote schedule, and runs the full exchange in
    ``on_start``: it presents each scheduled quote to the buyer's ``respond``
    and records every bundle. The vendor never reacts to the buyer's
    counteroffers (it follows its script), which keeps the evaluation a clean
    test of the buyer's ``respond`` rather than an echo loop. The session is
    closed through the plugin's own ``close`` on both the agreement and the
    no-deal path, so the protocol surface is exercised end to end.

    Example::

        agent = CheckoutVendorAgent(
            AgentId("vendor-0"), AgentId("shopper-0"), "pair-0", buyer_neg,
            buyer_weights, (40, 120, 1, 21), 0.0, 115, 70, schedule, 10,
        )
    """

    def __init__(
        self,
        vendor_id: AgentId,
        shopper_id: AgentId,
        sid: str,
        buyer_neg: Any,
        buyer_weights: dict[str, float],
        bounds: tuple[int, int, int, int],
        reservation: float,
        budget: int,
        floor: int,
        schedule: list[tuple[int, int]],
        max_rounds: int,
    ) -> None:
        self._vendor_id = vendor_id
        self._shopper_id = shopper_id
        self._sid = sid
        self._buyer_neg = buyer_neg
        self._buyer_weights = buyer_weights
        self._bounds = bounds
        self._reservation = reservation
        self._budget = budget
        self._floor = floor
        self._schedule = schedule
        self._max_rounds = max_rounds

    async def on_start(self, ctx: AgentContext) -> None:
        """Reveal both configurations, then walk the quote schedule against the buyer.

        The vendor concedes price round by round; the buyer plugin evaluates
        each quote. The checkout ends the moment the buyer accepts (deal) or
        the schedule is exhausted (no deal). Both endings go through the
        plugin's ``close``.

        Example::

            await agent.on_start(ctx)
        """
        plo, phi, dlo, dhi = self._bounds

        await ctx.send(
            self._shopper_id,
            _checkoutcfg_frame(
                self._shopper_id,
                "buyer",
                self._buyer_weights["price"],
                self._buyer_weights["deadline"],
                plo,
                phi,
                dlo,
                dhi,
                self._reservation,
                self._budget,
            ).encode(),
        )
        await ctx.send(
            self._shopper_id,
            _checkoutcfg_frame(
                self._vendor_id,
                "seller",
                VENDOR_WEIGHTS["price"],
                VENDOR_WEIGHTS["deadline"],
                plo,
                phi,
                dlo,
                dhi,
                self._reservation,
                self._floor,
            ).encode(),
        )

        # The buyer opens from its best-for-self position; the vendor's
        # scripted quotes then drive the exchange.
        buyer_opener = Terms(price=Money(amount=plo), conditions={"deadline_days": dlo})
        session = await self._buyer_neg.open(self._vendor_id, buyer_opener)

        for rnd, (price, deadline) in enumerate(self._schedule, start=1):
            vendor_quote = Terms(price=Money(amount=price), conditions={"deadline_days": deadline})
            await self._buyer_neg.offer(session, vendor_quote)
            resp = await self._buyer_neg.respond(session)

            quote = _quote_frame(self._sid, self._vendor_id, "seller", rnd, price, deadline)
            await ctx.send(self._shopper_id, quote.encode())

            if resp.accepted:
                session.status = NegotiationStatus.AGREED
                agreement = await self._buyer_neg.close(session)
                terms = agreement.terms if agreement is not None else vendor_quote
                dp, dd = _terms_pd(terms)
                await ctx.send(
                    self._shopper_id, _deal_frame(self._sid, dp, dd, self._shopper_id).encode()
                )
                return

            if resp.counter_terms is not None:
                cp, cd = _terms_pd(resp.counter_terms)
                counter = _quote_frame(self._sid, self._shopper_id, "buyer", rnd, cp, cd)
                await ctx.send(self._shopper_id, counter.encode())

        await self._buyer_neg.close(session)
        await ctx.send(self._shopper_id, _no_deal_frame(self._sid, self._max_rounds).encode())


def checkout_market_factory(config: ScenarioConfig, plugins: dict[str, Any]) -> dict[AgentId, Any]:
    """Build ten pairs: a scripted margin-only vendor against the configured buyer.

    The population comes from :data:`N_PAIRS`; the YAML ``agents`` block is
    descriptive documentation of the same shape (the sibling market
    scenario's convention). Each pair's buyer weights, money caps, and
    vendor schedule are derived in the factory from a generator seeded only
    by ``(config.seed, pair_index)`` with a fixed draw order, before any
    agent runs. The buyer is the
    configured ``negotiation`` plugin, instantiated through
    :func:`_construct_negotiator` so swapping the layer in the YAML swaps the
    strategy under test; its instance is also injected via ``_agent_plugins``
    so the shopper node's ``ctx`` carries it.

    Example::

        agents = checkout_market_factory(config, plugins)
    """
    neg_cls = plugins["negotiation"]
    plo, phi = PRICE_RANGE
    dlo, dhi = DEADLINE_RANGE
    bounds = (plo, phi, dlo, dhi)

    agents: dict[AgentId, Any] = {}
    overrides: dict[AgentId, dict[str, Any]] = {}

    for i in range(N_PAIRS):
        shopper_id = AgentId(f"shopper-{i}")
        vendor_id = AgentId(f"vendor-{i}")
        sid = f"pair-{i}"

        rng = random.Random(f"{config.seed}:{i}")
        w_deadline = round(rng.uniform(BUYER_WEIGHT_LOW, BUYER_WEIGHT_HIGH), 6)
        buyer_weights = {"price": round(1.0 - w_deadline, 6), "deadline": w_deadline}
        budget = rng.randint(*BUDGET_RANGE)
        floor = rng.randint(*FLOOR_RANGE)
        schedule = _vendor_schedule(rng, bounds, MAX_ROUNDS)

        buyer_candidate: dict[str, Any] = {
            "weights": buyer_weights,
            "price_range": PRICE_RANGE,
            "deadline_range": DEADLINE_RANGE,
            "side": "buyer",
            "patience": PATIENCE,
            "reservation": RESERVATION,
            "budget": budget,
            "max_rounds": MAX_ROUNDS,
        }
        buyer_neg = _construct_negotiator(neg_cls, shopper_id, buyer_candidate)

        agents[vendor_id] = CheckoutVendorAgent(
            vendor_id,
            shopper_id,
            sid,
            buyer_neg,
            buyer_weights,
            bounds,
            RESERVATION,
            budget,
            floor,
            schedule,
            MAX_ROUNDS,
        )
        agents[shopper_id] = CheckoutShopperAgent(shopper_id)
        overrides[shopper_id] = {"negotiation": buyer_neg}

    plugins["_agent_plugins"] = overrides
    return agents
