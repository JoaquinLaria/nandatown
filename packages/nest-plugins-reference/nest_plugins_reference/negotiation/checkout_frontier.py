# SPDX-License-Identifier: Apache-2.0
"""Buyer-side checkout negotiation that never settles below the exchanged frontier.

Bilateral bargaining over *price* and *delivery deadline*, built for the
agentic-checkout setting: a shopping agent negotiating purchase terms with a
vendor under a hard wallet budget, or a vendor quoting under a hard unit-cost
floor. Attributes ride in the existing ``Terms`` model (``terms.price`` plus
``terms.conditions["deadline_days"]``), the same encoding the ``pareto``
plugin uses, so the two interoperate in mixed fleets.

The strategy layers one commerce-shaped rule on top of classical monotonic
concession, and that rule changes the guarantee class:

- Keeney & Raiffa additive multi-attribute utility with per-attribute
  patience: the attribute an agent weights more concedes more slowly, so the
  candidate region widens along a genuine two-dimensional frontier slice
  rather than collapsing to a scalar.
- Rosenschein & Zlotkin monotonic concession (Zeuthen): the own-utility
  aspiration floor ``alpha(t)`` is non-increasing, and every *fresh* counter
  is additionally capped by the agent's previous fresh counter, so demands
  never rise.
- **Best-standing-quote acceptance** (the ANAC "accept the best bid so far"
  family): every bundle the counterparty ever put on the table stays live.
  Once the best of them clears the aspiration floor, the agent either accepts
  it (when it is the offer on the table) or re-proposes it verbatim. Nothing
  generous is ever "ephemeral": the failure mode pinned by
  ``test_fsj_tradeoff_does_not_guarantee_pareto_optimality`` for the merged
  ``pareto`` plugin, where a good early logroll is gone by the time
  aspiration decays, cannot occur here. In self-play, no agreement is
  Pareto-dominated by any budget-feasible bundle either side exchanged.
- **Bounded termination**: the patience horizon is a hard stop, not just an
  aspiration freeze. At or past ``max_rounds`` responds the agent stops
  conceding fresh ground, re-proposes a clearing standing quote at most once
  per session, and otherwise declines to counter (``counter_terms=None``).
  Aspiration is frozen past the horizon, so both a repeated fresh counter
  and a repeated echo would replay the same deterministic exchange forever;
  standing down is the only honest move. In self-play every session ends,
  within roughly ``max_rounds`` rounds plus a small constant, in an
  acceptance or a breakdown the plugin itself initiates, never one imposed
  by a driver's round cap.

Hard money constraints are first-class: a buyer never accepts, echoes, or
proposes a bundle priced above ``budget``; a vendor never goes below
``floor``. Caps are constraints, not utility terms, so a great-looking bundle
the wallet cannot cover is simply ineligible. Cap conflicts end in an honest
breakdown on two paths: an agent whose own cap excludes every feasible bundle
declines to counter immediately, and with mutually exclusive but individually
feasible caps (a wallet strictly below the vendor's floor) neither side can
ever accept, so both stand down at the patience horizon instead of haggling
forever.

Weights are rounded to six decimal places on construction. Trace validators
compare utilities with a ``1e-9`` tolerance; on integer bundle grids,
six-decimal weights make every nonzero utility difference far larger than
that tolerance, so "almost equal" utilities are exactly equal and tie
handling stays consistent between the plugin and the offline validators.

The plugin is Tier-1 deterministic: no wall-clock, no unseeded randomness.
Session ids come from a per-instance counter; the only RNG is the optional
weights-from-seed path, seeded solely from ``(agent_id, seed)``.

Example::

    neg = CheckoutFrontier(
        AgentId("shopper"),
        weights={"price": 0.15, "deadline": 0.85},
        price_range=(40, 120),
        deadline_range=(1, 21),
        side="buyer",
        budget=110,
    )
    session = await neg.open(AgentId("vendor"), Terms(price=Money(amount=40)))
"""

from __future__ import annotations

import random

from nest_core.types import (
    AgentId,
    Agreement,
    Money,
    NegotiationResponse,
    NegotiationSession,
    NegotiationStatus,
    Terms,
)

_EPS = 1e-9
"""Utility comparison tolerance, matching the offline validators' epsilon."""

_UNMOVED_ATTRIBUTE_WEIGHT = 2.0
"""Distance coefficient for the attribute the counterparty held fixed.

A counterparty that concedes price while holding the deadline is signalling
the deadline is what it values. Weighting the *unmoved* attribute more in the
counter-selection distance pulls the reply closer to their deadline, i.e. the
agent concedes on the attribute the opponent indicated (Faratin, Sierra &
Jennings style trade-off, steered by the observed concession direction).
"""


class CheckoutFrontier:
    """Multi-attribute checkout negotiation with best-standing-quote acceptance.

    Each agent knows only its own private configuration: utility weights,
    feasible ranges, reservation level, and its hard money cap (a buyer's
    ``budget`` or a vendor's ``floor``). Negotiation runs over price and
    deadline carried in ``Terms``; all private configuration enters through
    the constructor, never through the protocol methods.

    Example::

        neg = CheckoutFrontier(
            AgentId("vendor"),
            weights={"price": 1.0, "deadline": 0.0},
            price_range=(40, 120),
            deadline_range=(1, 21),
            side="seller",
            floor=60,
        )
        session = await neg.open(AgentId("shopper"), Terms(price=Money(amount=120)))
    """

    def __init__(
        self,
        agent_id: AgentId,
        *,
        weights: dict[str, float] | None = None,
        seed: int | None = None,
        price_range: tuple[int, int] = (40, 120),
        deadline_range: tuple[int, int] = (1, 21),
        side: str = "buyer",
        patience: float = 0.9,
        reservation: float = 0.0,
        budget: int | None = None,
        floor: int | None = None,
        max_rounds: int = 12,
    ) -> None:
        self._agent_id = agent_id
        self._price_range = price_range
        self._deadline_range = deadline_range
        self._side = side
        self._patience = patience
        self._reservation = reservation
        self._budget = budget
        self._floor = floor
        self._max_rounds = max_rounds
        self._weights = self._resolve_weights(weights, seed)
        self._rounds: dict[str, int] = {}
        self._quotes: dict[str, list[tuple[int, int]]] = {}
        self._fresh_bar: dict[str, float] = {}
        self._horizon_echoes: dict[str, int] = {}
        self._session_counter = 0

    async def open(self, partner: AgentId, terms: Terms) -> NegotiationSession:
        """Open a negotiation with own opening terms and a deterministic session id.

        Example::

            session = await neg.open(AgentId("vendor"), Terms(price=Money(amount=40)))
        """
        self._session_counter += 1
        return NegotiationSession(
            id=f"checkout-{self._agent_id}-{self._session_counter}",
            initiator=self._agent_id,
            partner=partner,
            status=NegotiationStatus.OPEN,
            current_terms=terms,
            history=[terms],
        )

    async def offer(self, session: NegotiationSession, terms: Terms) -> None:
        """Record a counterparty offer; it stays live as a standing quote.

        Every priced bundle the counterparty puts on the table is remembered
        for the rest of the session, so a generous early quote can still be
        accepted (or re-proposed) after aspiration decays.

        Example::

            await neg.offer(session, Terms(price=Money(amount=108)))
        """
        session.current_terms = terms
        session.history.append(terms)
        if terms.price is not None:
            bundle = self._clamp(*self._extract(terms))
            self._quotes.setdefault(session.id, []).append(bundle)

    async def respond(self, session: NegotiationSession) -> NegotiationResponse:
        """Accept or echo the best standing quote once it clears aspiration, else concede.

        The decision order is the mechanism:

        1. Among every cap-eligible bundle the counterparty has offered so
           far, find the best by own utility (earliest offer wins exact
           ties). If it clears this round's aspiration floor, accept it when
           it is the offer currently on the table, otherwise re-propose it
           verbatim; when a mutually clearing bundle exists, a counterparty
           running the same rule accepts its own returning quote, so
           convergence costs at most one extra round.
        2. At or past the patience horizon (``max_rounds`` responds), stand
           down: re-propose a clearing standing quote at most once per
           session, never concede fresh ground, and otherwise decline to
           counter. Aspiration is frozen there, so repeating either move
           would replay the same deterministic exchange forever.
        3. Otherwise counter with a fresh bundle from the per-attribute
           allowance box: cap-eligible, at or above aspiration, never above
           the previous fresh counter's own utility, and nearest to the
           counterparty's last quote in a distance weighted toward the
           attribute they signalled they value.
        4. If no cap-eligible bundle exists at all, decline to counter: a
           wallet that covers nothing has no honest bid to make.

        Example::

            resp = await neg.respond(session)
        """
        opponent = session.current_terms
        if opponent is None or opponent.price is None:
            return NegotiationResponse(accepted=True)

        round_index = self._rounds.get(session.id, 0)
        self._rounds[session.id] = round_index + 1
        alpha = self._aspiration(round_index)
        at_horizon = round_index >= self._max_rounds

        quotes = self._quotes.get(session.id, [])
        eligible_quotes = [q for q in quotes if self._eligible(q[0])]
        best = self._first_max(eligible_quotes)
        if best is not None and self._score(*best) >= alpha - _EPS:
            current = self._clamp(*self._extract(opponent))
            if current == best and self._eligible(current[0]):
                return NegotiationResponse(accepted=True)
            if at_horizon:
                if self._horizon_echoes.get(session.id, 0) >= 1:
                    return NegotiationResponse(accepted=False, counter_terms=None)
                self._horizon_echoes[session.id] = 1
            return NegotiationResponse(accepted=False, counter_terms=self._to_terms(best))

        if at_horizon:
            return NegotiationResponse(accepted=False, counter_terms=None)
        counter = self._fresh_counter(session.id, round_index, alpha)
        if counter is None:
            return NegotiationResponse(accepted=False, counter_terms=None)
        return NegotiationResponse(accepted=False, counter_terms=self._to_terms(counter))

    async def close(self, session: NegotiationSession) -> Agreement | None:
        """Return an agreement only for an ``AGREED`` session, else mark it rejected.

        Strict by design: a checkout must not capture payment without both
        signatures. Unlike the reference ``alternating_offers`` close, which
        settles on any non-``None`` current terms, an unagreed session here is
        a breakdown and returns ``None``.

        Example::

            agreement = await neg.close(session)
        """
        if session.status == NegotiationStatus.AGREED:
            return Agreement(
                session_id=session.id,
                terms=session.current_terms or Terms(),
                parties=[session.initiator, session.partner],
            )
        session.status = NegotiationStatus.REJECTED
        return None

    def utility(self, terms: Terms) -> float:
        """Return this agent's additive multi-attribute utility for ``terms``.

        Price and deadline value functions are normalized to ``[0, 1]`` with
        the directional convention of ``side`` (a buyer prefers a low price
        and a short deadline, a seller the opposite) and combined with the
        agent's weights. Inputs are clamped into the feasible ranges.

        Example::

            neg = CheckoutFrontier(
                AgentId("b"),
                weights={"price": 0.5, "deadline": 0.5},
                price_range=(40, 50),
                deadline_range=(1, 5),
                side="buyer",
            )
            neg.utility(Terms(price=Money(amount=40), conditions={"deadline_days": 1}))  # 1.0
        """
        return self._score(*self._extract(terms))

    def _resolve_weights(
        self, weights: dict[str, float] | None, seed: int | None
    ) -> dict[str, float]:
        """Resolve the utility weights: explicit, derived from seed, or equal.

        Explicit weights win. With only a ``seed``, the price weight is drawn
        from a generator seeded solely by ``(agent_id, seed)``, so the same
        identity and seed always produce the same preferences. All weights
        are rounded to six decimals to keep utility gaps on the integer
        bundle grid far above the validators' ``1e-9`` tolerance.
        """
        if weights is not None:
            w_price = round(weights["price"], 6)
            return {"price": w_price, "deadline": round(weights["deadline"], 6)}
        if seed is not None:
            rng = random.Random(f"{self._agent_id}:{seed}")
            w_price = round(rng.uniform(0.1, 0.9), 6)
            return {"price": w_price, "deadline": round(1.0 - w_price, 6)}
        return {"price": 0.5, "deadline": 0.5}

    def _aspiration(self, round_index: int) -> float:
        """Non-increasing own-utility floor (Zeuthen / monotonic concession).

        ``alpha(t) = reservation + (1 - reservation) * patience ** t`` with
        ``t`` capped at ``max_rounds``.
        """
        t = min(round_index, self._max_rounds)
        return self._reservation + (1.0 - self._reservation) * (self._patience**t)

    def _eligible(self, price: int) -> bool:
        """Whether a price respects this agent's own hard money cap.

        Caps are private: a buyer checks only its budget, a vendor only its
        floor. Cross-party feasibility is the validators' job.
        """
        if self._side == "buyer":
            return self._budget is None or price <= self._budget
        return self._floor is None or price >= self._floor

    def _first_max(self, bundles: list[tuple[int, int]]) -> tuple[int, int] | None:
        """Best bundle by own utility; the earliest offer wins exact ties.

        The earliest-tie rule is load-bearing: under the counterparty's own
        monotonic concession, the earliest of equally-good-for-us quotes is
        the one best for *them*, so echoing it can never trade away their
        surplus for nothing.
        """
        best: tuple[int, int] | None = None
        best_u = float("-inf")
        for bundle in bundles:
            u = self._score(*bundle)
            if u > best_u + _EPS:
                best, best_u = bundle, u
        return best

    def _fresh_counter(self, sid: str, round_index: int, alpha: float) -> tuple[int, int] | None:
        """Pick a fresh concession bundle, or ``None`` when no eligible bundle exists.

        Candidates come from the per-attribute allowance box around the
        agent's ideal corner, filtered to the utility band between this
        round's aspiration and the previous fresh counter (demands never
        rise). If the box is empty the band widens to the whole grid; if the
        band itself is unreachable the agent restates its best eligible
        bundle. Selection is by direction-weighted distance to the
        counterparty's last quote with explicit integer tie-breaks, so every
        step is deterministic.
        """
        bar = self._fresh_bar.get(sid, 1.0)
        eligible = [b for b in self._grid() if self._eligible(b[0])]
        if not eligible:
            return None

        in_band = [b for b in eligible if alpha - _EPS <= self._score(*b) <= bar + _EPS]
        box = set(self._allowance_box(round_index))
        candidates = [b for b in in_band if b in box] or in_band
        if not candidates:
            candidates = [max(eligible, key=lambda b: (self._score(*b), -b[0], -b[1]))]

        target = self._last_quote(sid)
        c_price, c_deadline = self._direction_weights(sid)
        plo, phi = self._price_range
        dlo, dhi = self._deadline_range
        p_span, d_span = phi - plo, dhi - dlo

        def distance(b: tuple[int, int]) -> tuple[float, int, int]:
            dp = (b[0] - target[0]) / p_span
            dd = (b[1] - target[1]) / d_span
            return (c_price * dp * dp + c_deadline * dd * dd, b[0], b[1])

        chosen = min(candidates, key=distance)
        self._fresh_bar[sid] = self._score(*chosen)
        return chosen

    def _allowance_box(self, round_index: int) -> list[tuple[int, int]]:
        """Bundles within this round's per-attribute concession allowance.

        Each attribute may move away from the agent's ideal corner by
        ``span * (1 - patience_a ** t)`` where ``patience_a`` grows with the
        attribute's weight: the agent concedes more slowly on what it values.
        The box therefore widens at different per-attribute rates, tracing a
        two-dimensional frontier slice instead of a scalar schedule.
        """
        plo, phi = self._price_range
        dlo, dhi = self._deadline_range
        t = min(round_index, self._max_rounds)
        allow_p = (phi - plo) * (1.0 - self._attribute_patience("price") ** t)
        allow_d = (dhi - dlo) * (1.0 - self._attribute_patience("deadline") ** t)
        if self._side == "buyer":
            prices = range(plo, min(phi, plo + int(allow_p)) + 1)
            deadlines = range(dlo, min(dhi, dlo + int(allow_d)) + 1)
        else:
            prices = range(max(plo, phi - int(allow_p)), phi + 1)
            deadlines = range(max(dlo, dhi - int(allow_d)), dhi + 1)
        return [(p, d) for p in prices for d in deadlines]

    def _attribute_patience(self, attribute: str) -> float:
        """Per-attribute concession discount, higher for the heavier weight."""
        return self._patience + (1.0 - self._patience) * self._weights[attribute]

    def _direction_weights(self, sid: str) -> tuple[float, float]:
        """Distance coefficients steering concession toward the opponent's signal.

        Comparing the counterparty's last two quotes, the attribute they held
        (smaller normalized move) is the one they value; it gets the heavier
        coefficient so the fresh counter lands closer to them on it.
        """
        quotes = self._quotes.get(sid, [])
        if len(quotes) < 2:
            return 1.0, 1.0
        plo, phi = self._price_range
        dlo, dhi = self._deadline_range
        (pp, pd), (lp, ld) = quotes[-2], quotes[-1]
        moved_p = abs(lp - pp) / (phi - plo)
        moved_d = abs(ld - pd) / (dhi - dlo)
        if moved_p < moved_d - _EPS:
            return _UNMOVED_ATTRIBUTE_WEIGHT, 1.0
        if moved_d < moved_p - _EPS:
            return 1.0, _UNMOVED_ATTRIBUTE_WEIGHT
        return 1.0, 1.0

    def _last_quote(self, sid: str) -> tuple[int, int]:
        """The counterparty's most recent bundle, or own ideal corner before any."""
        quotes = self._quotes.get(sid, [])
        if quotes:
            return quotes[-1]
        return self._ideal()

    def _ideal(self) -> tuple[int, int]:
        """This side's utility-1.0 corner of the feasible rectangle."""
        plo, phi = self._price_range
        dlo, dhi = self._deadline_range
        if self._side == "buyer":
            return plo, dlo
        return phi, dhi

    def _grid(self) -> list[tuple[int, int]]:
        """Deterministic enumeration of every feasible (price, deadline) bundle."""
        plo, phi = self._price_range
        dlo, dhi = self._deadline_range
        return [(p, d) for p in range(plo, phi + 1) for d in range(dlo, dhi + 1)]

    def _extract(self, terms: Terms) -> tuple[int, int]:
        """Pull the raw (price, deadline) pair out of ``terms`` as ints."""
        plo, _ = self._price_range
        dlo, _ = self._deadline_range
        price = terms.price.amount if terms.price is not None else plo
        deadline = int(terms.conditions.get("deadline_days", dlo))
        return price, deadline

    def _clamp(self, price: int, deadline: int) -> tuple[int, int]:
        """Clamp a (price, deadline) pair into the feasible ranges."""
        plo, phi = self._price_range
        dlo, dhi = self._deadline_range
        return max(plo, min(phi, price)), max(dlo, min(dhi, deadline))

    def _score(self, price: int, deadline: int) -> float:
        """Additive MAUT score for a (price, deadline) pair after clamping."""
        plo, phi = self._price_range
        dlo, dhi = self._deadline_range
        p, d = self._clamp(price, deadline)
        if self._side == "buyer":
            f_price = (phi - p) / (phi - plo)
            f_deadline = (dhi - d) / (dhi - dlo)
        else:
            f_price = (p - plo) / (phi - plo)
            f_deadline = (d - dlo) / (dhi - dlo)
        return self._weights["price"] * f_price + self._weights["deadline"] * f_deadline

    def _to_terms(self, bundle: tuple[int, int]) -> Terms:
        """Wrap a (price, deadline) bundle in the shared ``Terms`` encoding."""
        return Terms(price=Money(amount=bundle[0]), conditions={"deadline_days": bundle[1]})
