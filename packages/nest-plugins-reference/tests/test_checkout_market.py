# SPDX-License-Identifier: Apache-2.0
"""Validator unit tests and the end-to-end discrimination gate for the checkout market.

Three layers:

1. **Validator direct-call**: hand-built event lists drive each branch of
   ``validate_checkout_pareto_efficient`` (dominated deal fails, a clean
   frontier passes, a cap-infeasible dominator does not count, a no-deal is
   not a dominance failure, an empty trace trips the vacuous guard) and of
   ``validate_checkout_budget_and_floor`` (over-budget quote, over-budget
   deal, below-floor deal).
2. **End-to-end discrimination** (the core deliverable): boot the real
   ``checkout_market`` scenario through ``ScenarioRunner`` under seeds 42, 7,
   1337. With ``negotiation: checkout_frontier`` every validator PASSES; with
   ``negotiation: alternating_offers`` (deadline-blind, timeout-driven)
   ``validate_checkout_pareto_efficient`` FAILS on a deal dominated by the
   vendor's early rush quote. The layer is overridden in-test; the shipped
   YAML is untouched.
3. **Byte-determinism**: the same seed writes a byte-identical trace twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path
from typing import Any

import pytest
from nest_core.plugins import PluginRegistry
from nest_core.runner import ScenarioRunner
from nest_core.scenario import ScenarioConfig
from nest_core.validators import (
    ValidationResult,
    validate_checkout_budget_and_floor,
    validate_checkout_pareto_efficient,
    validate_trace,
)

SCENARIO_PATH = Path(__file__).resolve().parents[3] / "scenarios" / "checkout_market.yaml"


def _send(msg: str) -> dict[str, Any]:
    """A minimal send event carrying a colon-delimited checkout frame."""
    return {"kind": "send", "msg": msg, "agent": "driver", "to": "peer"}


def _cfg_frames(budget: int = 120, floor: int = 60, pair: int = 0) -> list[dict[str, Any]]:
    """Config frames for one deadline-loving buyer and one margin-only vendor."""
    return [
        _send(f"checkoutcfg:shopper-{pair}:buyer:0.100000:0.900000:40:120:1:21:0.000000:{budget}"),
        _send(f"checkoutcfg:vendor-{pair}:seller:1.000000:0.000000:40:120:1:21:0.000000:{floor}"),
    ]


# Validator direct-call tests: Pareto efficiency


def test_pareto_validator_flags_dominated_deal() -> None:
    """A deal dominated by a cap-feasible exchanged bundle is caught and named."""
    events = [
        *_cfg_frames(),
        _send("quote:pair-0:vendor-0:seller:2:108:1"),  # the rush logroll, feasible
        _send("quote:pair-0:shopper-0:buyer:2:40:1"),
        _send("quote:pair-0:vendor-0:seller:9:84:19"),  # late, slow lane
        _send("deal:pair-0:84:19:shopper-0"),  # ... and it is what gets agreed
    ]
    result = validate_checkout_pareto_efficient(events)[0]
    assert not result.passed
    assert "dominated by" in result.detail
    assert "(108,1)" in result.detail


def test_pareto_validator_passes_clean_frontier() -> None:
    """A session whose deal is the frontier bundle passes."""
    events = [
        *_cfg_frames(pair=1),
        _send("quote:pair-1:vendor-1:seller:2:108:1"),
        _send("quote:pair-1:shopper-1:buyer:2:40:1"),
        _send("deal:pair-1:108:1:shopper-1"),
    ]
    result = validate_checkout_pareto_efficient(events)[0]
    assert result.passed, result.detail


def test_pareto_validator_ignores_cap_infeasible_dominator() -> None:
    """A dominating bundle nobody could transact is not evidence of a better deal.

    The rush quote at 115 dominates the closed deal in utility space, but the
    buyer's wallet is 110: no checkout could ever clear it, so the validator
    must not fail the session over it.
    """
    events = [
        *_cfg_frames(budget=110, pair=2),
        _send("quote:pair-2:vendor-2:seller:2:115:1"),  # dominates, but over budget
        _send("quote:pair-2:shopper-2:buyer:2:40:1"),
        _send("quote:pair-2:vendor-2:seller:9:84:19"),
        _send("deal:pair-2:84:19:shopper-2"),
    ]
    result = validate_checkout_pareto_efficient(events)[0]
    assert result.passed, result.detail


def test_pareto_validator_ignores_no_deal_sessions() -> None:
    """A no-deal is a legitimate breakdown, not a dominance failure."""
    events = [
        *_cfg_frames(pair=1),
        _send("quote:pair-1:vendor-1:seller:2:108:1"),
        _send("quote:pair-1:shopper-1:buyer:2:40:1"),
        _send("deal:pair-1:108:1:shopper-1"),
        # A second session that simply broke down.
        *_cfg_frames(budget=45, floor=50, pair=2),
        _send("quote:pair-2:vendor-2:seller:1:120:21"),
        _send("quote:pair-2:shopper-2:buyer:1:40:1"),
        _send("no_deal:pair-2:10"),
    ]
    result = validate_checkout_pareto_efficient(events)[0]
    assert result.passed, result.detail
    assert "pair-2" not in result.detail


def test_pareto_validator_fails_vacuous_trace() -> None:
    """A trace with no scorable deal must fail, not silently pass."""
    result = validate_checkout_pareto_efficient([])[0]
    assert not result.passed
    assert "no checkout negotiation" in result.detail


# Validator direct-call tests: budget and floor discipline


def test_budget_validator_flags_quote_beyond_wallet() -> None:
    """A buyer bidding beyond its own budget is broken even if the deal lands under."""
    events = [
        *_cfg_frames(budget=100),
        _send("quote:pair-0:vendor-0:seller:1:110:1"),
        _send("quote:pair-0:shopper-0:buyer:1:105:1"),  # exceeds its own wallet
        _send("deal:pair-0:95:1:shopper-0"),
    ]
    result = validate_checkout_budget_and_floor(events)[0]
    assert not result.passed
    assert "exceeds budget" in result.detail


def test_budget_validator_flags_deal_outside_caps() -> None:
    """A closed deal must satisfy both parties' caps."""
    over_budget = [
        *_cfg_frames(budget=100),
        _send("quote:pair-0:vendor-0:seller:1:105:1"),
        _send("quote:pair-0:shopper-0:buyer:1:40:1"),
        _send("deal:pair-0:105:1:shopper-0"),
    ]
    result = validate_checkout_budget_and_floor(over_budget)[0]
    assert not result.passed
    assert "exceeds budget" in result.detail

    below_floor = [
        *_cfg_frames(floor=90, pair=1),
        _send("quote:pair-1:vendor-1:seller:1:120:21"),
        _send("quote:pair-1:shopper-1:buyer:1:85:1"),
        _send("deal:pair-1:85:1:vendor-1"),
    ]
    result = validate_checkout_budget_and_floor(below_floor)[0]
    assert not result.passed
    assert "below floor" in result.detail


def test_budget_validator_allows_asks_below_counterparty_floor() -> None:
    """A buyer may open beneath the vendor's floor: an ask is not a transaction."""
    events = [
        *_cfg_frames(floor=60),
        _send("quote:pair-0:vendor-0:seller:1:120:21"),
        _send("quote:pair-0:shopper-0:buyer:1:40:1"),  # below the vendor's floor
        _send("deal:pair-0:108:1:shopper-0"),
    ]
    result = validate_checkout_budget_and_floor(events)[0]
    assert result.passed, result.detail


def test_budget_validator_fails_vacuous_trace() -> None:
    """A trace with no capped session must fail, not silently pass."""
    result = validate_checkout_budget_and_floor([])[0]
    assert not result.passed
    assert "no checkout negotiation" in result.detail


# End-to-end discrimination gate


def _run_scenario(seed: int, negotiation: str, trace_path: Path) -> list[ValidationResult]:
    """Run the checkout scenario with a chosen negotiation layer; validate its trace."""
    config = ScenarioConfig.from_yaml(str(SCENARIO_PATH))
    config.seed = seed
    config.layers.negotiation = negotiation
    config.output.trace = str(trace_path)
    runner = ScenarioRunner(config, registry=PluginRegistry())
    asyncio.run(runner.run())
    return validate_trace(trace_path, "checkout_market")


@pytest.mark.parametrize("seed", [42, 7, 1337])
def test_checkout_frontier_passes_all_validators(seed: int) -> None:
    """The plugin under test reaches only non-dominated, cap-disciplined deals."""
    if not SCENARIO_PATH.exists():
        pytest.skip(f"scenario not found at {SCENARIO_PATH}")
    with tempfile.TemporaryDirectory() as tmp:
        results = _run_scenario(seed, "checkout_frontier", Path(tmp) / f"cf_{seed}.jsonl")
    assert results, "expected validators to run"
    assert all(r.passed for r in results), [(r.name, r.detail) for r in results]


@pytest.mark.parametrize("seed", [42, 7, 1337])
def test_alternating_offers_fails_pareto_validator(seed: int) -> None:
    """The deadline-blind reference plugin is caught: it closes a dominated deal."""
    if not SCENARIO_PATH.exists():
        pytest.skip(f"scenario not found at {SCENARIO_PATH}")
    with tempfile.TemporaryDirectory() as tmp:
        results = _run_scenario(seed, "alternating_offers", Path(tmp) / f"ao_{seed}.jsonl")
    pareto = next(r for r in results if r.name == "checkout_pareto_efficient")
    assert not pareto.passed, f"seed={seed}: alternating_offers should be caught"
    assert "dominated by" in pareto.detail


def test_same_seed_writes_byte_identical_trace() -> None:
    """Tier-1 determinism, end to end: one seed, two runs, identical bytes."""
    if not SCENARIO_PATH.exists():
        pytest.skip(f"scenario not found at {SCENARIO_PATH}")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory() as tmp:
        first = Path(tmp) / "first.jsonl"
        second = Path(tmp) / "second.jsonl"
        _run_scenario(42, "checkout_frontier", first)
        _run_scenario(42, "checkout_frontier", second)
        assert digest(first) == digest(second)
