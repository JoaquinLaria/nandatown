# Experiment: message_drop 0.0 → 0.3 on `marketplace`

**Scenario:** `marketplace` (built-in, 50 buyers / 50 sellers, contract-net + alternating-offers, seed 42, 10000 ticks)

**Change:** `failures.message_drop` from `0.0` to `0.3` (see `marketplace_drop30.yaml`). Everything else is identical to the built-in scenario.

## Hypothesis (before running)

A 30% independent per-message drop rate should roughly halve throughput (each buy round needs a request *and* a response to survive, so ~`0.7*0.7 = 49%` of rounds succeed), pushing `delivery_rate` toward ~0.5 and `deal_rate` down proportionally, while total message *volume* stays close to the baseline (agents keep trying).

## Result

| metric | baseline (0.0) | drop30 (0.3) |
|---|---|---|
| message_count | 2000 | 299 |
| correlation_ids | 1000 | 174 |
| delivery_rate | 1.000 | 0.718 |
| deal_rate | 0.532 | 0.439 |
| dropped_count | 0 | 49 |
| validator `marketplace_all_responded` | PASS | **FAIL** (40 unanswered buy requests) |

Full reports: `reports/marketplace_baseline.html`, `reports/marketplace_drop30.html`. Raw traces: `traces/marketplace.jsonl`, `traces/marketplace_drop30.jsonl`.

## What surprised me, and how I investigated it

Message *volume* didn't stay roughly constant like I expected — it collapsed from 2000 to 299 (an ~85% drop, far more than the 30% drop rate alone would explain), and `marketplace_all_responded` started failing outright.

I read `packages/nest-core/nest_core/scenarios_builtin/marketplace.py` (`BuyerAgent.on_message`, lines ~95-159). The buyer only sends its *next* round's buy request from inside the handler for the *previous* round's response — there is no timeout or retry logic anywhere in the loop. So if either a buyer's request or a seller's response is dropped, that buyer's negotiation thread stalls permanently for the rest of the run; it never gets a second chance. With up to 10 sequential rounds per buyer, a single drop early in a buyer's sequence removes all of its remaining rounds too, which compounds far faster than the flat 30% drop rate. That is what the 40 unanswered requests and the `marketplace_all_responded` failure are actually showing — not delivery noise, but a protocol with no failure-recovery path.

## Takeaway

`failures.message_drop` in this scenario doesn't model "noisy but self-healing" comms — it models permanent, silent loss for whichever agent lock-steps on a response. The gap it exposes: `contract_net`/`alternating_offers` here have no retry/timeout layer, so any lossy transport silently amputates the affected agents' remaining task instead of degrading gracefully.

## Reproduce

```bash
uv sync
uv run nest run marketplace                                    # baseline
uv run nest run experiments/marketplace_drop30.yaml             # experiment
uv run nest inspect traces/marketplace_drop30.jsonl
uv run nest report --output reports/marketplace_drop30.html traces/marketplace_drop30.jsonl
uv run python -c "from pathlib import Path; from nest_core.validators import validate_trace
for r in validate_trace(Path('traces/marketplace_drop30.jsonl'), 'marketplace'): print(r)"
```

## AI / tool use disclosure

Used Claude Code (Sonnet 5) to: run `nest doctor`/`nest run`/`nest inspect`/`nest report`, copy and edit the scenario YAML, grep/read the marketplace scenario source to explain the observed collapse in message volume, and draft this README. All commands were actually executed locally against the real traces above (not fabricated) — outputs quoted here are copy-pasted from the CLI runs.
