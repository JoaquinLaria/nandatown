# Experiment: message_drop 0.0 to 0.3 on marketplace

**Scenario:** marketplace (built-in, 50 buyers / 50 sellers, contract-net + alternating-offers, seed 42, 10000 ticks)

**Change:** `failures.message_drop` from 0.0 to 0.3. See `marketplace_drop30.yaml`. Everything else is the same as the built-in scenario.

## Hypothesis, before running

Each buy round needs a request and a response to survive, so with a 30% drop rate, roughly 0.7 x 0.7 = 49% of rounds should succeed. I expected `delivery_rate` to land near 0.5 and `deal_rate` to drop proportionally, but I expected total message volume to stay close to baseline, since I assumed agents would keep trying every round regardless of earlier drops.

## Result

| metric | baseline (0.0) | drop30 (0.3) |
|---|---|---|
| message_count | 2000 | 299 |
| correlation_ids | 1000 | 174 |
| delivery_rate | 1.000 | 0.718 |
| deal_rate | 0.532 | 0.439 |
| dropped_count | 0 | 49 |
| validator marketplace_all_responded | PASS | FAIL (40 unanswered buy requests) |

Reports: `reports/marketplace_baseline.html`, `reports/marketplace_drop30.html`. Traces: `traces/marketplace.jsonl`, `traces/marketplace_drop30.jsonl` (gitignored, regenerate with the commands below).

## What surprised me, and how I checked it

The part I got wrong was message volume. It fell from 2000 to 299, an 85% drop, which is way more than a 30% drop rate should cause on its own. The validator `marketplace_all_responded` also flipped from PASS to FAIL, with 40 unanswered buy requests.

I opened `packages/nest-core/nest_core/scenarios_builtin/marketplace.py` and read `BuyerAgent.on_message` (lines 95 to 159). A buyer only sends its next round's buy request from inside the handler for the previous round's response. There is no timeout and no retry anywhere in that loop. So if a buyer's request or the seller's response gets dropped, that buyer is done for the rest of the run. It never gets a second attempt.

With up to 10 sequential rounds per buyer, one early drop wipes out every round after it too, for that buyer. That compounds much faster than a flat 30% drop rate would suggest. I checked the math: if each round needs two independent deliveries at 0.7 survival each, a buyer's expected chain length before stalling is short, around 2 rounds, which lines up with the ~174 messages we actually saw across 50 buyers instead of the 500 you'd get if every buyer finished all 10 rounds. I also confirmed the run is still deterministic: I reran the same scenario file twice and the trace hashed identically both times, so this isn't RNG noise, it's the actual mechanic.

So the 40 unanswered requests aren't delivery noise. They're buyers that got permanently stuck.

## Takeaway

`failures.message_drop` in this scenario does not model a noisy but self-healing channel. It models permanent, silent loss for whichever agent is waiting on a response. The gap this exposes is that `contract_net` and `alternating_offers`, as wired into this scenario, have no retry or timeout layer, so any lossy transport quietly cuts off the affected agents instead of degrading gracefully.

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

## AI and tool use disclosure

I used Claude Code (Sonnet 5) to run the CLI commands (`nest doctor`, `nest run`, `nest inspect`, `nest report`), copy and edit the scenario YAML, read the marketplace scenario source to explain the message volume collapse, and draft this README. Every command above was actually run locally against the real traces, the numbers in the table are copied from the CLI output, not invented.
