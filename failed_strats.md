# ASH_COATED_OSMIUM — failed / neutral strategies

Baseline: `template.py` with deep-mid FV + AR(1) shift, `ACO_K=0.242392`.
Across 3 days: **282,024 total / 43,970 ASH**.

---

## Multi-level quoting (L >= 2)

Posted maker orders at multiple price levels inside the spread (best + 1, best + 2, ...).
Monotonically worse at every L. Initial version returned 0 PnL because `self.position`
nets long/short so per-layer capacity double-counted against POS_LIMIT, causing the whole
order batch to be rejected by the exchange. Fixed by tracking separate buy/sell budgets
against `POS_LIMIT +- initial_position` — no longer breaks, but still loses PnL vs single-level.
The outer layers mostly get filled against adverse flow while the inner layer was already
capturing everything worth capturing.

## Aggressive take edge sweep (ACO_TAKE_EDGE)

Swept the take gate `sell_price <= fair_value - edge` over {-1, 0, 1, 2}.

| edge | ASH |
|------|-----|
| -1   | 43,529 (-441) |
|  0   | 43,748 (-222) |
|  1   | 43,748 (-222)   (baseline before deep-mid swap) |
|  2   | 43,748 (-222) |

Note: sweep predates the deep-mid FV change; totals reference pre-deep-mid baseline.
No edge > 1 wins; edge < 1 bleeds.

## Defensive-only take band

Dropped the aggressive take layer (`sell_price <= fair_value - 1`), kept only the
`position < 0 & sell_price <= fair_value` unwind clause.
Result: **281,989 total / 43,935 ASH** (-35). Effectively flat.
The MM layer at `bid_wall+1 / ask_wall-1` already captures what the take layer was catching,
so removing it doesn't hurt — but it also doesn't help.

## Inventory skew (FV shift)

Shifted `fair_value -= alpha * initial_position` so quotes lean toward unwinding when loaded.

| alpha | ASH    | delta  |
|-------|--------|--------|
| 0.00  | 43,970 | base   |
| 0.01  | 43,987 | +17    |
| 0.02  | 43,994 | +24    |
| 0.05  | 43,858 | -112   |
| 0.10  | 40,974 | -2,996 |
| 0.20  |  9,108 | -34,862 |
| 0.40  |  2,194 | -41,776 |

Best alpha = 0.02 with only +24 ASH (noise-level). Larger alpha collapses PnL because
the shifted FV crosses the spread, disabling the overbid/underbid gates (
`overbidding_price < fair_value` / `underbidding_price > fair_value` stop triggering).
The MM layer already handles inventory implicitly via POS_LIMIT caps so explicit skew
adds nothing.
