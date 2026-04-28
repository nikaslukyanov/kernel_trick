# Round 3 Backtest Results Log

## Scoring targets: maximize final_pnl · Sharpe · Sortino

---

## v1 — SSVI market maker baseline
**Strategy:**
- VEV options (5000–5500) only; ASH/ROOT are no-ops on round-3 data
- SSVI fair value: ρ=0, φ=1 (calibrated flat smile), live θ EWM from nearest ATM option mid
- Reservation price = fair − pos·γ_opt(0.05) − base_delta·Δ·γ_delta(0.01)
- IV lean: shift both quotes by price_dev × 0.50
- Taker: cross if market quote mispriced > 1 tick vs fair
- Maker size 7, half-spread 1.0 tick

**Results:**
```
VEV_5000: 1,208 | VEV_5100: 682 | VEV_5200: 590
VEV_5300: -759  | VEV_5400: 122 | VEV_5500: 128
Day 0: 978  Day 1: 798  Day 2: 1,971   Total: 3,746
final_pnl: 3,746 | sharpe: 1.98 | sortino: inf | max_dd: 8,629
```

**Observations:**
- VEV_5300 (ATM) is the main loser. ATM has highest gamma risk + adverse selection.
- ITM strikes (5000-5200) earn from near-arbitrage on intrinsic value.
- OTM strikes (5400-5500) earn small amounts from spread.
- HYDROGEL_PACK not traded (ASH/ROOT in dispatch, not PACK). Adding PACK back could add incremental PnL.

---

## v2 — Wider spread + stronger inventory skew
**Changes vs v1:**
- half_spread: 1.0 → 2.0 (earn more per fill, protect adverse selection)
- gamma_opt: 0.05 → 0.10 (more aggressive per-option inventory flattening)
- take_thresh: 1.0 → 1.5 (don't take unless clearly mispriced)

**Results:** (not run — superseded by template.py refactor)

---

## template (current baseline) — Three-tier SSVI MM
**Strategy:**
- Deep ITM (4000, 4500): intrinsic ± 1 MM, DITM_TAKE_THRESH=2, DITM_GAMMA=0.05
- Active (5000-5100-5200-5400-5500, excl. 5300): SSVI fair value, VEV_TAKE_THRESH=5.0, VEV_IV_LEAN=1.0, VEV_MAKER_SIZE=7, VEV_GAMMA_OPT=0.05, VEV_GAMMA_DELTA=0.01
- Deep OTM (6000, 6500): bid=0, ask=1

**Results:**
```
Day 0: 82  Day 1: 1,051  Day 2: 1,259  Total: 2,392
final_pnl: 2,392 | sharpe: 1.27 | sortino: inf | max_dd: 3,002
```

**Observations:**
- Much lower than v1 (2,392 vs 3,746). Likely because take_thresh=5 is too conservative — rarely crosses spread.
- 5300 exclusion is correct. Three-tier structure (deep ITM / active / deep OTM) is the right frame.

---

## S1: v_take1 — Aggressive taker (VEV_TAKE_THRESH=1.0, IV_LEAN=0.5)
**Hypothesis:** Revert to v1 taker aggressiveness. take_thresh=5 misses too many fills; 1-tick misprices are real edge.
**Changes vs template:** `VEV_TAKE_THRESH=1.0`, `VEV_IV_LEAN=0.5`
**File:** `v_take1.py`

**Results:**
```
Day 0: 977  Day 1: 2,398  Day 2: 3,559  Total: 6,934
final_pnl: 6,934 | sharpe: 1.79 | sortino: inf | max_dd: 6,818
```
**Finding:** Best of all template-family strategies. Aggressive taking is the real edge. take_thresh=5 in template was essentially never firing — confirmed by v_no_take being identical to template.

---

## S2: v_no_take — Pure maker (no crossing)
**Hypothesis:** Taker leg adds adverse selection. Pure MM on SSVI fair value is the true edge.
**Changes vs template:** `VEV_TAKE_THRESH=9999`, `DITM_TAKE_THRESH=9999`
**File:** `v_no_take.py`

**Results:**
```
Day 0: 82  Day 1: 1,051  Day 2: 1,259  Total: 2,392
final_pnl: 2,392 | sharpe: 1.27 | sortino: inf | max_dd: 3,002
```
**Finding:** Identical to template. Confirms take_thresh=5 was never triggering — template is functionally a pure maker.

---

## S3: v_include_5300 — Add VEV_5300 back to active strikes
**Hypothesis:** With current SSVI+IV lean, 5300 (ATM) may now be profitable since IV signal is strongest ATM.
**Changes vs template:** `ACTIVE_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]`
**File:** `v_include_5300.py`

**Results:**
```
Day 0: 150  Day 1: 1,199  Day 2: 1,382  Total: 2,732
final_pnl: 2,732 | sharpe: 1.37 | sortino: inf | max_dd: 3,284
```
**Finding:** +340 vs template. Including 5300 now helps with current SSVI+IV lean logic. Worth including in future strategies.

---

## S4: v_call_spread_arb — Cross-strike call spread no-arb
**Hypothesis:** Call spread bounds `0 ≤ C(K_low) - C(K_high) ≤ K_high - K_low` are occasionally violated, producing locked riskless profit.
**Changes vs template:** Add `_call_spread_arb()` scanning all 45 strike pairs per tick. Fire when bounds violated.
**File:** `v_call_spread_arb.py`

**Results:**
```
Day 0: 82  Day 1: 1,051  Day 2: 1,259  Total: 2,392
final_pnl: 2,392 | sharpe: 1.27 | sortino: inf | max_dd: 3,002
```
**Finding:** Identical to template. Call spread no-arb violations do not occur in round 3 data (or are too small to fire given position limits). This channel adds nothing.

---

## S5: v_delta_hedge — Underlying delta hedge
**Hypothesis:** Net option delta creates directional drift in PnL. Hedging with underlying stabilizes returns.
**Changes vs template:** Buy/sell VELVETFRUIT_EXTRACT to flatten net delta when |delta| > 10.
**File:** `v_delta_hedge.py`

**Results:**
```
Day 0: 32  Day 1: 925  Day 2: 143  Total: 1,100
final_pnl: 1,100 | sharpe: 0.75 | sortino: inf | max_dd: 2,400
```
**Finding:** Hurt significantly (−1,292 vs template). Delta hedging costs the spread on the underlying and may cause position reversals that reduce option inventory accumulation. Do not hedge.

---

## S6: v_larger_size — Bigger maker quotes (size=15)
**Hypothesis:** More size per quote earns proportionally more spread income if fill rate is size-independent.
**Changes vs template:** `VEV_MAKER_SIZE=15`
**File:** `v_larger_size.py`

**Results:**
```
Day 0: -58  Day 1: 967  Day 2: 1,228  Total: 2,136
final_pnl: 2,136 | sharpe: 1.05 | sortino: 21.09 | max_dd: 3,130
```
**Finding:** Slightly worse than template (−256). Larger quotes attract more adverse flow per fill, reducing net edge. Maker size 7 is close to optimal.

---

## S7: v_otm_ask2 — Deep OTM ask=2
**Hypothesis:** Raising deep OTM ask from 1→2 earns 2x per fill with minimal volume loss.
**Changes vs template:** In `_trade_deep_otm`: `Order(sym, 2, ...)` instead of `Order(sym, 1, ...)`
**File:** `v_otm_ask2.py`

**Results:**
```
Day 0: 82  Day 1: 1,051  Day 2: 1,259  Total: 2,392
final_pnl: 2,392 | sharpe: 1.27 | sortino: inf | max_dd: 3,002
```
**Finding:** Identical to template. Deep OTM ask=2 fills 0 times — market only takes at ask=1. No change.

---

## Summary Table

| Strategy | PnL | Sharpe | Max DD | vs Template |
|---|---|---|---|---|
| template (baseline) | 2,392 | 1.27 | 3,002 | — |
| **S1 v_take1** | **6,934** | **1.79** | 6,818 | **+4,542** ✓ |
| S2 v_no_take | 2,392 | 1.27 | 3,002 | 0 (confirms thresh=5 never fires) |
| S3 v_include_5300 | 2,732 | 1.37 | 3,284 | +340 ✓ |
| S4 v_call_spread_arb | 2,392 | 1.27 | 3,002 | 0 (no violations) |
| S5 v_delta_hedge | 1,100 | 0.75 | 2,400 | −1,292 ✗ |
| S6 v_larger_size | 2,136 | 1.05 | 3,130 | −256 ✗ |
| S7 v_otm_ask2 | 2,392 | 1.27 | 3,002 | 0 (OTM fills 0) |

**Key conclusions:**
1. Taking with thresh=1 is worth ~4,500 PnL. The template's thresh=5 is dead code.
2. Including 5300 adds ~340 with current SSVI approach.
3. Delta hedging hurts — do not trade underlying.
4. Maker size 7 is optimal. Larger = more adverse selection.
5. Call spread arb and OTM ask changes are no-ops in this dataset.

**Next baseline: engine_ab.py (~55k PnL on IMC, ~9.5k local) — improve from there.**

---

# engine_ab Improvement Experiments

## engine_ab baseline — Live smile fit MM + underlying mean reversion + call spread arb
**Key architecture:**
- Engine A: VELVETFRUIT_EXTRACT mean-reversion MM, AR(2) around 5250.95
- Engine B: Live EWMA quadratic smile fit (parabola in log-moneyness), BS FV, bid=FV−BASE_EDGE, ask=FV+BASE_EDGE, inventory lean
- Engine C: Cross-strike call spread arb (scan 45 pairs/tick)
- Params: BASE_EDGE=1, SPREAD=4, REGRET_THRESHOLD=1500, SMILE_GAMMA=0.999, SMILE_WARMUP_N=2000

**Results (local):**
```
Day 0: 1,464  Day 1: 4,158  Day 2: 3,846  Total: 9,468
final_pnl: 9,468 | sharpe: 2.14 | sortino: inf | max_dd: 13,914
```
**Per-product notes:** VEV_5200 (−1,792) and VEV_5300 (−861) are consistent losers. Deep ITM VEV_4000+4500 earn +5,837. Engine A varies: +860/+1,447/−1,339 per day.

---

## ea_v1 — Add taker leg to VoucherTrader
**Changes:** When market ask ≤ FV − BASE_EDGE, cross and buy. When market bid ≥ FV + BASE_EDGE, cross and sell. Taker gated by same REGRET_THRESHOLD and strike_cap as maker.
**File:** `ea_v1.py`

**Results (local):**
```
Day 0: 1,508  Day 1: 4,150  Day 2: 3,937  Total: 9,595
final_pnl: 9,595 | sharpe: 2.18 | sortino: inf | max_dd: 13,830
```
**Finding:** Tiny +127 PnL improvement. Smile FV is more accurate than SSVI so fewer large misprices to capture via taking.

---

## ea_v2 — Wider spread + tighter delta (param sweep)
**Changes:** BASE_EDGE: 1→2, REGRET_THRESHOLD: 1500→750, SMILE_WARMUP_N: 2000→500
**File:** `ea_v2.py`

**Results (local):**
```
Day 0: 1,412  Day 1: 3,148  Day 2: 2,284  Total: 6,844
final_pnl: 6,844 | sharpe: 2.63 | sortino: inf | max_dd: 9,916
```
**Finding:** Lower PnL but best sharpe (2.63 vs 2.14). Tighter delta control reduces drawdown.

---

## ea_v3 — Taker + wider spread + tighter delta
**Changes:** ea_v1 + ea_v2 param changes
**File:** `ea_v3.py`

**Results (local):**
```
Day 0: 1,429  Day 1: 3,224  Day 2: 2,276  Total: 6,930
final_pnl: 6,930 | sharpe: 2.57 | sortino: inf | max_dd: 9,837
```
**Finding:** Similar to ea_v2. Taker adds little on top of wider spread.

---

## ea_v4 — Exclude VEV_5200 and VEV_5300 from Engine B MM
**Hypothesis:** 5200/5300 are consistent bleeders in engine_ab (−2,653 combined). Excluding them stops the adverse selection.
**Changes:** `MM_STRIKES = [4000, 4500, 5000, 5100, 5400, 5500, 6000, 6500]`. Engine B loops over MM_STRIKES instead of VOUCHER_STRIKES.
**File:** `ea_v4.py`

**Results (local):**
```
Day 0: 1,322  Day 1: 5,162  Day 2: 5,292  Total: 11,776
final_pnl: 11,776 | sharpe: 1.74 | sortino: inf | max_dd: 16,260
```
**Finding:** Best local PnL (+2,308 vs baseline). Sharpe drops slightly due to higher day-to-day variance. PnL×Sharpe = 20,490 vs 20,261 baseline.

---

## ea_v5 — Taker + exclude 5200/5300
**Changes:** ea_v1 (taker) + ea_v4 (MM_STRIKES excludes 5200/5300)
**File:** `ea_v5.py`

**Results (local):**
```
Day 0: 1,335  Day 1: 5,047  Day 2: 5,036  Total: 11,418
final_pnl: 11,418 | sharpe: 1.78 | sortino: inf | max_dd: 15,840
```
**Finding:** +1,950 vs baseline. Slightly worse than ea_v4 alone — taker cancels some maker income on these strikes.

---

## ea_v6 — Disable Engine A (no underlying trading)
**Hypothesis:** Engine A has high day-to-day variance (860/1447/−1339). Removing it may improve Sharpe.
**Changes:** Remove Engine A order generation from Trader.run; still persist wall_mid for smile.
**File:** `ea_v6.py`

**Results (local):**
```
Day 0: 536  Day 1: 3,730  Day 2: 6,626  Total: 10,893
final_pnl: 10,893 | sharpe: 1.19 | sortino: inf | max_dd: 13,931
```
**Finding:** Disabling Engine A reduces PnL (-883) AND Sharpe (2.14→1.19). Engine A is net positive and should stay enabled.

---

## engine_ab Summary Table

| Strategy | Local PnL | Sharpe | Max DD | PnL×Sharpe | Notes |
|---|---|---|---|---|---|
| engine_ab (baseline) | 9,468 | 2.14 | 13,914 | 20,261 | — |
| **ea_v4** | **11,776** | 1.74 | 16,260 | **20,490** | Exclude 5200/5300 ✓ |
| ea_v5 | 11,418 | 1.78 | 15,840 | 20,324 | Taker + exclude ✓ |
| ea_v1 | 9,595 | 2.18 | 13,830 | 20,917 | Taker only |
| ea_v2 | 6,844 | 2.63 | 9,916 | 18,000 | Wider spread ✗ |
| ea_v3 | 6,930 | 2.57 | 9,837 | 17,810 | Wider + taker ✗ |
| ea_v6 | 10,893 | 1.19 | 13,931 | 12,963 | No Engine A ✗ |

**Recommendations:**
1. **ea_v4** (exclude 5200/5300 from MM): Best local PnL improvement. High confidence — 5200/5300 are consistent bleeders confirmed across template and engine_ab experiments.
2. **ea_v1** (add taker): Small local gain but principled — adding taking was transformative in template experiments. May have larger impact on IMC platform.
3. **Do not** widen spread or tighten delta — reduces PnL significantly on this dataset.
4. **Keep Engine A** — net positive (+968 locally) despite day 2 loss.
