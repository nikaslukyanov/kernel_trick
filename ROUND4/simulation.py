"""
Round 4 manual: position simulator.

Workflow:
  1. Pick a side (buy/sell) and signed quantity for each contract you want to trade.
  2. Each contract is paid for at the worst-of-book quote (ask if buying, bid if selling).
     This mimics a market order at touch.
  3. We simulate 100 GBM paths (the scoring config) plus a larger MC for distribution.
  4. PnL per contract = qty * SIZE * (terminal_payoff_per_unit - entry_price_per_unit).
     SIZE is the per-contract multiplier from the manual ("contract size of 3,000").

Sign convention: positive qty = long, negative qty = short.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mc_engine import (
    simulate_paths,
    call_payoff, put_payoff,
    binary_put_payoff, knockout_put_payoff, chooser_payoff,
)

# --- world constants (duplicated from mc_engine for clarity at the trade layer) ---
S0            = 49.975
STEPS_PER_DAY = 4
DAYS_PER_YEAR = 252
DT            = 1.0 / (DAYS_PER_YEAR * STEPS_PER_DAY)
T1_DAYS       = 14                                    # 2-week / chooser decision
T2_DAYS       = 21                                    # 3-week
T1_STEPS      = T1_DAYS * STEPS_PER_DAY               # 56
T2_STEPS      = T2_DAYS * STEPS_PER_DAY               # 84

CONTRACT_SIZE = 3_000          # multiplier; per intro panel of screenshot
SCORING_PATHS = 100            # the manual scores on average across 100 sims
DIST_PATHS    = 20_000         # for visualizing PnL distribution

# Binary put AC_40_BP: pays 10 XIRECs if S_T < 40, else 0 (confirmed by user).
BIN_AMOUNT = 10.0

# --- contracts table ---------------------------------------------------------
# `payoff_fn(paths)` must return a per-path payoff array (per unit, no multiplier).
# `tau` only matters under r != 0; kept for clarity.
def _call_T(K, n_steps):
    return lambda paths: call_payoff(paths[:, n_steps], K)

def _put_T(K, n_steps):
    return lambda paths: put_payoff(paths[:, n_steps], K)

CONTRACTS = {
    # name: dict of fields
    "AC_50_P":   {"bid": 12.00, "ask": 12.05, "cap": 50,  "tau_steps": T2_STEPS,
                  "payoff": _put_T(50, T2_STEPS)},
    "AC_50_C":   {"bid": 12.00, "ask": 12.05, "cap": 50,  "tau_steps": T2_STEPS,
                  "payoff": _call_T(50, T2_STEPS)},
    "AC_35_P":   {"bid":  4.33, "ask":  4.35, "cap": 50,  "tau_steps": T2_STEPS,
                  "payoff": _put_T(35, T2_STEPS)},
    "AC_40_P":   {"bid":  6.50, "ask":  6.55, "cap": 50,  "tau_steps": T2_STEPS,
                  "payoff": _put_T(40, T2_STEPS)},
    "AC_45_P":   {"bid":  9.05, "ask":  9.10, "cap": 50,  "tau_steps": T2_STEPS,
                  "payoff": _put_T(45, T2_STEPS)},
    "AC_60_C":   {"bid":  8.80, "ask":  8.85, "cap": 50,  "tau_steps": T2_STEPS,
                  "payoff": _call_T(60, T2_STEPS)},
    "AC_50_P_2": {"bid":  9.70, "ask":  9.75, "cap": 50,  "tau_steps": T1_STEPS,
                  "payoff": _put_T(50, T1_STEPS)},
    "AC_50_C_2": {"bid":  9.70, "ask":  9.75, "cap": 50,  "tau_steps": T1_STEPS,
                  "payoff": _call_T(50, T1_STEPS)},
    "AC_50_CO":  {"bid": 22.20, "ask": 22.30, "cap": 50,  "tau_steps": T2_STEPS,
                  "payoff": lambda paths: chooser_payoff(paths, 50, T1_STEPS, T2_STEPS)},
    "AC_40_BP":  {"bid":  5.00, "ask":  5.10, "cap": 50,  "tau_steps": T2_STEPS,
                  "payoff": lambda paths: binary_put_payoff(paths[:, T2_STEPS], 40, BIN_AMOUNT)},
    "AC_45_KO":  {"bid":  0.15, "ask":  0.175,"cap": 500, "tau_steps": T2_STEPS,
                  "payoff": lambda paths: knockout_put_payoff(paths, K=45, barrier=35)},
    # Spot underlying — exposure linear in S_T - S0, no premium beyond touch slippage
    "AC":        {"bid": 49.975,"ask": 50.025,"cap": 200, "tau_steps": T2_STEPS,
                  "payoff": lambda paths: paths[:, T2_STEPS]},
}


def entry_price(name, qty):
    """Long pays ask, short receives bid."""
    c = CONTRACTS[name]
    if qty > 0: return c["ask"]
    if qty < 0: return c["bid"]
    return 0.0


def validate_positions(positions):
    """Raise if any |qty| exceeds cap or contract is unknown."""
    for name, qty in positions.items():
        if name not in CONTRACTS:
            raise KeyError(f"unknown contract: {name}")
        cap = CONTRACTS[name]["cap"]
        if abs(qty) > cap:
            raise ValueError(f"{name}: |qty|={abs(qty)} > cap={cap}")


def simulate(positions, n_paths=DIST_PATHS, seed=0, plot=True, verbose=True):
    """
    positions: dict {contract_name: signed_qty}
    Returns dict with summary stats and the per-path PnL array.
    """
    validate_positions(positions)
    paths = simulate_paths(n_paths, n_steps=T2_STEPS, seed=seed)

    # build per-path PnL by summing each leg
    total_pnl = np.zeros(n_paths)
    leg_table = []
    for name, qty in positions.items():
        if qty == 0:
            continue
        c = CONTRACTS[name]
        payoff_per_unit = c["payoff"](paths)            # shape (n_paths,)
        entry = entry_price(name, qty)
        # PnL per contract per path = payoff - entry; multiply by qty * size
        leg_pnl = qty * CONTRACT_SIZE * (payoff_per_unit - entry)
        total_pnl += leg_pnl
        leg_table.append((name, qty, entry, payoff_per_unit.mean(),
                          leg_pnl.mean(), leg_pnl.std()))

    mu = total_pnl.mean()
    sd = total_pnl.std(ddof=1)
    sd_score = sd / np.sqrt(SCORING_PATHS)
    sharpe_score = mu / sd_score if sd_score > 0 else float("nan")
    prob_score_pos = norm.cdf(sharpe_score) if sd_score > 0 else float("nan")

    summary = {
        "mean_pnl": mu,
        "std_pnl":  sd,
        "std_score": sd_score,
        "sharpe_score": sharpe_score,
        "prob_score_pos": prob_score_pos,
        "p05":      np.percentile(total_pnl, 5),
        "p50":      np.percentile(total_pnl, 50),
        "p95":      np.percentile(total_pnl, 95),
        "min":      total_pnl.min(),
        "max":      total_pnl.max(),
        "prob_profit": (total_pnl > 0).mean(),
    }

    if verbose:
        print(f"\n=== Position legs ({len(leg_table)} active) ===")
        print(f"{'name':12s} {'qty':>5s} {'entry':>8s} {'E[payoff]':>10s} "
              f"{'E[PnL]':>14s} {'sd[PnL]':>14s}")
        for row in leg_table:
            print(f"{row[0]:12s} {row[1]:>5d} {row[2]:>8.3f} {row[3]:>10.3f} "
                  f"{row[4]:>14.1f} {row[5]:>14.1f}")
        print("\n=== Total PnL distribution (over {} paths) ===".format(n_paths))
        for k, v in summary.items():
            print(f"  {k:14s} = {v:>14.4f}" if isinstance(v, float) else f"  {k:14s} = {v}")

        # paste-ready rounded position block
        print("\n=== positions (rounded) ===")
        for name, qty in positions.items():
            if abs(qty) >= 0.5:
                print(f"    '{name:10s}': {int(round(qty)):+5d},")

        # also report what the manual's own scoring criterion (avg PnL over 100 sims) gives
        score_paths = simulate_paths(SCORING_PATHS, n_steps=T2_STEPS, seed=seed + 1)
        score_pnl = np.zeros(SCORING_PATHS)
        for name, qty in positions.items():
            if qty == 0: continue
            entry = entry_price(name, qty)
            score_pnl += qty * CONTRACT_SIZE * (CONTRACTS[name]["payoff"](score_paths) - entry)
        print(f"\n  scoring proxy (avg PnL over {SCORING_PATHS} fresh paths) = {score_pnl.mean():.2f}")

    if plot:
        _, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(total_pnl, bins=80, alpha=0.85, color="C0", edgecolor="white")
        ax.axvline(0, color="k", lw=1)
        ax.axvline(summary["mean_pnl"], color="C1", lw=2, label=f"mean = {summary['mean_pnl']:.0f}")
        ax.axvline(summary["p05"], color="C3", ls="--", lw=1, label=f"5% = {summary['p05']:.0f}")
        ax.axvline(summary["p95"], color="C2", ls="--", lw=1, label=f"95% = {summary['p95']:.0f}")
        ax.set(xlabel="terminal PnL per path", ylabel="paths",
               title="PnL distribution across simulated paths")
        ax.legend()
        plt.tight_layout()
        plt.savefig("pnl_distribution.png", dpi=130, bbox_inches="tight")
        print("  saved pnl_distribution.png")

    return {"summary": summary, "pnl": total_pnl, "legs": leg_table}


if __name__ == "__main__":


    # seed 1 
    # positions = {
    #     'AC_50_P':   +12,
    #     'AC_50_C':   +14,
    #     'AC_35_P':   -35,
    #     'AC_45_P':   +50,
    #     'AC_50_P_2':    +9,
    #     'AC_50_C_2':    +5,
    #     'AC_50_CO':   -14,
    #     'AC_40_BP':   -50,
    # }
    
    # # seed 42
    positions = {
        'AC_50_P':   +11,
        'AC_50_C':   +14,
        'AC_35_P':   -34,
        'AC_45_P':   +50,
        'AC_50_P_2':  +9,
        'AC_50_C_2':    +5,
        'AC_50_CO':   -14,
        'AC_40_BP':   -50,
    }

    # abd 
    # positions = {
    #     'AC_50_P':   0,
    #     'AC_50_C':   0,
    #     'AC_35_P':   0,
    #     'AC_45_P':   0,
    #     'AC_60_C': -50,
    #     'AC_50_P_2':  50,
    #     'AC_50_C_2':  50,
    #     'AC_50_CO':   -50,
    #     'AC_40_BP':   -50,
    #     'AC_45_KO': -500
    # }

    simulate(positions, n_paths=1_000_000, seed=3)
