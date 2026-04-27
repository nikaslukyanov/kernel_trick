"""
Round 4 portfolio QP.

Maximize  E[PnL]  -  0.5 * lambda * Var[PnL_score]
subject to per-contract caps |q_i| <= cap_i.

Key design choices motivated by the upstream analysis:

  1. Bid-ask is real. Long leg pays the ASK, short leg receives the BID.
     We split each instrument into two non-negative legs (q_long, q_short)
     with q_i = q_long_i - q_short_i. The optimizer never picks both >0
     simultaneously because that's a strictly dominated round-trip cost.

  2. Score = mean PnL over 100 fresh paths (manual rule), so the relevant
     risk is Var[mean] = Var[PnL] / N_score. Scoring N is constant, so it
     does not change argmax for fixed lambda, but we report the score-level
     std so lambda has interpretable units.

  3. Variance is computed on a SINGLE shared MC sample (paths). All legs
     use the same noise so cross-leg covariance (chooser-vs-replication,
     KO-vs-vanilla put, etc) is captured — that's the whole reason the
     QP is preferable to per-leg sizing.

  4. Cap on AC_45_KO is 500 vs 50 elsewhere (manual table).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from mc_engine import (
    simulate_paths, T1_STEPS, T2_STEPS,
    call_payoff, put_payoff,
    binary_put_payoff, knockout_put_payoff, chooser_payoff,
)

# ----- world / contracts (single source of truth) ---------------------------
SIZE = 3_000              # contract multiplier
N_SCORE = 100             # manual scoring sample size
BIN_AMOUNT = 10.0
N_PATHS = 1_000_000         # MC paths for the QP itself

MARKET = {
    # vanillas (T2 = 21d)
    'AC_50_P':   dict(kind='put',         strike=50, expiry=T2_STEPS, bid=12.00, ask=12.05, cap=50),
    'AC_50_C':   dict(kind='call',        strike=50, expiry=T2_STEPS, bid=12.00, ask=12.05, cap=50),
    'AC_35_P':   dict(kind='put',         strike=35, expiry=T2_STEPS, bid= 4.33, ask= 4.35, cap=50),
    'AC_40_P':   dict(kind='put',         strike=40, expiry=T2_STEPS, bid= 6.50, ask= 6.55, cap=50),
    'AC_45_P':   dict(kind='put',         strike=45, expiry=T2_STEPS, bid= 9.05, ask= 9.10, cap=50),
    'AC_60_C':   dict(kind='call',        strike=60, expiry=T2_STEPS, bid= 8.80, ask= 8.85, cap=50),
    # vanillas (T1 = 14d)
    'AC_50_P_2': dict(kind='put',         strike=50, expiry=T1_STEPS, bid= 9.70, ask= 9.75, cap=50),
    'AC_50_C_2': dict(kind='call',        strike=50, expiry=T1_STEPS, bid= 9.70, ask= 9.75, cap=50),
    # exotics
    'AC_50_CO':  dict(kind='chooser',     strike=50, expiry=T2_STEPS, decision=T1_STEPS,
                                                                    bid=22.20, ask=22.30, cap=50),
    'AC_40_BP':  dict(kind='binary_put',  strike=40, expiry=T2_STEPS, payout=BIN_AMOUNT,
                                                                    bid= 5.00, ask= 5.10, cap=50),
    'AC_45_KO':  dict(kind='ko_put',      strike=45, expiry=T2_STEPS, barrier=35,
                                                                    bid= 0.15, ask= 0.175, cap=500),
    'AC':        dict(kind='underlying',  strike=None, expiry=T2_STEPS,
                                                                    bid=49.975, ask=50.025, cap=200),
}


def per_unit_payoff(name, paths):
    """Per-path terminal payoff (no premium, no multiplier) for one unit long."""
    s = MARKET[name]
    k = s['kind']
    if k == 'call':        return call_payoff(paths[:, s['expiry']], s['strike'])
    if k == 'put':         return put_payoff (paths[:, s['expiry']], s['strike'])
    if k == 'binary_put':  return binary_put_payoff(paths[:, s['expiry']], s['strike'], s['payout'])
    if k == 'ko_put':      return knockout_put_payoff(paths, K=s['strike'], barrier=s['barrier'])
    if k == 'chooser':     return chooser_payoff(paths, s['strike'],
                                                  t1_step=s['decision'], T_step=s['expiry'])
    if k == 'underlying':  return paths[:, s['expiry']]
    raise ValueError(k)


# ----- build payoff matrices (long / short share path noise) ---------------
def build(seed=1):
    paths = simulate_paths(N_PATHS, n_steps=T2_STEPS, seed=seed)
    names = list(MARKET)
    caps  = np.array([MARKET[n]['cap'] for n in names], dtype=float)
    # Long leg: pay ask. Short leg: receive bid (so negate sign of premium).
    X_long  = np.column_stack([SIZE * (per_unit_payoff(n, paths) - MARKET[n]['ask']) for n in names])
    X_short = np.column_stack([SIZE * (MARKET[n]['bid'] - per_unit_payoff(n, paths)) for n in names])
    return names, caps, X_long, X_short


# ----- QP solver -----------------------------------------------------------
def solve(lam, names, caps, X_long, X_short):
    """
    Variables: z = [q_long, q_short], each in [0, cap].
    PnL_path = X_long @ q_long + X_short @ q_short.
    Maximize mean - 0.5*lam*var.
    """
    n = len(names)
    bounds = [(0, c) for c in caps] * 2

    def neg_obj(z):
        ql, qs = z[:n], z[n:]
        pnl = X_long @ ql + X_short @ qs
        return -(pnl.mean() - 0.5 * lam * pnl.var(ddof=1))

    def grad(z):
        ql, qs = z[:n], z[n:]
        pnl = X_long @ ql + X_short @ qs
        # d mean / d q = E[X], d var / d q = 2 * Cov(pnl, X)
        mean_grad_l = X_long.mean(axis=0)
        mean_grad_s = X_short.mean(axis=0)
        c = pnl - pnl.mean()
        var_grad_l = 2 * (X_long  - X_long.mean(0))  .T @ c / (len(pnl) - 1)
        var_grad_s = 2 * (X_short - X_short.mean(0)) .T @ c / (len(pnl) - 1)
        gl = -(mean_grad_l - 0.5 * lam * var_grad_l)
        gs = -(mean_grad_s - 0.5 * lam * var_grad_s)
        return np.concatenate([gl, gs])

    z0 = np.zeros(2 * n)
    res = minimize(neg_obj, z0, jac=grad, bounds=bounds, method='L-BFGS-B',
                   options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 5000})
    q = res.x[:n] - res.x[n:]
    pnl = X_long @ res.x[:n] + X_short @ res.x[n:]
    return q, pnl


def report(q, pnl, names):
    mu, sd = pnl.mean(), pnl.std(ddof=1)
    sd_score = sd / np.sqrt(N_SCORE)        # std of mean of 100 paths
    print(f"  E[PnL]   = {mu:>12,.0f}")
    print(f"  std PnL  = {sd:>12,.0f}   (single path)")
    print(f"  std mean = {sd_score:>12,.0f}   (over {N_SCORE} scoring paths)")
    print(f"  Sharpe (score) = {mu/sd_score:.2f}")
    print(f"  P(score>0) over 100 trials = {norm.cdf(mu/sd_score):.4f}")
    print(f"  positions (rounded):")
    for name, qi in zip(names, q):
        if abs(qi) >= 0.5:
            print(f"    '{name:10s}': {int(round(qi)):+5d},")


# ----- frontier sweep ------------------------------------------------------
if __name__ == "__main__":
    names, caps, X_long, X_short = build(seed=42)

    LAMBDAS = [1e-9, 1e-8, 1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 1e-4]

    print("=== lambda frontier (summary) ===")
    print(f"{'lambda':>10s}  {'E[PnL]':>12s}  {'std':>12s}  {'std/√100':>10s}  "
          f"{'Sharpe':>7s}  {'P(score>0)':>10s}")
    cached = {}
    for lam in LAMBDAS:
        q, pnl = solve(lam, names, caps, X_long, X_short)
        cached[lam] = (q, pnl)
        mu, sd = pnl.mean(), pnl.std(ddof=1)
        sd_s = sd / np.sqrt(N_SCORE)
        p_pos = norm.cdf(mu / sd_s)
        print(f"{lam:>10.1e}  {mu:>12,.0f}  {sd:>12,.0f}  {sd_s:>10,.0f}  "
              f"{mu/sd_s:>7.2f}  {p_pos:>10.4f}")

    print("\n=== per-lambda detail ===")
    for lam in LAMBDAS:
        q, pnl = cached[lam]
        print(f"\n--- lambda = {lam:.0e} ---")
        report(q, pnl, names)
