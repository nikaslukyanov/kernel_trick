"""
Gardener Guild dashboard — robust bid optimizer.

Run:
    pip3 install dash plotly numpy scipy
    python3 ROUND3/manual/dashboard.py

Opens http://127.0.0.1:8050
"""
from __future__ import annotations

import re
import numpy as np
from scipy.stats import beta as beta_dist
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go

# ---- model -----------------------------------------------------------------

GRID = np.arange(670, 921, 5)
N = len(GRID)
P_RES = np.full(N, 1 / N)


def uniform_pmf() -> np.ndarray:
    return np.full(N, 1 / N)


def normal_pmf(mu: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-6)
    w = np.exp(-0.5 * ((GRID - mu) / sigma) ** 2)
    return w / w.sum()


def beta_pmf(a: float, b: float) -> np.ndarray:
    t = np.clip((GRID - 670) / 250, 1e-6, 1 - 1e-6)
    w = beta_dist.pdf(t, max(a, 1e-3), max(b, 1e-3))
    s = w.sum()
    return w / s if s > 0 else uniform_pmf()


def bimodal_pmf() -> np.ndarray:
    w = np.exp(-0.5 * ((GRID - 760) / 15) ** 2) + np.exp(-0.5 * ((GRID - 870) / 15) ** 2)
    return w / w.sum()


def point_pmf(x: float) -> np.ndarray:
    p = np.zeros(N)
    p[int(np.argmin(np.abs(GRID - x)))] = 1.0
    return p


def mean_of(pmf: np.ndarray) -> float:
    return float((GRID * pmf).sum())


def ev_scalar(b1: int, b2: int, a2: float):
    p1 = P_RES[GRID < b1].sum()
    p2 = P_RES[(GRID >= b1) & (GRID < b2)].sum()
    if b2 >= 920:
        pen = 0.0
    else:
        pen = 1.0 if b2 >= a2 else ((920 - a2) / (920 - b2)) ** 3
    pnl1 = p1 * (920 - b1)
    pnl2 = p2 * (920 - b2) * pen
    return pnl1 + pnl2, pnl1, pnl2, p1, p2, pen


# Precompute EV_TENSOR[k, i, j] = EV(b1=GRID[i], b2=GRID[j], avg_b2=GRID[k])
# NaN where j < i (b2 < b1). Shape (N, N, N). Built once at startup.
print("Precomputing EV tensor…")
EV_TENSOR = np.full((N, N, N), np.nan)
for _k in range(N):
    _a2 = float(GRID[_k])
    _p1 = np.array([P_RES[GRID < b1].sum() for b1 in GRID])
    for _i in range(N):
        for _j in range(_i, N):
            _b2 = int(GRID[_j])
            _p2 = P_RES[(GRID >= GRID[_i]) & (GRID < GRID[_j])].sum()
            _pen = (0.0 if _b2 >= 920
                    else (1.0 if _b2 >= _a2
                          else ((920 - _a2) / (920 - _b2)) ** 3))
            EV_TENSOR[_k, _i, _j] = (_p1[_i] * (920 - GRID[_i])
                                      + _p2 * (920 - _b2) * _pen)
print("EV tensor ready.")


def optimize_robust(a2_pmf: np.ndarray):
    """Return (obj_value, b1*, b2*, point_ev_at_E[a2])."""
    ev_mat = np.einsum("k,kij->ij", a2_pmf, np.nan_to_num(EV_TENSOR, nan=-1e9))
    ev_mat[np.isnan(EV_TENSOR[0])] = -np.inf
    flat = np.argmax(ev_mat)
    i_star, j_star = divmod(int(flat), N)
    obj_v = float(ev_mat[i_star, j_star])
    b1_star = int(GRID[i_star])
    b2_star = int(GRID[j_star])
    e_a2 = mean_of(a2_pmf)
    point_ev = float(EV_TENSOR[int(np.argmin(np.abs(GRID - e_a2))), i_star, j_star])
    return obj_v, b1_star, b2_star, point_ev


# ---- path -> pmf -----------------------------------------------------------

_PATH_RE = re.compile(r"[ML]\s*([-\d.eE]+)[,\s]+([-\d.eE]+)")


def path_to_pmf(path: str) -> np.ndarray | None:
    pts = [(float(x), float(y)) for x, y in _PATH_RE.findall(path)]
    if len(pts) < 2:
        return None
    xs = np.array([p[0] for p in pts])
    ys = np.clip(np.array([p[1] for p in pts]), 0, None)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    ux, inv = np.unique(xs, return_inverse=True)
    uy, cnt = np.zeros_like(ux), np.zeros_like(ux)
    for ii, jj in enumerate(inv):
        uy[jj] += ys[ii]
        cnt[jj] += 1
    uy /= np.maximum(cnt, 1)
    if len(ux) < 2:
        return None
    pmf = np.clip(np.interp(GRID, ux, uy, left=0.0, right=0.0), 0, None)
    s = pmf.sum()
    return pmf / s if s > 0 else None


# ---- styling ---------------------------------------------------------------

BG = "#0b1220"
PANEL = "#111a2b"
TEAL = "#4fd1c5"
GOLD = "#d4a84a"
MUTED = "#7b8aa1"
TEXT = "#e6edf7"
GREEN = "#6fbf87"
RED = "#e07070"
FONT = "ui-monospace, SFMono-Regular, Menlo, monospace"

_base_layout = dict(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    font=dict(family=FONT, color=TEXT, size=11),
    margin=dict(l=40, r=20, t=20, b=30),
    showlegend=False, bargap=0.02,
    xaxis=dict(gridcolor="#1b2740", zeroline=False),
    yaxis=dict(gridcolor="#1b2740", zeroline=False),
)


def fig_b2_empty() -> go.Figure:
    fig = go.Figure()
    layout = {**_base_layout, "height": 270,
              "xaxis": dict(range=[670, 920], gridcolor="#1b2740", zeroline=False,
                            title="opponent b2 bid"),
              "yaxis": dict(gridcolor="#1b2740", zeroline=False, title="density")}
    fig.update_layout(**layout)
    return fig


def fig_b2_distro(pmf: np.ndarray) -> go.Figure:
    cdf = np.cumsum(pmf)
    a2 = mean_of(pmf)
    fig = go.Figure()
    fig.add_bar(x=GRID, y=pmf, marker_color=TEAL, opacity=0.75,
                hovertemplate="b2=%{x}<br>p=%{y:.4f}<extra></extra>")
    fig.add_scatter(x=GRID, y=cdf * pmf.max() / max(cdf.max(), 1e-9),
                    mode="lines", line=dict(color=GOLD, dash="dash"), hoverinfo="skip")
    fig.add_vline(x=a2, line=dict(color=GOLD, width=1),
                  annotation_text=f"mean={a2:.1f}",
                  annotation_position="top left", annotation_font_color=GOLD)
    fig.update_layout(**_base_layout, dragmode="drawopenpath",
                      newshape=dict(line=dict(color=GOLD, width=2)), height=270,
                      xaxis_title="opponent b2 bid", yaxis_title="density")
    return fig


def fig_a2_distro(pmf: np.ndarray) -> go.Figure:
    a2 = mean_of(pmf)
    fig = go.Figure()
    fig.add_bar(x=GRID, y=pmf, marker_color=GREEN, opacity=0.75,
                hovertemplate="avg_b2=%{x}<br>p=%{y:.4f}<extra></extra>")
    fig.add_vline(x=a2, line=dict(color=GOLD, width=1),
                  annotation_text=f"E[avg_b2]={a2:.1f}",
                  annotation_position="top left", annotation_font_color=GOLD)
    fig.update_layout(**_base_layout, dragmode="drawopenpath",
                      newshape=dict(line=dict(color=GREEN, width=2)), height=210,
                      xaxis_title="possible avg_b2 value", yaxis_title="prior weight")
    return fig


def fig_sensitivity(b1: int, b2: int, a2_pmf: np.ndarray) -> go.Figure:
    b1_idx = int(np.argmin(np.abs(GRID - b1)))
    b2_idx = int(np.argmin(np.abs(GRID - b2)))
    evs = EV_TENSOR[:, b1_idx, b2_idx]
    e_a2 = mean_of(a2_pmf)
    ev_at_e = float(np.interp(e_a2, GRID, evs))
    pmf_scaled = a2_pmf / max(a2_pmf.max(), 1e-9) * float(np.nanmax(evs)) * 0.25

    fig = go.Figure()
    fig.add_bar(x=GRID, y=pmf_scaled, marker_color=GREEN, opacity=0.22, name="prior")
    fig.add_scatter(x=GRID, y=evs, mode="lines", line=dict(color=TEAL, width=2),
                    hovertemplate="avg_b2=%{x}<br>EV=%{y:.2f}<extra></extra>")
    fig.add_vline(x=b2, line=dict(color=RED, width=1, dash="dot"),
                  annotation_text=f"b2*={b2} (cliff)",
                  annotation_position="top right", annotation_font_color=RED)
    fig.add_vline(x=e_a2, line=dict(color=GOLD, width=1),
                  annotation_text=f"E[avg]={e_a2:.0f}",
                  annotation_position="top left", annotation_font_color=GOLD)
    fig.add_scatter(x=[e_a2], y=[ev_at_e], mode="markers",
                    marker=dict(color=GOLD, size=10), hoverinfo="skip")
    fig.update_layout(**_base_layout, height=210,
                      xaxis_title="true avg_b2 (unknown on submission day)",
                      yaxis_title="E[PnL] per counterparty")
    return fig


# ---- layout ----------------------------------------------------------------

app = Dash(__name__)
app.title = "Gardener Guild"

panel_style = {"background": PANEL, "padding": "16px 18px", "borderRadius": "8px",
               "border": "1px solid #1b2740"}
h_style = {"margin": "0 0 4px 0", "color": TEXT, "fontSize": "15px", "fontWeight": 600}
sub_style = {"margin": "0 0 10px 0", "color": MUTED, "fontSize": "11px"}
btn_style = {"background": "transparent", "color": TEXT, "border": f"1px solid {MUTED}",
             "padding": "4px 10px", "marginRight": "6px", "marginTop": "6px",
             "borderRadius": "4px", "fontFamily": FONT, "fontSize": "11px", "cursor": "pointer"}
readout_row = {"display": "flex", "justifyContent": "space-between", "padding": "6px 0",
               "borderBottom": "1px solid #1b2740", "fontSize": "12px"}

_DEFAULT_B2_PMF = point_pmf(835)
_DEFAULT_A2_PMF = point_pmf(835)


def _inp(id_, val, step=1, w="70px"):
    return dcc.Input(id=id_, type="number", value=val, step=step,
                     style={"width": w, "background": BG, "color": TEXT,
                            "border": f"1px solid {MUTED}", "padding": "3px 6px",
                            "borderRadius": "3px", "fontFamily": FONT})


def _lbl(txt, color=MUTED):
    return html.Span(txt, style={"color": color, "fontSize": "11px"})


app.layout = html.Div(
    style={"background": BG, "color": TEXT, "minHeight": "100vh",
           "padding": "20px", "fontFamily": FONT},
    children=[
        dcc.Store(id="pmf-store", data=point_pmf(835).tolist()),
        dcc.Store(id="a2-pmf-store", data=point_pmf(835).tolist()),

        html.Div(style={"display": "grid", "gridTemplateColumns": "1.4fr 1fr", "gap": "16px"},
                 children=[

            # ===== LEFT COL =====
            html.Div(children=[

                # Panel 1 — b2 distribution
                html.Div(id="panel-b2", children=[html.Div(style=panel_style, children=[
                    html.H3("Opponent b2 distribution", style=h_style),
                    html.P("Draw density over opponents' individual second bids. "
                           "Mean auto-propagates to avg_b2 prior when mode = Derived.", style=sub_style),
                    dcc.Graph(id="distro",
                              figure=fig_b2_distro(_DEFAULT_B2_PMF),
                              config={"displayModeBar": True,
                                      "modeBarButtonsToAdd": ["drawopenpath", "eraseshape"],
                                      "displaylogo": False}),
                    html.Div([
                        html.Button("UNIFORM", id="btn-uniform", style=btn_style),
                        html.Button("CLEAR", id="btn-clear", style=btn_style),
                    ]),
                    html.Div(style={"marginTop": "10px", "display": "flex", "gap": "8px",
                                    "alignItems": "center", "flexWrap": "wrap"}, children=[
                        _lbl("NORMAL", GOLD), _lbl("μ"), _inp("mu", 795),
                        _lbl("σ"), _inp("sigma", 25),
                        html.Button("APPLY", id="btn-normal",
                                    style={**btn_style, "border": f"1px solid {GOLD}", "color": GOLD}),
                    ]),
                ])]),

                # Panel 2 — avg_b2 prior
                html.Div(style={**panel_style, "marginTop": "14px"}, children=[
                    html.H3("avg_b2 prior — uncertainty about mean opponent bid", style=h_style),
                    html.P("Your belief about what avg_b2 will be. Used for robust optimization.",
                           style=sub_style),
                    html.Div(id="a2-chart-wrapper", children=[
                        dcc.Graph(id="a2-distro",
                                  figure=fig_a2_distro(_DEFAULT_A2_PMF),
                                  config={"displayModeBar": True,
                                          "modeBarButtonsToAdd": ["drawopenpath", "eraseshape"],
                                          "displaylogo": False}),
                        html.Div([
                            html.Button("NASH (835)", id="btn-a2-nash", style=btn_style),
                            html.Button("LEVEL-0 (795)", id="btn-a2-level0", style=btn_style),
                            html.Button("UNIFORM", id="btn-a2-uniform", style=btn_style),
                            html.Button("CLEAR", id="btn-a2-clear", style=btn_style),
                        ]),
                        html.Div(style={"marginTop": "10px", "display": "flex", "gap": "8px",
                                        "alignItems": "center", "flexWrap": "wrap"}, children=[
                            _lbl("NORMAL", GOLD), _lbl("μ"), _inp("mu-a2", 840),
                            _lbl("σ"), _inp("sigma-a2", 30),
                            html.Button("APPLY", id="btn-a2-normal",
                                        style={**btn_style, "border": f"1px solid {GOLD}", "color": GOLD}),
                        ]),
                        html.Div(style={"marginTop": "8px", "display": "flex", "gap": "8px",
                                        "alignItems": "center", "flexWrap": "wrap"}, children=[
                            _lbl("BETA (rescaled 670–920)", GREEN),
                            _lbl("α"), _inp("beta-a", 2, step=0.1, w="60px"),
                            _lbl("β"), _inp("beta-b", 2, step=0.1, w="60px"),
                            html.Button("APPLY", id="btn-a2-beta",
                                        style={**btn_style, "border": f"1px solid {GREEN}", "color": GREEN}),
                        ]),
                    ]),
                ]),

                # Panel 3 — sensitivity
                html.Div(style={**panel_style, "marginTop": "14px"}, children=[
                    html.H3("Sensitivity — EV vs true avg_b2", style=h_style),
                    html.P("Teal = EV(b1*,b2*, true_avg). Green = your prior. "
                           "Red = b2* penalty cliff. Gold dot = EV at E[avg_b2].", style=sub_style),
                    dcc.Graph(id="sensitivity", config={"displayModeBar": False}),
                ]),
            ]),

            # ===== RIGHT COL =====
            html.Div(style=panel_style, children=[
                html.H3("Recommended bids", style=h_style),
                html.P("Updates live as you redraw.", style=sub_style),


                # Bids box
                html.Div(style={"border": f"1px solid {GOLD}", "padding": "14px",
                                "borderRadius": "6px"}, children=[
                    html.Div("— OPTIMAL BIDS —", style={"color": GOLD, "fontSize": "10px",
                                                          "letterSpacing": "2px", "textAlign": "center"}),
                    html.Div(style={"display": "flex", "justifyContent": "space-around",
                                    "marginTop": "10px"}, children=[
                        html.Div([html.Div("b1*", style={"color": MUTED, "fontSize": "10px",
                                                          "letterSpacing": "2px"}),
                                  html.Div(id="b1-out", style={"color": TEAL, "fontSize": "30px",
                                                                "fontWeight": 600})]),
                        html.Div([html.Div("b2*", style={"color": MUTED, "fontSize": "10px",
                                                          "letterSpacing": "2px"}),
                                  html.Div(id="b2-out", style={"color": GREEN, "fontSize": "30px",
                                                                "fontWeight": 600})]),
                        html.Div([html.Div("EV @ E[avg]", style={"color": MUTED, "fontSize": "10px",
                                                                   "letterSpacing": "2px"}),
                                  html.Div(id="ev-out", style={"color": GOLD, "fontSize": "30px",
                                                                "fontWeight": 600})]),
                    ]),
                ]),

                html.Div(style={"marginTop": "16px"}, children=[
                    html.Div("PRIOR STATS", style={"color": MUTED, "fontSize": "10px",
                                                    "letterSpacing": "2px", "marginBottom": "6px"}),
                    html.Div(style=readout_row, children=[
                        html.Span("E[avg_b2] under prior  [price, 670–920]"),
                        html.Span(id="a2-out", style={"color": GOLD})]),
                    html.Div(style=readout_row, children=[
                        html.Span("E[EV] over full prior  [pnl per counterparty]"),
                        html.Span(id="ev-point-out", style={"color": TEAL})]),
                    html.Div(style=readout_row, children=[
                        html.Span("penalty factor at E[avg_b2]  [0–1]"),
                        html.Span(id="pen-out", style={"color": GOLD})]),
                ]),

                html.Div(style={"marginTop": "16px"}, children=[
                    html.Div("TRADE BREAKDOWN  (at E[avg_b2])",
                             style={"color": MUTED, "fontSize": "10px",
                                    "letterSpacing": "2px", "marginBottom": "6px"}),
                    html.Div(style=readout_row, children=[
                        html.Span("P(r ≤ b1*) — b1-leg"),
                        html.Span(id="p1-out", style={"color": TEAL})]),
                    html.Div(style=readout_row, children=[
                        html.Span("P(b1 < r ≤ b2*) — b2-leg"),
                        html.Span(id="p2-out", style={"color": GREEN})]),
                    html.Div(style=readout_row, children=[
                        html.Span("P(no trade)"),
                        html.Span(id="p0-out", style={"color": MUTED})]),
                    html.Div(style=readout_row, children=[
                        html.Span("b1-leg EV contribution"),
                        html.Span(id="pnl1-out", style={"color": TEAL})]),
                    html.Div(style=readout_row, children=[
                        html.Span("b2-leg EV contribution"),
                        html.Span(id="pnl2-out", style={"color": GREEN})]),
                ]),

                html.Div(style={"marginTop": "16px"}, children=[
                    html.Div("SCALE → TOTAL PnL", style={"color": MUTED, "fontSize": "10px",
                                                          "letterSpacing": "2px", "marginBottom": "6px"}),
                    html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center",
                                    "flexWrap": "wrap", "marginBottom": "8px"}, children=[
                        _lbl("gardeners per atom"), _inp("n-per-atom", 1)]),
                    html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center",
                                    "flexWrap": "wrap", "marginBottom": "10px"}, children=[
                        _lbl("items per gardener"), _inp("items-per", 1)]),
                    html.Div(style=readout_row, children=[
                        html.Span("gardener count  (51 × n)"),
                        html.Span(id="ngard-out", style={"color": TEAL})]),
                    html.Div(style=readout_row, children=[
                        html.Span("total items"),
                        html.Span(id="nitems-out", style={"color": TEAL})]),
                    html.Div(style=readout_row, children=[
                        html.Span("TOTAL E[PnL]"),
                        html.Span(id="grand-out", style={"color": GOLD, "fontSize": "16px",
                                                          "fontWeight": 700})]),
                ]),
            ]),
        ]),
    ],
)


# ---- callbacks -------------------------------------------------------------




# b2 distribution -----------------------------------------------------------

@app.callback(
    Output("pmf-store", "data"),
    Output("distro", "figure"),
    Output("a2-pmf-store", "data", allow_duplicate=True),
    Output("a2-distro", "figure", allow_duplicate=True),
    Input("btn-uniform", "n_clicks"),
    Input("btn-normal", "n_clicks"),
    Input("btn-clear", "n_clicks"),
    Input("distro", "relayoutData"),
    State("mu", "value"),
    State("sigma", "value"),
    State("pmf-store", "data"),
    prevent_initial_call=True,
)
def update_b2_pmf(_u, _n, _c, relayout, mu, sigma, current):
    trig = ctx.triggered_id
    pmf = np.array(current) if current else uniform_pmf()

    if trig == "btn-uniform":
        pmf = uniform_pmf()
    elif trig == "btn-normal":
        pmf = normal_pmf(float(mu or 795), float(sigma or 25))
    elif trig == "btn-clear":
        pass
    elif trig == "distro" and relayout:
        shapes = relayout.get("shapes")
        if shapes:
            for sh in reversed(shapes):
                new = path_to_pmf(sh.get("path", ""))
                if new is not None:
                    pmf = new
                    break
        else:
            for k in relayout:
                if k.endswith(".path"):
                    new = path_to_pmf(relayout[k])
                    if new is not None:
                        pmf = new
                        break
            else:
                return no_update, no_update, no_update, no_update
    else:
        return no_update, no_update, no_update, no_update

    # always derive a2 prior as point mass at mean of b2 distro
    a2_pmf = point_pmf(mean_of(pmf))
    return pmf.tolist(), fig_b2_distro(pmf), a2_pmf.tolist(), fig_a2_distro(a2_pmf)


# avg_b2 prior ---------------------------------------------------------------

_A2_DIRECT_TRIGGERS = {
    "btn-a2-nash", "btn-a2-level0", "btn-a2-uniform",
    "btn-a2-normal", "btn-a2-beta", "btn-a2-clear", "a2-distro",
}


@app.callback(
    Output("a2-pmf-store", "data"),
    Output("a2-distro", "figure"),
    Output("pmf-store", "data", allow_duplicate=True),
    Output("distro", "figure", allow_duplicate=True),
    Input("btn-a2-nash", "n_clicks"),
    Input("btn-a2-level0", "n_clicks"),
    Input("btn-a2-uniform", "n_clicks"),
    Input("btn-a2-normal", "n_clicks"),
    Input("btn-a2-beta", "n_clicks"),
    Input("btn-a2-clear", "n_clicks"),
    Input("a2-distro", "relayoutData"),
    Input("pmf-store", "data"),
    State("mu-a2", "value"),
    State("sigma-a2", "value"),
    State("beta-a", "value"),
    State("beta-b", "value"),
    State("a2-pmf-store", "data"),
    prevent_initial_call=True,
)
def update_a2_pmf(_nash, _l0, _uni, _norm, _beta_btn, _clr, relayout,
                  b2_pmf_data, mu_a2, sigma_a2, ba, bb, current):
    trig = ctx.triggered_id
    pmf = np.array(current) if current else _DEFAULT_A2_PMF.copy()

    clear_b2 = trig in _A2_DIRECT_TRIGGERS

    if trig == "btn-a2-nash":
        pmf = point_pmf(835)
    elif trig == "btn-a2-level0":
        pmf = point_pmf(795)
    elif trig == "btn-a2-uniform":
        pmf = uniform_pmf()
    elif trig == "btn-a2-normal":
        pmf = normal_pmf(float(mu_a2 or 840), float(sigma_a2 or 30))
    elif trig == "btn-a2-beta":
        pmf = beta_pmf(float(ba or 2), float(bb or 2))
    elif trig == "btn-a2-clear":
        pass
    elif trig == "a2-distro" and relayout:
        shapes = relayout.get("shapes")
        found = False
        if shapes:
            for sh in reversed(shapes):
                new = path_to_pmf(sh.get("path", ""))
                if new is not None:
                    pmf = new
                    found = True
                    break
        else:
            for k in relayout:
                if k.endswith(".path"):
                    new = path_to_pmf(relayout[k])
                    if new is not None:
                        pmf = new
                        found = True
                        break
        if not found and not shapes:
            # pure pan/zoom — no draw, don't clear either
            return no_update, no_update, no_update, no_update
    elif trig == "pmf-store":
        return no_update, no_update, no_update, no_update
    else:
        return no_update, no_update, no_update, no_update

    b2_fig = fig_b2_empty() if clear_b2 else no_update
    b2_data = uniform_pmf().tolist() if clear_b2 else no_update
    return pmf.tolist(), fig_a2_distro(pmf), b2_data, b2_fig


# main refresh — triggered only by a2-pmf-store and right-col inputs ---------

@app.callback(
    Output("sensitivity", "figure"),
    Output("b1-out", "children"),
    Output("b2-out", "children"),
    Output("ev-out", "children"),
    Output("a2-out", "children"),
    Output("ev-point-out", "children"),
    Output("pen-out", "children"),
    Output("p1-out", "children"),
    Output("p2-out", "children"),
    Output("p0-out", "children"),
    Output("pnl1-out", "children"),
    Output("pnl2-out", "children"),
    Output("ngard-out", "children"),
    Output("nitems-out", "children"),
    Output("grand-out", "children"),
    Input("a2-pmf-store", "data"),
    Input("n-per-atom", "value"),
    Input("items-per", "value"),
)
def refresh(a2_pmf_data, n_per_atom, items_per):
    a2_pmf = np.array(a2_pmf_data) if a2_pmf_data else _DEFAULT_A2_PMF.copy()

    obj_v, b1_star, b2_star, ev_point = optimize_robust(a2_pmf)

    e_a2 = mean_of(a2_pmf)
    _, pnl1, pnl2, p1, p2, pen = ev_scalar(b1_star, b2_star, e_a2)

    n_atom = float(n_per_atom or 0)
    n_item = float(items_per or 0)
    n_gard = 51 * n_atom
    n_tot = n_gard * n_item
    grand = ev_point * n_tot

    return (
        fig_sensitivity(b1_star, b2_star, a2_pmf),
        f"{b1_star}", f"{b2_star}", f"{ev_point:,.1f}",
        f"{e_a2:.1f}", f"{obj_v:,.2f}", f"{pen:.3f}",
        f"{p1*100:.1f}%", f"{p2*100:.1f}%", f"{(1-p1-p2)*100:.1f}%",
        f"{pnl1:,.2f}", f"{pnl2:,.2f}",
        f"{n_gard:,.0f}", f"{n_tot:,.0f}", f"{grand:,.2f}",
    )


if __name__ == "__main__":
    app.run(debug=True)
