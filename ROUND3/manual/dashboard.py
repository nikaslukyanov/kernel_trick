"""
Gardener Guild dashboard.

Run:
    pip3 install dash plotly numpy
    python3 ROUND3/manual/dashboard.py

Opens http://127.0.0.1:8050
"""
from __future__ import annotations

import re
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go

# ---- model -----------------------------------------------------------------

GRID = np.arange(670, 921, 5)
N = len(GRID)               # 51
P_RES = np.full(N, 1 / N)   # fixed uniform reserve PMF per spec


def uniform_pmf() -> np.ndarray:
    return np.full(N, 1 / N)


def normal_pmf(mu: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-6)
    w = np.exp(-0.5 * ((GRID - mu) / sigma) ** 2)
    return w / w.sum()


def bimodal_pmf() -> np.ndarray:
    w = np.exp(-0.5 * ((GRID - 760) / 15) ** 2) + np.exp(-0.5 * ((GRID - 870) / 15) ** 2)
    return w / w.sum()


def point_pmf(x: float) -> np.ndarray:
    p = np.zeros(N)
    p[int(np.argmin(np.abs(GRID - x)))] = 1.0
    return p


def mean_of(pmf: np.ndarray) -> float:
    return float((GRID * pmf).sum())


def ev(b1: int, b2: int, a2: float):
    p1 = P_RES[GRID <= b1].sum()
    p2 = P_RES[(GRID > b1) & (GRID <= b2)].sum()
    if b2 >= 920:
        pen = 0.0  # no margin anyway
    else:
        pen = 1.0 if b2 >= a2 else ((920 - a2) / (920 - b2)) ** 3
    pnl1 = p1 * (920 - b1)
    pnl2 = p2 * (920 - b2) * pen
    return pnl1 + pnl2, pnl1, pnl2, p1, p2, pen


def optimize(a2: float):
    best_v, best_b1, best_b2 = -1.0, int(GRID[0]), int(GRID[-1])
    for b1 in GRID:
        for b2 in GRID[GRID >= b1]:
            v, *_ = ev(int(b1), int(b2), a2)
            if v > best_v:
                best_v, best_b1, best_b2 = v, int(b1), int(b2)
    return best_v, best_b1, best_b2


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
    # average duplicate x
    ux, inv = np.unique(xs, return_inverse=True)
    uy = np.zeros_like(ux)
    cnt = np.zeros_like(ux)
    for i, j in enumerate(inv):
        uy[j] += ys[i]
        cnt[j] += 1
    uy = uy / np.maximum(cnt, 1)
    if len(ux) < 2:
        return None
    # interpolate onto GRID, clamp endpoints
    pmf = np.interp(GRID, ux, uy, left=0.0, right=0.0)
    pmf = np.clip(pmf, 0, None)
    s = pmf.sum()
    if s <= 0:
        return None
    return pmf / s


# ---- styling ---------------------------------------------------------------

BG = "#0b1220"
PANEL = "#111a2b"
TEAL = "#4fd1c5"
GOLD = "#d4a84a"
MUTED = "#7b8aa1"
TEXT = "#e6edf7"
GREEN = "#6fbf87"

FONT = "ui-monospace, SFMono-Regular, Menlo, monospace"


def fig_distro(pmf: np.ndarray) -> go.Figure:
    cdf = np.cumsum(pmf)
    fig = go.Figure()
    fig.add_bar(
        x=GRID, y=pmf, marker_color=TEAL, opacity=0.75, name="density f(p)",
        hovertemplate="b2=%{x}<br>p=%{y:.4f}<extra></extra>",
    )
    fig.add_scatter(
        x=GRID, y=cdf * pmf.max() / max(cdf.max(), 1e-9),
        mode="lines", line=dict(color=GOLD, dash="dash"), name="CDF (scaled)",
        hoverinfo="skip",
    )
    a2 = mean_of(pmf)
    fig.add_vline(x=a2, line=dict(color=GOLD, width=1),
                  annotation_text=f"avg_b2 = {a2:.2f}",
                  annotation_position="top left",
                  annotation_font_color=GOLD)
    fig.update_layout(
        dragmode="drawopenpath",
        newshape=dict(line=dict(color=GOLD, width=2)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family=FONT, color=TEXT, size=11),
        margin=dict(l=40, r=20, t=20, b=30), height=320,
        xaxis=dict(gridcolor="#1b2740", zeroline=False, title="b2 bid (counterparty)"),
        yaxis=dict(gridcolor="#1b2740", zeroline=False, title="density f(p)"),
        showlegend=False, bargap=0.02,
    )
    return fig


def fig_ev_curve(pmf: np.ndarray, b1_star: int, b2_star: int) -> go.Figure:
    a2 = mean_of(pmf)
    ys = []
    for b2 in GRID[GRID >= b1_star]:
        v, *_ = ev(b1_star, int(b2), a2)
        ys.append(v)
    xs = GRID[GRID >= b1_star]
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=TEAL, width=2),
                    name="E[PnL]", hovertemplate="b2=%{x}<br>EV=%{y:.2f}<extra></extra>")
    fig.add_scatter(x=[b2_star], y=[ev(b1_star, b2_star, a2)[0]],
                    mode="markers", marker=dict(color=GOLD, size=10),
                    hoverinfo="skip", showlegend=False)
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family=FONT, color=TEXT, size=11),
        margin=dict(l=40, r=20, t=20, b=30), height=220,
        xaxis=dict(gridcolor="#1b2740", zeroline=False, title=f"b2 (b1 held at joint-optimum {b1_star})"),
        yaxis=dict(gridcolor="#1b2740", zeroline=False, title="E[PnL] per counterparty"),
        showlegend=False,
    )
    return fig


# ---- layout ----------------------------------------------------------------

app = Dash(__name__)
app.title = "Gardener Guild"

panel_style = {
    "background": PANEL, "padding": "16px 18px", "borderRadius": "8px",
    "border": "1px solid #1b2740",
}
h_style = {"margin": "0 0 4px 0", "color": TEXT, "fontSize": "15px", "fontWeight": 600}
sub_style = {"margin": "0 0 12px 0", "color": MUTED, "fontSize": "11px"}
btn_style = {
    "background": "transparent", "color": TEXT, "border": f"1px solid {MUTED}",
    "padding": "4px 10px", "marginRight": "6px", "marginTop": "6px",
    "borderRadius": "4px", "fontFamily": FONT, "fontSize": "11px",
    "cursor": "pointer",
}
readout_row = {"display": "flex", "justifyContent": "space-between", "padding": "6px 0",
               "borderBottom": "1px solid #1b2740", "fontSize": "12px"}

app.layout = html.Div(
    style={"background": BG, "color": TEXT, "minHeight": "100vh",
           "padding": "20px", "fontFamily": FONT},
    children=[
        dcc.Store(id="pmf-store", data=uniform_pmf().tolist()),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1.4fr 1fr",
                        "gap": "16px"}, children=[
            # LEFT COL
            html.Div(children=[
                html.Div(style=panel_style, children=[
                    html.H3("Opponent b2 distribution", style=h_style),
                    html.P("Draw inside the plot (open path tool in toolbar) to sculpt density "
                           "over other players' second bids. Or use presets. Reserve prices "
                           "are fixed uniform per spec; only avg_b2 flexes.", style=sub_style),
                    dcc.Graph(id="distro", config={
                        "displayModeBar": True,
                        "modeBarButtonsToAdd": ["drawopenpath", "eraseshape"],
                        "displaylogo": False,
                    }),
                    html.Div(children=[
                        html.Button("UNIFORM", id="btn-uniform", style=btn_style),
                        html.Button("BIMODAL", id="btn-bimodal", style=btn_style),
                        html.Button("POINT @ 920", id="btn-point920", style=btn_style),
                        html.Button("POINT @ 795", id="btn-point795", style=btn_style),
                        html.Button("CLEAR SHAPES", id="btn-clear", style=btn_style),
                    ]),
                    html.Div(style={"marginTop": "12px", "display": "flex",
                                    "gap": "8px", "alignItems": "center",
                                    "flexWrap": "wrap"}, children=[
                        html.Span("NORMAL", style={"color": GOLD, "fontSize": "11px"}),
                        html.Span("μ", style={"color": MUTED, "fontSize": "11px"}),
                        dcc.Input(id="mu", type="number", value=795, step=1,
                                  style={"width": "70px", "background": BG,
                                         "color": TEXT, "border": f"1px solid {MUTED}",
                                         "padding": "3px 6px", "borderRadius": "3px",
                                         "fontFamily": FONT}),
                        html.Span("σ", style={"color": MUTED, "fontSize": "11px"}),
                        dcc.Input(id="sigma", type="number", value=25, step=1,
                                  style={"width": "70px", "background": BG,
                                         "color": TEXT, "border": f"1px solid {MUTED}",
                                         "padding": "3px 6px", "borderRadius": "3px",
                                         "fontFamily": FONT}),
                        html.Button("APPLY NORMAL", id="btn-normal",
                                    style={**btn_style, "border": f"1px solid {GOLD}",
                                           "color": GOLD}),
                    ]),
                ]),
                html.Div(style={**panel_style, "marginTop": "16px"}, children=[
                    html.H3("Expected PnL vs b2", style=h_style),
                    html.P("Fix b1 at its optimum; sweep b2. Gold dot = global optimum.",
                           style=sub_style),
                    dcc.Graph(id="ev-curve", config={"displayModeBar": False}),
                ]),
            ]),
            # RIGHT COL
            html.Div(style=panel_style, children=[
                html.H3("Recommended bids", style=h_style),
                html.P("Updates live as you redraw distribution.", style=sub_style),
                html.Div(style={"border": f"1px solid {GOLD}", "padding": "14px",
                                "borderRadius": "6px", "marginTop": "10px"}, children=[
                    html.Div("— OPTIMAL BIDS —", style={"color": GOLD, "fontSize": "10px",
                                                         "letterSpacing": "2px",
                                                         "textAlign": "center"}),
                    html.Div(style={"display": "flex", "justifyContent": "space-around",
                                    "marginTop": "10px"}, children=[
                        html.Div(children=[
                            html.Div("b1*", style={"color": MUTED, "fontSize": "10px",
                                                    "letterSpacing": "2px"}),
                            html.Div(id="b1-out", style={"color": TEAL, "fontSize": "30px",
                                                          "fontWeight": 600}),
                        ]),
                        html.Div(children=[
                            html.Div("b2*", style={"color": MUTED, "fontSize": "10px",
                                                    "letterSpacing": "2px"}),
                            html.Div(id="b2-out", style={"color": GREEN, "fontSize": "30px",
                                                          "fontWeight": 600}),
                        ]),
                        html.Div(children=[
                            html.Div("E[PnL]", style={"color": MUTED, "fontSize": "10px",
                                                       "letterSpacing": "2px"}),
                            html.Div(id="ev-out", style={"color": GOLD, "fontSize": "30px",
                                                          "fontWeight": 600}),
                        ]),
                    ]),
                ]),
                html.Div(style={"marginTop": "18px"}, children=[
                    html.Div("DISTRIBUTION STATS", style={"color": MUTED, "fontSize": "10px",
                                                           "letterSpacing": "2px",
                                                           "marginBottom": "6px"}),
                    html.Div(style=readout_row, children=[
                        html.Span("avg_b2 (mean of opponents)"),
                        html.Span(id="a2-out", style={"color": GOLD}),
                    ]),
                    html.Div(style=readout_row, children=[
                        html.Span("mode of distribution"),
                        html.Span(id="mode-out", style={"color": TEAL}),
                    ]),
                ]),
                html.Div(style={"marginTop": "18px"}, children=[
                    html.Div("TRADE BREAKDOWN", style={"color": MUTED, "fontSize": "10px",
                                                        "letterSpacing": "2px",
                                                        "marginBottom": "6px"}),
                    html.Div(style=readout_row, children=[
                        html.Span("P(r ≤ b1*) — b1-leg hit rate"),
                        html.Span(id="p1-out", style={"color": TEAL}),
                    ]),
                    html.Div(style=readout_row, children=[
                        html.Span("P(b1 < r ≤ b2*) — b2-leg hit rate"),
                        html.Span(id="p2-out", style={"color": GREEN}),
                    ]),
                    html.Div(style=readout_row, children=[
                        html.Span("P(no trade)"),
                        html.Span(id="p0-out", style={"color": MUTED}),
                    ]),
                    html.Div(style=readout_row, children=[
                        html.Span("penalty factor at b2*"),
                        html.Span(id="pen-out", style={"color": GOLD}),
                    ]),
                ]),
                html.Div(style={"marginTop": "18px"}, children=[
                    html.Div("EV DECOMPOSITION (per counterparty)",
                             style={"color": MUTED, "fontSize": "10px",
                                    "letterSpacing": "2px", "marginBottom": "6px"}),
                    html.Div(style=readout_row, children=[
                        html.Span("b1-leg:  P·(920 − b1)"),
                        html.Span(id="pnl1-out", style={"color": TEAL}),
                    ]),
                    html.Div(style=readout_row, children=[
                        html.Span("b2-leg:  P·(920 − b2)·penalty"),
                        html.Span(id="pnl2-out", style={"color": GREEN}),
                    ]),
                    html.Div(style=readout_row, children=[
                        html.Span("total E[PnL] per counterparty"),
                        html.Span(id="total-out", style={"color": GOLD, "fontWeight": 600}),
                    ]),
                ]),
                html.Div(style={"marginTop": "18px"}, children=[
                    html.Div("SCALE → TOTAL PnL", style={"color": MUTED, "fontSize": "10px",
                                                         "letterSpacing": "2px",
                                                         "marginBottom": "6px"}),
                    html.Div(style={"display": "flex", "gap": "10px",
                                    "alignItems": "center", "flexWrap": "wrap",
                                    "marginBottom": "8px"}, children=[
                        html.Span("gardeners per reserve atom", style={"color": MUTED,
                                                                        "fontSize": "11px"}),
                        dcc.Input(id="n-per-atom", type="number", value=1, step=1, min=0,
                                  style={"width": "70px", "background": BG, "color": TEXT,
                                         "border": f"1px solid {MUTED}", "padding": "3px 6px",
                                         "borderRadius": "3px", "fontFamily": FONT}),
                    ]),
                    html.Div(style={"display": "flex", "gap": "10px",
                                    "alignItems": "center", "flexWrap": "wrap",
                                    "marginBottom": "10px"}, children=[
                        html.Span("items per gardener", style={"color": MUTED,
                                                                "fontSize": "11px"}),
                        dcc.Input(id="items-per", type="number", value=1, step=1, min=0,
                                  style={"width": "70px", "background": BG, "color": TEXT,
                                         "border": f"1px solid {MUTED}", "padding": "3px 6px",
                                         "borderRadius": "3px", "fontFamily": FONT}),
                    ]),
                    html.Div(style=readout_row, children=[
                        html.Span("gardener count  (51 × n_per_atom)"),
                        html.Span(id="ngard-out", style={"color": TEAL}),
                    ]),
                    html.Div(style=readout_row, children=[
                        html.Span("total items  (× items_per)"),
                        html.Span(id="nitems-out", style={"color": TEAL}),
                    ]),
                    html.Div(style=readout_row, children=[
                        html.Span("TOTAL E[PnL]"),
                        html.Span(id="grand-out", style={"color": GOLD, "fontSize": "16px",
                                                          "fontWeight": 700}),
                    ]),
                ]),
            ]),
        ]),
    ],
)


# ---- callbacks -------------------------------------------------------------

@app.callback(
    Output("pmf-store", "data"),
    Output("distro", "figure", allow_duplicate=True),
    Input("btn-uniform", "n_clicks"),
    Input("btn-bimodal", "n_clicks"),
    Input("btn-point920", "n_clicks"),
    Input("btn-point795", "n_clicks"),
    Input("btn-normal", "n_clicks"),
    Input("btn-clear", "n_clicks"),
    Input("distro", "relayoutData"),
    State("mu", "value"),
    State("sigma", "value"),
    State("pmf-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def update_pmf(_u, _b, _p920, _p795, _n, _c, relayout, mu, sigma, current):
    trig = ctx.triggered_id
    pmf = np.array(current) if current else uniform_pmf()
    if trig == "btn-uniform":
        pmf = uniform_pmf()
    elif trig == "btn-bimodal":
        pmf = bimodal_pmf()
    elif trig == "btn-point920":
        pmf = point_pmf(920)
    elif trig == "btn-point795":
        pmf = point_pmf(795)
    elif trig == "btn-normal":
        pmf = normal_pmf(float(mu or 795), float(sigma or 25))
    elif trig == "btn-clear":
        pass  # keep pmf; just re-render fresh fig (shapes gone)
    elif trig == "distro" and relayout:
        # prefer newest drawn shape
        shapes = relayout.get("shapes")
        if shapes:
            for sh in reversed(shapes):
                path = sh.get("path")
                if path:
                    new = path_to_pmf(path)
                    if new is not None:
                        pmf = new
                        break
        else:
            # handle per-shape edits like shapes[0].path
            path_keys = [k for k in relayout.keys() if k.endswith(".path")]
            for k in path_keys:
                new = path_to_pmf(relayout[k])
                if new is not None:
                    pmf = new
                    break
            else:
                return no_update, no_update
    else:
        return no_update, no_update
    return pmf.tolist(), fig_distro(pmf)


@app.callback(
    Output("distro", "figure"),
    Output("ev-curve", "figure"),
    Output("b1-out", "children"),
    Output("b2-out", "children"),
    Output("ev-out", "children"),
    Output("a2-out", "children"),
    Output("mode-out", "children"),
    Output("p1-out", "children"),
    Output("p2-out", "children"),
    Output("p0-out", "children"),
    Output("pen-out", "children"),
    Output("pnl1-out", "children"),
    Output("pnl2-out", "children"),
    Output("total-out", "children"),
    Output("ngard-out", "children"),
    Output("nitems-out", "children"),
    Output("grand-out", "children"),
    Input("pmf-store", "data"),
    Input("n-per-atom", "value"),
    Input("items-per", "value"),
)
def refresh(pmf_data, n_per_atom, items_per):
    pmf = np.array(pmf_data) if pmf_data else uniform_pmf()
    a2 = mean_of(pmf)
    v_star, b1_star, b2_star = optimize(a2)
    total, pnl1, pnl2, p1, p2, pen = ev(b1_star, b2_star, a2)
    mode_x = int(GRID[int(np.argmax(pmf))])
    n_atom = float(n_per_atom or 0)
    n_item = float(items_per or 0)
    n_gard = 51 * n_atom
    n_tot_items = n_gard * n_item
    grand = total * n_tot_items
    return (
        fig_distro(pmf),
        fig_ev_curve(pmf, b1_star, b2_star),
        f"{b1_star}",
        f"{b2_star}",
        f"{v_star:,.1f}",
        f"{a2:.2f}",
        f"{mode_x}",
        f"{p1*100:.1f}%",
        f"{p2*100:.1f}%",
        f"{(1-p1-p2)*100:.1f}%",
        f"{pen:.3f}",
        f"{pnl1:,.2f}",
        f"{pnl2:,.2f}",
        f"{total:,.2f}",
        f"{n_gard:,.0f}",
        f"{n_tot_items:,.0f}",
        f"{grand:,.2f}",
    )


if __name__ == "__main__":
    app.run(debug=True)
