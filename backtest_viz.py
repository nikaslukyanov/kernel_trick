import json
import glob
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
from pathlib import Path
from dash import Dash, dcc, html, Input, Output

LOG_FILES = sorted(glob.glob('imc-prosperity-4-backtester/backtests/*.log'))

def load_log(path):
    with open(path) as f:
        data = json.load(f)

    activities = pd.read_csv(StringIO(data['activitiesLog']), sep=';')
    activities['timestamp'] = activities['timestamp'].astype(int)
    for col in ['bid_price_1', 'ask_price_1', 'mid_price']:
        activities[col] = activities[col].replace(0, float('nan'))

    own_orders = []
    for entry in data['logs']:
        raw = entry['lambdaLog']
        if not raw:
            continue
        parsed = json.loads(raw)
        ts = parsed[0][0]
        for symbol, price, qty in parsed[1]:
            own_orders.append({'timestamp': ts, 'symbol': symbol, 'price': price, 'qty': qty})

    orders_df = pd.DataFrame(own_orders) if own_orders else pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'qty'])
    return activities, orders_df


def build_figure(activities, orders_df):
    def tick_summary(prod_orders):
        rows = {}
        for ts, group in prod_orders.groupby('timestamp'):
            bids = group[group['qty'] > 0].sort_values('price', ascending=False)
            asks = group[group['qty'] < 0].sort_values('price')
            rows[ts] = {
                'our_bid_p': float(bids['price'].iloc[0])    if len(bids) else None,
                'our_bid_v': float(bids['qty'].iloc[0])      if len(bids) else None,
                'our_ask_p': float(asks['price'].iloc[0])    if len(asks) else None,
                'our_ask_v': float(abs(asks['qty'].iloc[0])) if len(asks) else None,
            }
        return pd.DataFrame.from_dict(rows, orient='index').reset_index().rename(columns={'index': 'timestamp'})

    products = list(activities['product'].unique())
    fig = make_subplots(rows=len(products), cols=1, subplot_titles=products,
                        shared_xaxes=False, vertical_spacing=0.08)

    for i, product in enumerate(products, start=1):
        mkt = activities[activities['product'] == product].copy()

        if len(orders_df):
            summary = tick_summary(orders_df[orders_df['symbol'] == product])
            mkt = mkt.merge(summary, on='timestamp', how='left')
        else:
            mkt[['our_bid_p', 'our_bid_v', 'our_ask_p', 'our_ask_v']] = None

        def fmt(row):
            parts = []
            if pd.notna(row['mid_price']):
                parts.append(f"<b>mid</b>: {row['mid_price']:.1f}")
            if pd.notna(row.get('our_bid_p')):
                parts.append(f"<b>our bid</b>: {row['our_bid_p']:.0f} × {row['our_bid_v']:.0f}")
            if pd.notna(row.get('our_ask_p')):
                parts.append(f"<b>our ask</b>: {row['our_ask_p']:.0f} × {row['our_ask_v']:.0f}")
            return "<br>".join(parts)

        hover = mkt.apply(fmt, axis=1)

        fig.add_trace(go.Scatter(x=mkt['timestamp'], y=mkt['mid_price'],   mode='lines', name='mid',    line=dict(color='black',     width=1),   legendgroup='g', showlegend=(i==1), hovertext=hover, hoverinfo='x+text'), row=i, col=1)
        fig.add_trace(go.Scatter(x=mkt['timestamp'], y=mkt['bid_price_1'], mode='lines', name='bid L1', line=dict(color='steelblue', width=0.8), legendgroup='g', showlegend=(i==1), opacity=0.7, hoverinfo='skip'), row=i, col=1)
        fig.add_trace(go.Scatter(x=mkt['timestamp'], y=mkt['ask_price_1'], mode='lines', name='ask L1', line=dict(color='salmon',    width=0.8), legendgroup='g', showlegend=(i==1), opacity=0.7, hoverinfo='skip'), row=i, col=1)

        prod_orders = orders_df[orders_df['symbol'] == product] if len(orders_df) else orders_df
        buys  = prod_orders[prod_orders['qty'] > 0]
        sells = prod_orders[prod_orders['qty'] < 0]

        fig.add_trace(go.Scatter(
            x=buys['timestamp'], y=buys['price'], mode='markers', name='my bid',
            marker=dict(symbol='triangle-up', size=6, color='blue', opacity=0.8),
            legendgroup='g', showlegend=(i==1),
            hovertemplate='<b>my bid</b><br>ts: %{x}<br>price: %{y}<br>qty: %{customdata}<extra></extra>',
            customdata=buys['qty'],
        ), row=i, col=1)

        fig.add_trace(go.Scatter(
            x=sells['timestamp'], y=sells['price'], mode='markers', name='my ask',
            marker=dict(symbol='triangle-down', size=6, color='orange', opacity=0.8),
            legendgroup='g', showlegend=(i==1),
            hovertemplate='<b>my ask</b><br>ts: %{x}<br>price: %{y}<br>qty: %{customdata}<extra></extra>',
            customdata=sells['qty'].abs(),
        ), row=i, col=1)

        fig.update_yaxes(title_text='price', row=i, col=1)
        fig.update_xaxes(title_text='timestamp', row=i, col=1)

    fig.update_layout(
        height=500 * len(products),
        hovermode='x',
        margin=dict(t=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
    )
    return fig


app = Dash(__name__)
app.layout = html.Div([
    html.H3('Backtest Visualizer', style={'fontFamily': 'monospace', 'margin': '16px'}),
    html.Div([
        html.Label('Log file:', style={'fontFamily': 'monospace', 'marginRight': '8px'}),
        dcc.Dropdown(
            id='log-picker',
            options=[{'label': Path(f).name, 'value': f} for f in LOG_FILES],
            value=LOG_FILES[-1],
            clearable=False,
            style={'width': '420px', 'fontFamily': 'monospace'},
        ),
    ], style={'display': 'flex', 'alignItems': 'center', 'margin': '0 16px 16px'}),
    dcc.Graph(id='chart', config={'scrollZoom': True}),
])

@app.callback(Output('chart', 'figure'), Input('log-picker', 'value'))
def update(log_file):
    activities, orders_df = load_log(log_file)
    return build_figure(activities, orders_df)


if __name__ == '__main__':
    app.run(debug=False, port=8050)
    print('Open http://localhost:8050')
