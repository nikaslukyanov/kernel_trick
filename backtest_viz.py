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

    fills = []
    for trade in data.get('tradeHistory', []):
        if trade['buyer'] == 'SUBMISSION':
            fills.append({'timestamp': trade['timestamp'], 'symbol': trade['symbol'], 'price': trade['price'], 'qty': trade['quantity']})
        elif trade['seller'] == 'SUBMISSION':
            fills.append({'timestamp': trade['timestamp'], 'symbol': trade['symbol'], 'price': trade['price'], 'qty': -trade['quantity']})
    fills_df = pd.DataFrame(fills) if fills else pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'qty'])

    return activities, orders_df, fills_df


THEMES = {
    'light': dict(
        bg='white', paper='white', grid='#e0e0e0', text='#333',
        mid='#333333', annotation='#333',
    ),
    'dark': dict(
        bg='#111111', paper='#111111', grid='#2a2a2a', text='#ccc',
        mid='#eeeeee', annotation='#ccc',
    ),
}


def build_figure(activities, orders_df, fills_df, sample=1, theme='light'):
    t = THEMES[theme]
    products = list(activities['product'].unique())
    fig = make_subplots(rows=len(products), cols=1, subplot_titles=products,
                        shared_xaxes=False, vertical_spacing=0.08)

    for i, product in enumerate(products, start=1):
        mkt = activities[activities['product'] == product].copy()
        mkt_plot = mkt.iloc[::sample]

        def fmt(row):
            parts = []
            if pd.notna(row['mid_price']):
                parts.append(f"<b>mid</b>: {row['mid_price']:.1f}")
            if pd.notna(row['bid_price_1']):
                parts.append(f"<b>mkt bid</b>: {row['bid_price_1']:.0f}")
            if pd.notna(row['ask_price_1']):
                parts.append(f"<b>mkt ask</b>: {row['ask_price_1']:.0f}")
            return "<br>".join(parts)

        hover = mkt_plot.apply(fmt, axis=1)

        fig.add_trace(go.Scatter(x=mkt_plot['timestamp'], y=mkt_plot['mid_price'],   mode='lines', name='mid',     line=dict(color=t['mid'],  width=1),   legendgroup='mid',     showlegend=(i==1), hovertext=hover, hoverinfo='x+text'), row=i, col=1)
        fig.add_trace(go.Scatter(x=mkt_plot['timestamp'], y=mkt_plot['bid_price_1'], mode='lines', name='mkt bid', line=dict(color='#00cc44', width=1.5), legendgroup='mkt bid', showlegend=(i==1), hoverinfo='skip'), row=i, col=1)
        fig.add_trace(go.Scatter(x=mkt_plot['timestamp'], y=mkt_plot['ask_price_1'], mode='lines', name='mkt ask', line=dict(color='#ff3333', width=1.5), legendgroup='mkt ask', showlegend=(i==1), hoverinfo='skip'), row=i, col=1)

        prod_orders = orders_df[orders_df['symbol'] == product] if len(orders_df) else orders_df
        buys  = prod_orders[prod_orders['qty'] > 0]
        sells = prod_orders[prod_orders['qty'] < 0]

        fig.add_trace(go.Scatter(
            x=buys['timestamp'], y=buys['price'], mode='markers', name='our bid',
            marker=dict(symbol='triangle-up', size=7, color='#4488ff', opacity=0.35),
            legendgroup='our bid', showlegend=(i==1),
            hovertemplate='<b>our bid</b><br>ts: %{x}<br>price: %{y}<br>qty: %{customdata}<extra></extra>',
            customdata=buys['qty'],
        ), row=i, col=1)

        fig.add_trace(go.Scatter(
            x=sells['timestamp'], y=sells['price'], mode='markers', name='our ask',
            marker=dict(symbol='triangle-down', size=7, color='#ffaa00', opacity=0.35),
            legendgroup='our ask', showlegend=(i==1),
            hovertemplate='<b>our ask</b><br>ts: %{x}<br>price: %{y}<br>qty: %{customdata}<extra></extra>',
            customdata=sells['qty'].abs(),
        ), row=i, col=1)

        prod_fills = fills_df[fills_df['symbol'] == product] if len(fills_df) else fills_df
        buy_fills  = prod_fills[prod_fills['qty'] > 0]
        sell_fills = prod_fills[prod_fills['qty'] < 0]

        fig.add_trace(go.Scatter(
            x=buy_fills['timestamp'], y=buy_fills['price'], mode='markers', name='fill buy',
            marker=dict(symbol='circle', size=9, color='#4488ff', line=dict(color='white', width=1.5)),
            legendgroup='fill buy', showlegend=(i==1),
            hovertemplate='<b>fill buy</b><br>ts: %{x}<br>price: %{y}<br>qty: %{customdata}<extra></extra>',
            customdata=buy_fills['qty'],
        ), row=i, col=1)

        fig.add_trace(go.Scatter(
            x=sell_fills['timestamp'], y=sell_fills['price'], mode='markers', name='fill sell',
            marker=dict(symbol='circle', size=9, color='#ffaa00', line=dict(color='white', width=1.5)),
            legendgroup='fill sell', showlegend=(i==1),
            hovertemplate='<b>fill sell</b><br>ts: %{x}<br>price: %{y}<br>qty: %{customdata}<extra></extra>',
            customdata=sell_fills['qty'].abs(),
        ), row=i, col=1)

        fig.update_yaxes(title_text='price', row=i, col=1, gridcolor=t['grid'], color=t['text'])
        fig.update_xaxes(title_text='timestamp', row=i, col=1, gridcolor=t['grid'], color=t['text'])
        for ann in fig.layout.annotations:
            ann.font.color = t['annotation']

    fig.update_layout(
        height=500 * len(products),
        hovermode='x',
        margin=dict(t=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0,
                    font=dict(color=t['text']), bgcolor='rgba(0,0,0,0)'),
        plot_bgcolor=t['bg'],
        paper_bgcolor=t['paper'],
        font=dict(color=t['text']),
    )
    return fig


app = Dash(__name__)
app.layout = html.Div(id='page', children=[
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
        html.Label('Resolution:', style={'fontFamily': 'monospace', 'marginLeft': '24px', 'marginRight': '8px'}),
        dcc.RadioItems(
            id='sample-rate',
            options=[
                {'label': 'Full', 'value': '1'},
                {'label': '1/5',  'value': '5'},
                {'label': '1/10', 'value': '10'},
                {'label': '1/25', 'value': '25'},
            ],
            value='1',
            inline=True,
            style={'fontFamily': 'monospace'},
            labelStyle={'marginRight': '12px'},
        ),
        html.Label('Theme:', style={'fontFamily': 'monospace', 'marginLeft': '24px', 'marginRight': '8px'}),
        dcc.RadioItems(
            id='theme',
            options=[{'label': 'Light', 'value': 'light'}, {'label': 'Dark', 'value': 'dark'}],
            value='light',
            inline=True,
            style={'fontFamily': 'monospace'},
            labelStyle={'marginRight': '12px'},
        ),
    ], style={'display': 'flex', 'alignItems': 'center', 'margin': '0 16px 16px'}),
    dcc.Graph(id='chart', config={'scrollZoom': True}),
])

@app.callback(
    Output('chart', 'figure'),
    Output('page', 'style'),
    Input('log-picker', 'value'),
    Input('sample-rate', 'value'),
    Input('theme', 'value'),
)
def update(log_file, sample_rate, theme):
    activities, orders_df, fills_df = load_log(log_file)
    fig = build_figure(activities, orders_df, fills_df, int(sample_rate), theme)
    bg = '#111111' if theme == 'dark' else 'white'
    color = '#cccccc' if theme == 'dark' else '#333333'
    return fig, {'backgroundColor': bg, 'color': color, 'minHeight': '100vh'}


if __name__ == '__main__':
    app.run(debug=False, port=8050)
    print('Open http://localhost:8050')
