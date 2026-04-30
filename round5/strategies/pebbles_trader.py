from datamodel import Order, OrderDepth, ProsperityEncoder, Symbol, TradingState
import json
import os
from typing import Any


####### LOGGER #######

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep=" ", end="\n"):
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state, orders, conversions, trader_data):
        base = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        m = (self.max_log_length - base) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, m)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, m),
            self.truncate(self.logs, m),
        ]))
        self.logs = ""

    def compress_state(self, s, td):
        return [s.timestamp, td, self.compress_listings(s.listings),
                self.compress_order_depths(s.order_depths), self.compress_trades(s.own_trades),
                self.compress_trades(s.market_trades), s.position, self.compress_observations(s.observations)]

    def compress_listings(self, l):
        return [[x.symbol, x.product, x.denomination] for x in l.values()]

    def compress_order_depths(self, od):
        return {s: [d.buy_orders, d.sell_orders] for s, d in od.items()}

    def compress_trades(self, t):
        return [[x.symbol, x.price, x.quantity, x.buyer, x.seller, x.timestamp] for tl in t.values() for x in tl]

    def compress_observations(self, o):
        co = {p: [x.bidPrice, x.askPrice, x.transportFees, x.exportTariff, x.importTariff, x.sugarPrice, x.sunlightIndex]
              for p, x in o.conversionObservations.items()}
        return [o.plainValueObservations, co]

    def compress_orders(self, o):
        return [[x.symbol, x.price, x.quantity] for ol in o.values() for x in ol]

    def to_json(self, v):
        return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, v, m):
        lo, hi, r = 0, min(len(v), m), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            c = v[:mid]
            if len(c) < len(v): c += "..."
            if len(json.dumps(c)) <= m:
                r = c; lo = mid + 1
            else:
                hi = mid - 1
        return r


logger = Logger()


####### CONFIG #######

import math

POS_LIMIT = 10
PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]

# MM params (passive limit only — never cross spread)
HALF_SPREAD = int(os.environ.get("HALF_SPREAD", "1"))   # ticks each side of fair
MAX_SKEW    = int(os.environ.get("MAX_SKEW", "4"))      # ticks shift at full inventory
QUOTE_SIZE  = int(os.environ.get("QUOTE_SIZE", "10"))

# Per-pebble z-score lean
ROLL_WIN    = int(os.environ.get("ROLL_WIN", "200"))    # ticks for rolling mean / std
WARMUP      = int(os.environ.get("WARMUP", "30"))
LEAN        = float(os.environ.get("LEAN", "1.0"))      # how strongly to pull fair toward rolling mean
Z_CAP       = float(os.environ.get("Z_CAP", "4.0"))     # clamp |z| to avoid extreme skews

# Basket-shock overlay: when sum_dev exceeds threshold, lean EXTRA on top of per-pebble
SUM_ANCHOR    = 50_000
BASKET_LEAN   = float(os.environ.get("BASKET_LEAN", "1.0"))   # 0 = ignore basket, 1+ = stronger
BASKET_THRESH = float(os.environ.get("BASKET_THRESH", "4.0")) # only apply basket lean when |sum_dev| >= this


def best_bid_ask(depth):
    bid = max(depth.buy_orders) if depth.buy_orders else None
    ask = min(depth.sell_orders) if depth.sell_orders else None
    return bid, ask


def mid_of(depth):
    bid, ask = best_bid_ask(depth)
    if bid is None or ask is None:
        return None
    return 0.5 * (bid + ask)


####### TRADER #######

class Trader:
    """
    MM each pebble with fair = rolling mean over last W ticks.
    Each pebble treated INDEPENDENTLY:
      z_P = (mid_P - mean_P) / std_P
      fair_P = mean_P  (i.e., mid_P - z_P × std_P × LEAN, with LEAN=1)
      reservation = fair_P - (pos/POS_LIMIT) × MAX_SKEW    (inv skew)
      bid/ask quoted around reservation, passive only.
    """

    def run(self, state: TradingState):
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}
        hist = mem.setdefault("hist", {})

        result: dict[Symbol, list[Order]] = {}

        depths = {s: state.order_depths.get(s) for s in PEBBLES}
        if any(d is None or not d.buy_orders or not d.sell_orders for d in depths.values()):
            return {}, 0, json.dumps(mem, separators=(",", ":"))

        mids = {s: mid_of(d) for s, d in depths.items()}
        if any(m is None for m in mids.values()):
            return {}, 0, json.dumps(mem, separators=(",", ":"))

        total_mid = sum(mids.values())
        sum_dev = total_mid - SUM_ANCHOR
        basket_active = abs(sum_dev) >= BASKET_THRESH
        logger.print(f"sum_dev={sum_dev:+.2f} basket_active={basket_active}")

        for sym in PEBBLES:
            depth = depths[sym]
            mid = mids[sym]
            pos = state.position.get(sym, 0)
            mkt_bid, mkt_ask = best_bid_ask(depth)

            # Update rolling history for this pebble
            h = hist.get(sym, [])
            h.append(round(mid, 2))
            if len(h) > ROLL_WIN:
                h = h[-ROLL_WIN:]
            hist[sym] = h

            # Per-pebble z-score lean
            if len(h) >= WARMUP:
                m = sum(h) / len(h)
                v = sum((x - m) ** 2 for x in h) / max(1, len(h) - 1)
                std = math.sqrt(max(v, 1e-9))
                z = max(-Z_CAP, min(Z_CAP, (mid - m) / std))
                fair = mid - z * std * LEAN
            else:
                fair = mid       # warmup — no signal yet

            # Basket-shock overlay: extra lean when sum_dev exceeds threshold
            if basket_active:
                weight = mid / total_mid
                fair -= sum_dev * weight * BASKET_LEAN

            # Inventory skew: when long, push reservation down → quotes drift down → ask becomes attractive
            reservation = fair - (pos / POS_LIMIT) * MAX_SKEW

            bid_price = int(reservation - HALF_SPREAD)
            ask_price = int(reservation + HALF_SPREAD + 0.999)  # ceil

            # Stay inside market spread, never cross
            bid_price = min(bid_price, mkt_ask - 1)
            ask_price = max(ask_price, mkt_bid + 1)
            if ask_price <= bid_price:
                ask_price = bid_price + 1

            buy_cap = POS_LIMIT - pos
            sell_cap = POS_LIMIT + pos

            orders: list[Order] = []
            if buy_cap > 0 and bid_price > 0:
                orders.append(Order(sym, bid_price, min(QUOTE_SIZE, buy_cap)))
            if sell_cap > 0:
                orders.append(Order(sym, ask_price, -min(QUOTE_SIZE, sell_cap)))
            if orders:
                result[sym] = orders

        out = json.dumps(mem, separators=(",", ":"))
        logger.flush(state, result, 0, out)
        return result, 0, out
