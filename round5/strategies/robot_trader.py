from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import json
import math
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

import os

POS_LIMIT = 10
SHOCK_THRESH = float(os.environ.get("SHOCK_THRESH", "0.005"))
ASSETS = ["ROBOT_DISHES", "ROBOT_IRONING"]

# Max ticks to hold before bailing out by crossing spread
MAX_HOLD_TICKS = int(os.environ.get("MAX_HOLD_TICKS", "20"))

# Entry size cap (lower than POS_LIMIT to avoid full-bag risk)
ENTRY_SIZE = int(os.environ.get("ENTRY_SIZE", "10")) 

# Simple MM on pure-GBM assets (proportional inventory skew, fixed half-spread)
MM_ASSETS = ["ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_LAUNDRY"]
MM_QUOTE_SIZE = int(os.environ.get("MM_QUOTE_SIZE", "1"))
MM_HALF_SPREAD = int(os.environ.get("MM_HALF_SPREAD", "2"))    # ticks each side of reservation
MM_MAX_SKEW   = int(os.environ.get("MM_MAX_SKEW", "4"))        # ticks of inv skew at max pos


def best_bid_ask(depth):
    bid = max(depth.buy_orders) if depth.buy_orders else None
    ask = min(depth.sell_orders) if depth.sell_orders else None
    return bid, ask


def mid_of(depth):
    bid, ask = best_bid_ask(depth)
    if bid is None or ask is None:
        return None
    return 0.5 * (bid + ask)


def simple_mm_quotes(mid, pos):
    """
    Linear inventory skew, fixed half-spread.
      skew_ticks = (pos / POS_LIMIT) * MM_MAX_SKEW
      reservation = mid - skew_ticks
      bid = reservation - half_spread
      ask = reservation + half_spread
    Pos at +limit -> reservation pushed down by MM_MAX_SKEW (force flatten).
    """
    skew = (pos / POS_LIMIT) * MM_MAX_SKEW
    reservation = mid - skew
    return reservation - MM_HALF_SPREAD, reservation + MM_HALF_SPREAD


####### TRADER #######

class Trader:
    """
    Shock-reversal trader with smart exit.

    Entry: on shock at t, cross spread to take max position against shock.
           Save anchor_mid (pre-shock mid) as expected reversion target.
    Exit:  on next ticks, post LIMIT at anchor_mid (don't pay spread).
           If reverted (pos closed via fills), great.
           If still holding after MAX_HOLD_TICKS, bail by crossing spread.
    """

    def run(self, state: TradingState):
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}

        result: dict[Symbol, list[Order]] = {}

        for sym in ASSETS:
            depth = state.order_depths.get(sym)
            if depth is None or not depth.buy_orders or not depth.sell_orders:
                continue

            mid = mid_of(depth)
            if mid is None:
                continue

            sm = mem.setdefault(sym, {
                "last_mid": mid,
                "anchor": None,        # pre-shock mid, exit target
                "hold_ticks": 0,       # how long we've been holding
                "entry_dir": 0,        # +1 = long, -1 = short
            })

            last_mid = sm.get("last_mid", mid)
            log_ret = math.log(mid / last_mid) if last_mid > 0 else 0.0

            pos = state.position.get(sym, 0)
            mkt_bid, mkt_ask = best_bid_ask(depth)
            bid_size = abs(depth.buy_orders[mkt_bid])
            ask_size = abs(depth.sell_orders[mkt_ask])

            orders: list[Order] = []

            shock = abs(log_ret) > SHOCK_THRESH

            # Step 1: handle existing position (close at anchor or bail out)
            if pos != 0 and not shock:
                anchor = sm.get("anchor")
                hold = sm.get("hold_ticks", 0) + 1
                sm["hold_ticks"] = hold

                if anchor is None:
                    # Should not happen, but bail anyway
                    if pos > 0:
                        orders.append(Order(sym, mkt_bid, -pos))
                    else:
                        orders.append(Order(sym, mkt_ask, -pos))
                elif hold >= MAX_HOLD_TICKS:
                    # Forced exit: cross spread
                    if pos > 0:
                        orders.append(Order(sym, mkt_bid, -min(pos, bid_size)))
                    else:
                        orders.append(Order(sym, mkt_ask, min(-pos, ask_size)))
                    logger.print(f"{sym} BAIL pos={pos} hold={hold}")
                else:
                    # Patient exit: post limit at anchor
                    if pos > 0:
                        target = max(int(round(anchor)), mkt_bid + 1)
                        orders.append(Order(sym, target, -pos))
                    else:
                        target = min(int(round(anchor)), mkt_ask - 1)
                        orders.append(Order(sym, target, -pos))

            # Step 2: shock fired → take new position against shock
            if shock:
                sm["anchor"] = last_mid
                sm["hold_ticks"] = 0
                sm["entry_dir"] = -1 if log_ret > 0 else 1

                if log_ret > 0:
                    target_pos = -ENTRY_SIZE
                    delta = target_pos - pos
                    qty = max(delta, -bid_size)
                    if qty < 0:
                        orders.append(Order(sym, mkt_bid, qty))
                else:
                    target_pos = ENTRY_SIZE
                    delta = target_pos - pos
                    qty = min(delta, ask_size)
                    if qty > 0:
                        orders.append(Order(sym, mkt_ask, qty))
                logger.print(f"{sym} SHOCK ret={log_ret:+.5f} anchor={last_mid:.1f} pos={pos}")

            # Clear anchor if flat (no MM overlay on DISHES/IRONING)
            if pos == 0 and not shock:
                sm["anchor"] = None
                sm["hold_ticks"] = 0
                sm["entry_dir"] = 0

            if orders:
                result[sym] = orders

            sm["last_mid"] = mid

        # ── Simple MM on GBM assets ────────────────────────────
        for sym in MM_ASSETS:
            depth = state.order_depths.get(sym)
            if depth is None or not depth.buy_orders or not depth.sell_orders:
                continue
            mkt_bid, mkt_ask = best_bid_ask(depth)
            mid = 0.5 * (mkt_bid + mkt_ask)
            pos = state.position.get(sym, 0)

            bid_f, ask_f = simple_mm_quotes(mid, pos)
            bid_price = int(math.floor(bid_f))
            ask_price = int(math.ceil(ask_f))

            # Stay inside market spread, never cross
            bid_price = min(bid_price, mkt_ask - 1)
            ask_price = max(ask_price, mkt_bid + 1)
            if ask_price <= bid_price:
                ask_price = bid_price + 1

            buy_cap = POS_LIMIT - pos
            sell_cap = POS_LIMIT + pos

            mm_orders: list[Order] = []
            if buy_cap > 0 and bid_price > 0:
                mm_orders.append(Order(sym, bid_price, min(MM_QUOTE_SIZE, buy_cap)))
            if sell_cap > 0:
                mm_orders.append(Order(sym, ask_price, -min(MM_QUOTE_SIZE, sell_cap)))
            if mm_orders:
                result[sym] = mm_orders

        out = json.dumps(mem, separators=(",", ":"))
        logger.flush(state, result, 0, out)
        return result, 0, out
