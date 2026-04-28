from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import json
import math
from typing import Any


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [depth.buy_orders, depth.sell_orders] for symbol, depth in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
                for trade_list in trades.values() for trade in trade_list]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_obs = {
            product: [
                obs.bidPrice,
                obs.askPrice,
                obs.transportFees,
                obs.exportTariff,
                obs.importTariff,
                obs.sugarPrice,
                obs.sunlightIndex,
            ]
            for product, obs in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conversion_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[order.symbol, order.price, order.quantity] for order_list in orders.values() for order in order_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        result = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                result = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return result


logger = Logger()

HYDRO = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
OPTION_BY_STRIKE = {strike: f"VEV_{strike}" for strike in STRIKES}

LIMIT = {HYDRO: 200, VELVET: 200}
for _strike in STRIKES:
    LIMIT[OPTION_BY_STRIKE[_strike]] = 300

USE_IV_OVERLAY = False
ENABLE_OPTIONS = True
ENABLE_BETA_LAYER = False
ENABLE_OPTION_PASSIVE_BETA = True
CP_DECAY = 0.90
ANCHOR = {HYDRO: 9990.95, VELVET: 5250.71}

# Compact fit from the Round 4 price series: options as extensions of VELVET.
BETA_FIT = {
    4000: (-4001.3, 1.000),
    4500: (-4499.8, 1.000),
    5000: (-4800.7, 0.963),
    5100: (-4435.5, 0.876),
    5200: (-3509.2, 0.686),
    5300: (-2214.9, 0.430),
    5400: (-926.4, 0.179),
    5500: (-414.0, 0.080),
}

LOSER_BUYERS = {
    HYDRO: {"Mark 38"},
    VELVET: {"Mark 55", "Mark 67"},
    "VEV_4000": {"Mark 38"},
    "VEV_5300": {"Mark 01"},
    "VEV_5400": {"Mark 01"},
    "VEV_5500": {"Mark 01"},
}
LOSER_SELLERS = {
    HYDRO: {"Mark 38"},
    VELVET: {"Mark 55"},
    "VEV_4000": {"Mark 38"},
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def best_bid_ask(depth: OrderDepth) -> tuple[int | None, int | None]:
    bid = max(depth.buy_orders) if depth.buy_orders else None
    ask = min(depth.sell_orders) if depth.sell_orders else None
    return bid, ask


def mid_price(depth: OrderDepth) -> float | None:
    bid, ask = best_bid_ask(depth)
    if bid is None or ask is None:
        return None
    return 0.5 * (bid + ask)


def append(values: list[Any], value: float, limit: int) -> list[float]:
    result = [float(v) for v in values[-limit + 1:]]
    result.append(round(float(value), 4))
    return result


def rolling_z(memory: dict[str, Any], bucket: str, key: str, value: float, length: int = 120, warmup: int = 30) -> tuple[float, float, float, int]:
    table = memory.setdefault(bucket, {})
    hist = [float(v) for v in table.get(key, [])]
    if len(hist) >= warmup:
        mean = sum(hist) / len(hist)
        var = sum((x - mean) ** 2 for x in hist) / max(1, len(hist) - 1)
        std = math.sqrt(max(var, 1e-6))
        z = (value - mean) / max(std, 0.08)
    else:
        mean, std, z = value, 1.0, 0.0
    table[key] = append(hist, value, length)
    return clamp(z, -4.0, 4.0), mean, std, len(hist)


def room_to_buy(state: TradingState, symbol: str, pending: int = 0) -> int:
    return LIMIT[symbol] - state.position.get(symbol, 0) - pending


def room_to_sell(state: TradingState, symbol: str, pending: int = 0) -> int:
    return LIMIT[symbol] + state.position.get(symbol, 0) + pending


class Trader:
    def fresh_memory(self) -> dict[str, Any]:
        return {
            "tick": 0,
            "ema": {},
            "last_mid": {},
            "hist": {},
            "cp": {},
        }

    def load_memory(self, data: str) -> dict[str, Any]:
        if data:
            try:
                loaded = json.loads(data)
                if isinstance(loaded, dict):
                    base = self.fresh_memory()
                    for key, value in base.items():
                        loaded.setdefault(key, value)
                    return loaded
            except Exception:
                pass
        return self.fresh_memory()

    def save_memory(self, memory: dict[str, Any]) -> str:
        return json.dumps(memory, separators=(",", ":"))

    def update_counterparty_flow(self, state: TradingState, memory: dict[str, Any]) -> None:
        cp = memory.setdefault("cp", {})
        for symbol, flow in list(cp.items()):
            flow["buy"] = round(float(flow.get("buy", 0.0)) * CP_DECAY, 4)
            flow["sell"] = round(float(flow.get("sell", 0.0)) * CP_DECAY, 4)
            if flow["buy"] + flow["sell"] < 0.05:
                cp.pop(symbol, None)

        for trades in state.market_trades.values():
            for trade in trades:
                flow = cp.setdefault(trade.symbol, {"buy": 0.0, "sell": 0.0})
                if trade.seller in LOSER_SELLERS.get(trade.symbol, set()):
                    flow["buy"] = round(float(flow.get("buy", 0.0)) + trade.quantity, 4)
                if trade.buyer in LOSER_BUYERS.get(trade.symbol, set()):
                    flow["sell"] = round(float(flow.get("sell", 0.0)) + trade.quantity, 4)

    def flow(self, memory: dict[str, Any], symbol: str) -> tuple[float, float]:
        flow = memory.get("cp", {}).get(symbol, {})
        return float(flow.get("buy", 0.0)), float(flow.get("sell", 0.0))

    def trade_mean_reverter(self, state: TradingState, memory: dict[str, Any], symbol: str, base_edge: float) -> list[Order]:
        depth = state.order_depths.get(symbol)
        if depth is None:
            return []
        bid, ask = best_bid_ask(depth)
        mid = mid_price(depth)
        if bid is None or ask is None or mid is None:
            return []

        prev_mid = float(memory["last_mid"].get(symbol, mid))
        memory["last_mid"][symbol] = mid
        ema = float(memory["ema"].get(symbol, ANCHOR[symbol]))
        ema = 0.985 * ema + 0.015 * mid
        memory["ema"][symbol] = ema
        momentum = mid - prev_mid
        anchor = ANCHOR[symbol]
        z, _, _, hist_len = rolling_z(memory, "hist", symbol, mid - anchor, 200, 60)
        buy_flow, sell_flow = self.flow(memory, symbol)

        if symbol == HYDRO:
            fair = mid + 0.42 * (anchor - ema) - 0.15 * momentum + clamp((buy_flow - sell_flow) / 12.0, -2.0, 2.0)
            quote_size = 16
        else:
            fair = mid + 0.16 * (anchor - mid) - 0.12 * momentum + clamp((buy_flow - sell_flow) / 12.0, -2.0, 2.0)
            memory["velvet_edge"] = round(fair - mid, 4)
            quote_size = 10
        pos = state.position.get(symbol, 0)
        orders: list[Order] = []
        pending = 0

        if ask <= fair - base_edge and room_to_buy(state, symbol, pending) > 0:
            qty = min(abs(depth.sell_orders[ask]), room_to_buy(state, symbol, pending), 26)
            orders.append(Order(symbol, ask, qty))
            pending += qty
        if bid >= fair + base_edge and room_to_sell(state, symbol, pending) > 0:
            qty = min(abs(depth.buy_orders[bid]), room_to_sell(state, symbol, pending), 26)
            orders.append(Order(symbol, bid, -qty))
            pending -= qty

        if hist_len >= 20 and abs(pos + pending) < 175:
            if mid < fair - 1.0 and room_to_buy(state, symbol, pending) > 0:
                price = min(bid + 1, ask - 1)
                if price < ask:
                    qty = min(room_to_buy(state, symbol, pending), max(3, quote_size - max(0, pos) // 15))
                    orders.append(Order(symbol, price, qty))
                    pending += qty
            elif mid > fair + 1.0 and room_to_sell(state, symbol, pending) > 0:
                price = max(ask - 1, bid + 1)
                if price > bid:
                    qty = min(room_to_sell(state, symbol, pending), max(3, quote_size + min(0, pos) // 15))
                    orders.append(Order(symbol, price, -qty))
                    pending -= qty

        return orders

    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call(self, spot: float, strike: int, t: float, sigma: float) -> float:
        if t <= 0 or sigma <= 0:
            return max(0.0, spot - strike)
        vol = sigma * math.sqrt(t)
        d1 = (math.log(max(spot, 1e-9) / strike) + 0.5 * sigma * sigma * t) / vol
        d2 = d1 - vol
        return spot * self.norm_cdf(d1) - strike * self.norm_cdf(d2)

    def trade_options(self, state: TradingState, memory: dict[str, Any], spot: float | None) -> dict[str, list[Order]]:
        result: dict[str, list[Order]] = {}
        if spot is None or not ENABLE_OPTIONS:
            return result

        for strike in STRIKES:
            symbol = OPTION_BY_STRIKE[strike]
            depth = state.order_depths.get(symbol)
            if depth is None:
                result[symbol] = []
                continue
            bid, ask = best_bid_ask(depth)
            mid = mid_price(depth)
            if bid is None or ask is None or mid is None:
                result[symbol] = []
                continue

            orders: list[Order] = []
            pending = 0
            buy_flow, sell_flow = self.flow(memory, symbol)
            pos = state.position.get(symbol, 0)
            spread = ask - bid

            if ENABLE_BETA_LAYER and strike in BETA_FIT:
                alpha, beta = BETA_FIT[strike]
                fair = alpha + beta * spot
                resid = mid - fair
                z, _, std, hist_len = rolling_z(memory, "hist", f"{symbol}:beta", resid, 180, 45)

                if USE_IV_OVERLAY and strike >= 5000:
                    t = max(0.2, 4.0 - state.timestamp / 1_000_000.0) / 365.0
                    iv_fair = self.bs_call(spot, strike, t, 0.22)
                    iv_z, _, _, _ = rolling_z(memory, "hist", f"{symbol}:iv", mid - iv_fair, 180, 45)
                    z = clamp(0.75 * z + 0.25 * iv_z, -4.0, 4.0)

                velvet_edge = float(memory.get("velvet_edge", 0.0))
                if USE_IV_OVERLAY:
                    velvet_edge -= 0.20 * z
                entry = 0.85 if strike <= 5200 else 1.25
                cap = 170 if strike <= 5200 else 90
                if hist_len >= 20 and abs(velvet_edge) > entry:
                    target = int(round(clamp(velvet_edge * beta * 42.0, -cap, cap)))
                    diff = target - (pos + pending)
                    if diff > 0 and room_to_buy(state, symbol, pending) > 0:
                        qty = min(diff, abs(depth.sell_orders[ask]), room_to_buy(state, symbol, pending), 12)
                        orders.append(Order(symbol, ask, qty))
                        pending += qty
                    elif diff < 0 and room_to_sell(state, symbol, pending) > 0:
                        qty = min(-diff, abs(depth.buy_orders[bid]), room_to_sell(state, symbol, pending), 12)
                        orders.append(Order(symbol, bid, -qty))
                        pending -= qty

                if hist_len >= 20 and abs(velvet_edge) > entry * 0.7:
                    if velvet_edge > 0 and room_to_buy(state, symbol, pending) > 0:
                        price = min(bid + 1, ask - 1)
                        if price < ask:
                            qty = min(room_to_buy(state, symbol, pending), 8)
                            orders.append(Order(symbol, price, qty))
                            pending += qty
                    elif velvet_edge < 0 and room_to_sell(state, symbol, pending) > 0:
                        price = max(ask - 1, bid + 1)
                        if price > bid:
                            qty = min(room_to_sell(state, symbol, pending), 8)
                            orders.append(Order(symbol, price, -qty))
                            pending -= qty

            if ENABLE_OPTION_PASSIVE_BETA and strike in BETA_FIT and spread >= 2:
                alpha, beta = BETA_FIT[strike]
                fair = alpha + beta * spot
                resid = mid - fair
                z, _, _, hist_len = rolling_z(memory, "hist", f"{symbol}:passive_beta", resid, 180, 30)
                if USE_IV_OVERLAY and strike >= 5000:
                    t = max(0.2, 4.0 - state.timestamp / 1_000_000.0) / 365.0
                    iv_fair = self.bs_call(spot, strike, t, 0.22)
                    iv_z, _, _, _ = rolling_z(memory, "hist", f"{symbol}:passive_iv", mid - iv_fair, 180, 30)
                    z = clamp(0.8 * z + 0.2 * iv_z, -4.0, 4.0)

                base_qty = 2 if strike in {4000, 4500} else 2
                if strike in {5300, 5400, 5500}:
                    base_qty = 4
                cap = 80 if strike <= 5200 else 60
                bid_price = min(bid + 1, ask - 1)
                ask_price = max(ask - 1, bid + 1)

                # Quote with the beta fair. We improve the touch by one tick so the
                # synthetic trader flow can choose us ahead of the visible queue.
                allow_bid = strike <= 5100 or (strike == 5200 and z < -2.0)
                if allow_bid and bid_price < ask and pos + pending < cap and room_to_buy(state, symbol, pending) > 0:
                    if fair >= bid_price + 0.6 and (z < 0.8 or hist_len < 30):
                        qty = min(base_qty, room_to_buy(state, symbol, pending), cap - (pos + pending))
                        if qty > 0:
                            orders.append(Order(symbol, bid_price, qty))
                            pending += qty
                if ask_price > bid and pos + pending > -cap and room_to_sell(state, symbol, pending) > 0:
                    loser_buyer = symbol in LOSER_BUYERS
                    if fair <= ask_price - 0.2 or loser_buyer or z > 0.5:
                        qty = min(base_qty, room_to_sell(state, symbol, pending), cap + (pos + pending))
                        if loser_buyer and sell_flow > buy_flow + 2:
                            qty = min(qty + 1, room_to_sell(state, symbol, pending), cap + (pos + pending))
                        if qty > 0:
                            orders.append(Order(symbol, ask_price, -qty))
                            pending -= qty

            # Trader-id layer: make the known losing traders interact with our quotes.
            if symbol in LOSER_BUYERS and room_to_sell(state, symbol, pending) > 0:
                qty = 2 if symbol in {"VEV_5300", "VEV_5400", "VEV_5500"} else 1
                cap = 45 if symbol in {"VEV_5300", "VEV_5400", "VEV_5500"} else 25
                if pos + pending > -cap:
                    price = max(ask - 1, bid + 1) if spread >= 2 else ask
                    orders.append(Order(symbol, price, -min(qty, room_to_sell(state, symbol, pending))))
                    pending -= min(qty, room_to_sell(state, symbol, pending))
            if symbol in LOSER_SELLERS and room_to_buy(state, symbol, pending) > 0:
                qty = 1
                if pos + pending < 25:
                    price = min(bid + 1, ask - 1) if spread >= 2 else bid
                    orders.append(Order(symbol, price, min(qty, room_to_buy(state, symbol, pending))))

            result[symbol] = orders
        return result

    def run(self, state: TradingState):
        memory = self.load_memory(state.traderData)
        memory["tick"] = int(memory.get("tick", 0)) + 1
        self.update_counterparty_flow(state, memory)

        orders: dict[Symbol, list[Order]] = {}
        orders[HYDRO] = self.trade_mean_reverter(state, memory, HYDRO, 3.0)
        orders[VELVET] = self.trade_mean_reverter(state, memory, VELVET, 1.4)

        spot = mid_price(state.order_depths[VELVET]) if VELVET in state.order_depths else None
        orders.update(self.trade_options(state, memory, spot))

        trader_data = self.save_memory(memory)
        conversions = 0
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
