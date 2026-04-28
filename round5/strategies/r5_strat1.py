from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from utils import Utils
import json
import math
from typing import Any
import numpy as np

####################################################################################
# SETTING UP THE LOGGER
####################################################################################

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

####################################################################################
# SOME UTILITY FUNCTIONS
####################################################################################

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

####################################################################################
# SETTING UP THE CONSTANTS NEEDED FOR TRADING
####################################################################################

HYDRO = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
OPTION_BY_STRIKE = {strike: f"VEV_{strike}" for strike in STRIKES}

LIMIT = 10

ENABLE_OPTIONS = True
CP_DECAY = 0.90
ANCHOR = {HYDRO: 9990.95, VELVET: 5250.71}

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

####################################################################################
# TRADER CLASS
####################################################################################
logger = Logger()

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

    def trade_pp(self, state: TradingState, memory: dict[str, Any], symbol: str, base_edge: float) -> list[Order]:
        '''
        This is a method implementing our trading strategy for Purification Pebbles
        So far we know that the large and small ones are negatively correlated
        '''
        pass


    def run(self, state: TradingState):
        memory = self.load_memory(state.traderData)
        memory["tick"] = int(memory.get("tick", 0)) + 1
        self.update_counterparty_flow(state, memory)

        orders: dict[Symbol, list[Order]] = {}


        trader_data = self.save_memory(memory)
        conversions = 0
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
