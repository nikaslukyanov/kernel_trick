from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import json
import math
from typing import Any

####### LOGGER #######

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
        return [state.timestamp, trader_data, self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades), state.position, self.compress_observations(state.observations)]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [order_depth.buy_orders, order_depth.sell_orders] for symbol, order_depth in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
                for trade_list in trades.values() for trade in trade_list]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_obs = {product: [obs.bidPrice, obs.askPrice, obs.transportFees, obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex]
                          for product, obs in observations.conversionObservations.items()}
        return [observations.plainValueObservations, conversion_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[order.symbol, order.price, order.quantity] for order_list in orders.values() for order in order_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        low, high = 0, min(len(value), max_length)
        result = ""
        while low <= high:
            midpoint = (low + high) // 2
            candidate = value[:midpoint]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                result = candidate
                low = midpoint + 1
            else:
                high = midpoint - 1
        return result

logger = Logger()

####### CONFIG #######

# AR(2) on VELVETFRUIT_EXTRACT mid, fitted in round3_timeseries.ipynb
ALPHA = 10.727097081004104
BETA1 = 0.840361578360772
BETA2 = 0.15759553490214856
MEAN  = ALPHA / (1.0 - BETA1 - BETA2)   # ≈ 5250.95

# Symbols
UNDERLYING      = "VELVETFRUIT_EXTRACT"
VOUCHER_PREFIX  = "VEV_"
# Trade only the two strikes with biggest detrended-IV residual stds (per options.ipynb): 5200, 5300
VOUCHER_STRIKES = [5200, 5300]

def voucher_symbol(strike: int) -> str:
    return f"{VOUCHER_PREFIX}{strike}"

# Position limits
U_LIMIT = 200    # VELVETFRUIT_EXTRACT
O_LIMIT = 300    # per voucher

# Engine A — underlying mean reversion (template_skew style around MEAN)
VEV_STD       = 32      # TODO: fit std dev of VEV mid from historical data
SKEW_MAX      = 3       # TODO: tune — max ticks of price skew at 1σ from MEAN
# SCALE_FACTOR from pseudocode is implicit in the size-scaling logic below.

# Engine B — voucher market making
T_EXPIRY  = 5.0 / 365.0   # TODO: held constant per user. 5 days till expiry. σ must match unit.
BASE_EDGE = 2             # TODO: tune — half-spread around BS fair value
SPREAD    = 4             # TODO: tune — magnitude of inventory-lean shift
VOUCHER_QUOTE_SIZE = 6    # per-side size on each voucher quote — small or won't get filled

REGRET_THRESHOLD = 1500   # TODO: tune — net portfolio delta threshold for throttle / lean denominator

# Per-strike implied vol. TODO: pull from options.ipynb smile fit (last day-2 parabola or rolling).
# Day-2 parabola (from options.ipynb): a=9.6627, b=-0.0056, c=0.2478, log_mon = ln(MEAN/K).
# Both K=5200 and K=5300 evaluate to ≈ 0.2487 → using that as placeholder.
IV_TABLE = {
    5200: 0.2487,
    5300: 0.2487,
}

####### BLACK-SCHOLES HELPERS #######

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(F: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return F * norm_cdf(d1) - K * norm_cdf(d2)

def bs_call_delta(F: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if F > K else 0.0
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

####### BASE TRADER #######

class ProductTrader:

    def __init__(self, symbol: str, state: TradingState, pos_limit: int):
        self.name      = symbol
        self.pos_limit = pos_limit
        self.state     = state
        order_depth    = state.order_depths.get(symbol, OrderDepth())
        position       = state.position.get(symbol, 0)

        self.initial_position = position
        self.position    = position
        self.orders      = []
        self.buy_volume  = 0
        self.sell_volume = 0

        self.mkt_buy_orders  = {p: abs(v) for p, v in sorted(order_depth.buy_orders.items(),  reverse=True)} if order_depth.buy_orders  else {}
        self.mkt_sell_orders = {p: abs(v) for p, v in sorted(order_depth.sell_orders.items())}              if order_depth.sell_orders else {}
        self.bid_wall = max(self.mkt_buy_orders)  if self.mkt_buy_orders  else None
        self.ask_wall = min(self.mkt_sell_orders) if self.mkt_sell_orders else None
        self.wall_mid = (self.bid_wall + self.ask_wall) / 2.0 if self.bid_wall is not None and self.ask_wall is not None else None

    @property
    def max_allowed_buy_volume(self):
        return self.pos_limit - self.initial_position - self.buy_volume

    @property
    def max_allowed_sell_volume(self):
        return self.pos_limit + self.initial_position - self.sell_volume

    def bid(self, price, quantity, logging=False):
        fill_volume = min(quantity, self.max_allowed_buy_volume)
        if fill_volume <= 0:
            return
        self.orders.append(Order(self.name, int(price), fill_volume))
        self.buy_volume += fill_volume
        self.position   += fill_volume
        if logging:
            logger.print(f"BID {self.name} {int(price)} x{fill_volume}")

    def ask(self, price, quantity, logging=False):
        fill_volume = min(quantity, self.max_allowed_sell_volume)
        if fill_volume <= 0:
            return
        self.orders.append(Order(self.name, int(price), -fill_volume))
        self.sell_volume += fill_volume
        self.position    -= fill_volume
        if logging:
            logger.print(f"ASK {self.name} {int(price)} x{fill_volume}")

    def get_orders(self):
        return {self.name: self.orders}


####### ENGINE A — VELVETFRUIT_EXTRACT MEAN REVERSION #######
# Anchored to wall_mid for fair value, quotes skewed toward AR(2) MEAN (5250.95).
# Below MEAN -> aggressive bid + inflated bid size. Above MEAN -> mirrored ask side.
class VelvetTrader(ProductTrader):

    def __init__(self, state: TradingState, last_wall_mid=None):
        super().__init__(UNDERLYING, state, U_LIMIT)
        if self.wall_mid is None:
            self.wall_mid = last_wall_mid

    def get_orders(self):
        if self.wall_mid is None:
            return {self.name: self.orders}
        fair_value = self.wall_mid

        # 1. TAKING
        for sell_price, sell_volume in self.mkt_sell_orders.items():
            if sell_price <= fair_value - 1:
                self.bid(sell_price, sell_volume)
            elif sell_price <= fair_value and self.initial_position < 0:
                self.bid(sell_price, min(sell_volume, abs(self.initial_position)))
        for buy_price, buy_volume in self.mkt_buy_orders.items():
            if buy_price >= fair_value + 1:
                self.ask(buy_price, buy_volume)
            elif buy_price >= fair_value and self.initial_position > 0:
                self.ask(buy_price, min(buy_volume, self.initial_position))

        # 2. MAKING — skew toward MEAN
        bid_ceiling = int(fair_value) if self.position < 0 else int(fair_value - 1)
        ask_floor   = int(fair_value) if self.position > 0 else int(fair_value + 1)

        thick_bid = next((p for p in self.mkt_buy_orders  if p <= fair_value and self.mkt_buy_orders[p]  > 1), None)
        thick_ask = next((p for p in self.mkt_sell_orders if p >= fair_value and self.mkt_sell_orders[p] > 1), None)

        raw_skew = (fair_value - MEAN) / VEV_STD * SKEW_MAX
        skew = int(round(max(-2 * SKEW_MAX, min(2 * SKEW_MAX, raw_skew))))

        base_bid = (thick_bid + 1) if thick_bid is not None else int(fair_value - 6)
        base_ask = (thick_ask - 1) if thick_ask is not None else int(fair_value + 6)

        bid_price = min(base_bid - skew, bid_ceiling)
        ask_price = max(base_ask - skew, ask_floor)

        std_devs = (fair_value - MEAN) / VEV_STD
        normalized = min(abs(std_devs) / 3, 1.0)
        if std_devs > 0:
            maker_buy_volume  = min(int(6 + (self.max_allowed_buy_volume  - 6) * normalized), self.max_allowed_buy_volume)
            maker_sell_volume = min(6, self.max_allowed_sell_volume)
        else:
            maker_buy_volume  = min(6, self.max_allowed_buy_volume)
            maker_sell_volume = min(int(6 + (self.max_allowed_sell_volume - 6) * normalized), self.max_allowed_sell_volume)

        self.bid(bid_price, maker_buy_volume)
        self.ask(ask_price, maker_sell_volume)
        return {self.name: self.orders}


####### ENGINE B — VOUCHER MARKET MAKING #######
# FV = BS_call(F=MEAN, K=strike, T=T_EXPIRY, σ=IV_TABLE[strike])  — constant per tick.
# Lean quotes against portfolio delta. Throttle = kill bid/ask side past threshold.
class VoucherTrader(ProductTrader):

    def __init__(self, state: TradingState, strike: int, portfolio_delta: float):
        super().__init__(voucher_symbol(strike), state, O_LIMIT)
        self.strike          = strike
        self.portfolio_delta = portfolio_delta

    def get_orders(self):
        sigma = IV_TABLE.get(self.strike)
        if sigma is None:
            return {self.name: self.orders}

        fv = bs_call(MEAN, self.strike, T_EXPIRY, sigma)

        inventory_lean = self.portfolio_delta / REGRET_THRESHOLD
        my_bid = fv - BASE_EDGE - inventory_lean * SPREAD
        my_ask = fv + BASE_EDGE - inventory_lean * SPREAD

        # 4. THROTTLING — skip the offending side rather than posting useless 0 / 999999 orders
        post_bid = self.portfolio_delta <=  REGRET_THRESHOLD
        post_ask = self.portfolio_delta >= -REGRET_THRESHOLD

        if post_bid and my_bid > 0:
            self.bid(int(round(my_bid)), VOUCHER_QUOTE_SIZE)
        if post_ask:
            self.ask(int(round(my_ask)), VOUCHER_QUOTE_SIZE)

        return {self.name: self.orders}


####### MAIN #######

class Trader:

    def __init__(self):
        pass

    def _portfolio_delta(self, state: TradingState) -> float:
        # Underlying contributes delta = position. Each voucher contributes pos × BS delta.
        delta = float(state.position.get(UNDERLYING, 0))
        for k in VOUCHER_STRIKES:
            pos = state.position.get(voucher_symbol(k), 0)
            if pos == 0:
                continue
            sigma = IV_TABLE.get(k)
            if sigma is None:
                continue
            delta += pos * bs_call_delta(MEAN, k, T_EXPIRY, sigma)
        return delta

    def run(self, state: TradingState):
        result: dict[Symbol, list[Order]] = {}
        trader_data = json.loads(state.traderData) if state.traderData else {}

        # AR(2) state for future scalping (S_next currently unused by either engine)
        prev_mid      = trader_data.get("prev_mid")
        prev_prev_mid = trader_data.get("prev_prev_mid")
        s_next = None
        if prev_mid is not None and prev_prev_mid is not None:
            s_next = BETA1 * prev_mid + BETA2 * prev_prev_mid + ALPHA

        portfolio_delta = self._portfolio_delta(state)
        logger.print(f"delta={portfolio_delta:.1f} mean={MEAN:.1f} s_next={s_next}")

        # Engine A
        try:
            vt = VelvetTrader(state, last_wall_mid=trader_data.get("vev_wall_mid"))
            result.update(vt.get_orders())
            if vt.wall_mid is not None:
                trader_data["prev_prev_mid"] = trader_data.get("prev_mid")
                trader_data["prev_mid"]      = vt.wall_mid
                trader_data["vev_wall_mid"]  = vt.wall_mid
        except Exception as e:
            logger.print(f"ERROR {UNDERLYING}: {e}")

        # Engine B — one VoucherTrader per strike
        for k in VOUCHER_STRIKES:
            try:
                vrt = VoucherTrader(state, k, portfolio_delta)
                result.update(vrt.get_orders())
            except Exception as e:
                logger.print(f"ERROR {voucher_symbol(k)}: {e}")

        out_trader_data = json.dumps(trader_data)
        logger.flush(state, result, 0, out_trader_data)
        return result, 0, out_trader_data
