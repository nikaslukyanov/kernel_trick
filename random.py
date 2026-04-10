from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import jsonpickle


class Trader:
    POSITION_LIMITS = {
        # fill these in for the round
        "PRODUCT1": 20,
        "PRODUCT2": 20,
    }

    DEFAULT_EDGE = {
        # minimum mispricing required before trading
        "PRODUCT1": 1,
        "PRODUCT2": 1,
    }

    def bid(self):
        # Only used in Round 2. Safe to leave here for every round.
        return 15

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # restore saved state from previous iteration
        if state.traderData:
            memory = jsonpickle.decode(state.traderData)
        else:
            memory = {
                "last_mid": {},
            }

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 0)

            fair_value = self.compute_fair_value(product, order_depth, memory)
            edge = self.DEFAULT_EDGE.get(product, 1)

            # how much more we are allowed to buy or sell
            buy_capacity = limit - position
            sell_capacity = limit + position

            # 1. take favorable asks
            if order_depth.sell_orders:
                for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
                    ask_volume = -ask_qty  # sell quantities are negative in the book
                    if ask_price < fair_value - edge and buy_capacity > 0:
                        trade_qty = min(ask_volume, buy_capacity)
                        orders.append(Order(product, ask_price, trade_qty))
                        buy_capacity -= trade_qty

            # 2. take favorable bids
            if order_depth.buy_orders:
                for bid_price, bid_qty in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price > fair_value + edge and sell_capacity > 0:
                        trade_qty = min(bid_qty, sell_capacity)
                        orders.append(Order(product, bid_price, -trade_qty))
                        sell_capacity -= trade_qty

            # 3. optional market making around fair value
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            if best_bid is not None and best_ask is not None:
                # post inside the spread if possible
                buy_quote = min(best_bid + 1, int(fair_value - edge))
                sell_quote = max(best_ask - 1, int(fair_value + edge))

                if buy_quote < best_ask and buy_capacity > 0:
                    post_qty = min(2, buy_capacity)
                    orders.append(Order(product, buy_quote, post_qty))

                if sell_quote > best_bid and sell_capacity > 0:
                    post_qty = min(2, sell_capacity)
                    orders.append(Order(product, sell_quote, -post_qty))

                memory["last_mid"][product] = (best_bid + best_ask) / 2

            result[product] = orders

        traderData = jsonpickle.encode(memory)
        return result, conversions, traderData

    def compute_fair_value(self, product: str, order_depth: OrderDepth, memory) -> float:
        """
        Replace this with your signal.
        For now: use simple midprice if both sides exist,
        otherwise fall back to last seen mid or 0.
        """
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2

        return memory["last_mid"].get(product, 0)