from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Order:
    price: int
    qty: int
    time: int  # 0 for book, 1 for me

def solve_auction(bids: List[Order], asks: List[Order], fair_value: float, fee: float = 0.0):
    # Potential clearing prices are any price levels existing in the combined book
    prices = sorted(list(set([o.price for o in bids] + [o.price for o in asks])))
    
    best_vol = -1
    clearing_price = 0
    
    # 1. Determine the clearing price (Standard Exchange Logic)
    # The price that maximizes volume. If tie, minimizes imbalance.
    for p in prices:
        total_demand = sum(o.qty for o in bids if o.price >= p)
        total_supply = sum(o.qty for o in asks if o.price <= p)
        volume = min(total_demand, total_supply)
        
        if volume > best_vol:
            best_vol = volume
            clearing_price = p
        elif volume == best_vol:
            # Tie-break: price closest to Fair Value (common in these challenges)
            if abs(p - fair_value) < abs(clearing_price - fair_value):
                clearing_price = p

    # 2. Calculate Fill for "Me" (Time 1)
    # We only get filled if our price is better than clearing_price, 
    # OR if we are at clearing_price and there's surplus after Time 0 orders.
    
    my_fill = 0
    
    # Logic for Buy Order
    my_buy = next((o for o in bids if o.time == 1), None)
    if my_buy:
        if my_buy.price > clearing_price:
            # We are aggressive, we get filled as much as possible
            total_supply_at_p = sum(o.qty for o in asks if o.price <= clearing_price)
            higher_priority_demand = sum(o.qty for o in bids if o.price > clearing_price or (o.price == clearing_price and o.time < my_buy.time))
            my_fill = max(0, min(my_buy.qty, total_supply_at_p - higher_priority_demand))
        elif my_buy.price == clearing_price:
            total_supply_at_p = sum(o.qty for o in asks if o.price <= clearing_price)
            higher_priority_demand = sum(o.qty for o in bids if o.price >= clearing_price and o.time < my_buy.time)
            my_fill = max(0, min(my_buy.qty, total_supply_at_p - higher_priority_demand))
        
        pnl = my_fill * (fair_value - clearing_price - fee)
        return pnl, my_fill, clearing_price

    # Logic for Sell Order
    my_sell = next((o for o in asks if o.time == 1), None)
    if my_sell:
        if my_sell.price < clearing_price:
            total_demand_at_p = sum(o.qty for o in bids if o.price >= clearing_price)
            higher_priority_supply = sum(o.qty for o in asks if o.price < clearing_price or (o.price == clearing_price and o.time < my_sell.time))
            my_fill = max(0, min(my_sell.qty, total_demand_at_p - higher_priority_supply))
        elif my_sell.price == clearing_price:
            total_demand_at_p = sum(o.qty for o in bids if o.price >= clearing_price)
            higher_priority_supply = sum(o.qty for o in asks if o.price <= clearing_price and o.time < my_sell.time)
            my_fill = max(0, min(my_sell.qty, total_demand_at_p - higher_priority_supply))
            
        pnl = my_fill * (clearing_price - fair_value - fee)
        return pnl, my_fill, clearing_price

    return 0, 0, clearing_price

def find_optimal(book_bids, book_asks, fv, fee, p_range, q_range):
    best_order = {"pnl": -1}
    
    for side in ["buy", "sell"]:
        for p in p_range:
            for q in q_range:
                bids = [Order(pb, qb, 0) for pb, qb in book_bids]
                asks = [Order(pa, qa, 0) for pa, qa in book_asks]
                
                if side == "buy": bids.append(Order(p, q, 1))
                else: asks.append(Order(p, q, 1))
                
                pnl, fill, cp = solve_auction(bids, asks, fv, fee)
                if pnl > best_order["pnl"]:
                    best_order = {"side": side, "price": p, "qty": q, "pnl": pnl, "fill": fill, "clear": cp}
    return best_order

# --- RUNNING THE SEARCH ---

# Dryland Flax
flax_bids = [(30, 30000), (29, 5000), (28, 12000), (27, 28000)]
flax_asks = [(28, 40000), (31, 20000), (32, 20000), (33, 30000)]
res_flax = find_optimal(flax_bids, flax_asks, 30.0, 0.0, range(26, 35), range(1000, 50001, 1000))

# Ember Mushroom
ember_bids = [(20, 43000), (19, 17000), (18, 6000), (17, 5000), (16,10000), (15, 5000), (14,10000), (13,7000)]
ember_asks = [(12, 20000), (13, 25000), (14, 35000), (15, 6000), (16,5000), (18, 10000), (19,12000)]
res_ember = find_optimal(ember_bids, ember_asks, 20.0, 0.1, range(11, 22), range(1000, 100001, 1000))

print(f"Dryland Flax: {res_flax}")
print(f"Ember Mushroom: {res_ember}")