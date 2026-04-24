# Order Sizing
For hydrogel, taker order sizes do not go above 6 shares. As such, we our top of the orderbook quotes should be 6 shares. No more, only less is necessary for max inventory purposes.

## Spoofing orderbook imbalance with remaining shares?
"""
A well-placed offer does not shout. It fits. Price, timing, and order size all play a role in making your trade more appealing, even when the differences seem small

Now, do not rush this part. Take a moment and imagine how your order looks from the other side of the trade. If it makes sense to them, it will probably work for you
""" - Advisor

Imagine an orderbook where we are the only bid, and there is a lot of ask volume, that bid probabably looks attractive?
We can (maybe) alter orderbook imbalance to encourage trades in our direction.
i.e.: Lets say the price is currently 9950. We know the mean is 10k. As such we want to go long and get our bid hit.
We can place large volume at the ask, and small volume (6 shares) at the bid. This makes it seem like there is selling pressure and the price will go down.

## Similar flow across different days?