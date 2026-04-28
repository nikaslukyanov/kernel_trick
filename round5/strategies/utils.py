import math
class Utils:
    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call(self, spot: float, strike: int, t: float, sigma: float) -> float:
        if t <= 0 or sigma <= 0:
            return max(0.0, spot - strike)
        vol = sigma * math.sqrt(t)
        d1 = (math.log(max(spot, 1e-9) / strike) + 0.5 * sigma * sigma * t) / vol
        d2 = d1 - vol
        return spot * self.norm_cdf(d1) - strike * self.norm_cdf(d2)

    def bs_delta(self, spot: float, strike: int, t: float, sigma: float = 0.22) -> float:
        if t <= 0 or sigma <= 0:
            return 1.0 if spot >= strike else 0.0
        vol = sigma * math.sqrt(t)
        d1 = (math.log(max(spot, 1e-9) / strike) + 0.5 * sigma * sigma * t) / vol
        return self.norm_cdf(d1)

    def avellaneda_stoikov(self, mid: float, pos: int, sigma: float, gamma: float = 0.002, kappa: float = 1.5) -> tuple[float, float]:
        sigma = max(sigma, 0.5)
        reservation = mid - pos * gamma * sigma * sigma
        half_spread = (gamma * sigma * sigma + (2.0 / gamma) * math.log(1.0 + gamma / kappa)) / 2.0
        half_spread = max(half_spread, 1.0)
        return reservation - half_spread, reservation + half_spread