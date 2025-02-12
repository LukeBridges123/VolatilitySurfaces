from scipy import stats
from scipy.optimize import root_scalar
import math

def call_price(price, time, strike, expiry, volatility, interest_rate):
    ttm = expiry - time
    d1 = math.log(price/strike) + (interest_rate + volatility**2 / 2) * ttm
    d1 = d1 / (volatility * math.sqrt(ttm))
    d2 = math.log(price / strike) + (interest_rate - volatility ** 2 / 2) * ttm
    d2 = d2 / (volatility * math.sqrt(ttm))

    prob1 = stats.norm.cdf(d1)
    prob2 = stats.norm.cdf(d2)

    return price * prob1 - strike * math.exp(-1 * interest_rate * ttm) * prob2

def implied_volatility(price, time, strike, expiry, interest_rate, option_price):
    def to_solve(sigma):
        return call_price(price, time, strike, expiry, sigma, interest_rate) - option_price
    return root_scalar(to_solve, x0=1, x1=2).root
print(call_price(32, 0, 35, 0.5, 0.3, 0.05))
print(implied_volatility(32, 0, 25, 0.5, 0.05, 7.9))