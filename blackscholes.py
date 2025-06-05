from scipy import stats
from scipy.optimize import root_scalar
import math

def black_scholes_call_price(expiry, strike, initial_price, interest, volatility):
    """
    Calculate the Black-Scholes price of a call option.
    :param expiry: time till expiry (in years). Equivalently, time when the option expires, assuming the current
    time is 0.
    :param strike: Strike price for the option.
    :param initial_price: Current price of the underlying asset (stock, etc.)
    :param interest: Annual interest rate.
    :param volatility: Volatility of the underlying asset.
    :return: Price of the option.
    """
    d1 = math.log(initial_price / strike) + (interest + volatility ** 2 / 2) * expiry
    d1 = d1 / (volatility * math.sqrt(expiry))
    d2 = math.log(initial_price / strike) + (interest - volatility ** 2 / 2) * expiry
    d2 = d2 / (volatility * math.sqrt(expiry))

    prob1 = stats.norm.cdf(d1)
    prob2 = stats.norm.cdf(d2)

    return initial_price * prob1 - strike * math.exp(-1 * interest * expiry) * prob2

def black_scholes_implied_volatility(expiry, strike, initial_price, interest, option_price):
    """
    Calculate the Black-Scholes implied volatility of an option at a given price.
    :param expiry: time till expiry (in years). Equivalently, time when the option expires, assuming the current
    time is 0.
    :param strike: Strike price for the option.
    :param initial_price: Current price of the underlying asset (stock, etc.)
    :param interest: Annual interest rate.
    :param option_price: Current price of the option.
    :return: Implied volatility of the stock.
    """
    def to_solve(sigma):
        return black_scholes_call_price(expiry, strike, initial_price, interest, sigma) - option_price
    return root_scalar(to_solve, x0=1, x1=2).root