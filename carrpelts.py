from scipy.optimize import root_scalar
from scipy.integrate import quad
from numpy import inf
import math


def forward_price(initial_price, interest, expiry):
    return initial_price * math.exp(interest * expiry)


def time_discount(interest, expiry):
    return math.exp(-1 * interest * expiry)


def find_z(h, tau, expiry, strike, initial_price, interest):
    """
    Find the `z` number used in calculating the Carr-Pelts option price.
    :param h: One of the Carr-Pelts parameters; a convex function of one variable.
    :param tau: One of the Carr-Pelts parameters; analogous to giving the volatility as a function of time.
    :param expiry: time till expiry (in years). Equivalently, time when the option expires, assuming the current
    time is 0.
    :param strike: Strike price for the option.
    :param initial_price: Current price of the underlying asset (stock, etc.)
    :param interest: Annual interest rate.
    :return: The number 'z' used in carr_pelts_price.
    """
    forward = forward_price(initial_price, interest, expiry)
    tau_T = tau(expiry)

    def to_solve(z):
        return h(z + tau_T) - h(z) - math.log(forward / strike)
    return root_scalar(to_solve, x0=0, x1=1).root


def carr_pelts_price(h, tau, expiry, strike, initial_price, interest):
    """
    The price of a call option according to the Carr-Pelts method.
    :param h: One of the Carr-Pelts parameters; a convex function of one variable.
    :param tau: One of the Carr-Pelts parameters; analogous to giving the volatility as a function of time.
    :param expiry: time till expiry (in years). Equivalently, time when the option expires, assuming the current
    time is 0.
    :param strike: Strike price for the option.
    :param initial_price: Current price of the underlying asset (stock, etc.)
    :param interest: Annual interest rate.
    :return: The Carr-Pelts option price.
    """
    forward = forward_price(initial_price, interest, expiry)
    discount = time_discount(interest, expiry)
    z = find_z(h, tau, expiry, strike, initial_price, interest)

    def omega(z):
        return quad(lambda x: math.exp(-1 * h(x)), -inf, z)[0]
    return discount * (forward * omega(z + tau(expiry)) - strike * omega(z))


def find_tau(expiries, volatilities):
    """
    Generate the 'tau' parameter function used in the Carr-Pelts method.
    :param expiries: An array of times, starting with 0, in increasing order (typically expiry times of some option).
    :param volatilities: An array of 'volatility-like quantities' corresponding to the expiries (typically implied
    volatilities of some option).
    :return: A function, tau, of a single real argument.
    """
    M = len(expiries) - 1
    precomputed_values = [0] * (M + 1)
    for i in range(M+1):
        for j in range(i):
            precomputed_values[i] += volatilities[j]**2 * (expiries[j + 1] - expiries[j])

    def new_tau(t):
        for i in range(M):
            if t < expiries[i]:
                return math.sqrt(precomputed_values[i] +
                                 volatilities[i] ** 2 * (t - expiries[i]))
        return math.sqrt(precomputed_values[M] +
                         volatilities[M] ** 2 * (t - expiries[M]))

    return new_tau


def first_tau(T):
    return 0.3 * T**(1/2)


def first_h(x):
    return (x**2 + math.log(2 * math.pi)) / 2


second_tau = find_tau([0, 1], [0.3, 0.3])
print(carr_pelts_price(first_h, first_tau, 0.5, 35, 32, 0.05))
print(carr_pelts_price(first_h, second_tau, 0.5, 35, 32, 0.05))