from scipy.optimize import root_scalar
from scipy.integrate import quad
from numpy import inf
import math


def forward_price(initial_price, interest, expiration):
    return initial_price * math.exp(interest * expiration)


def time_discount(interest, expiration):
    return math.exp(-1 * interest * expiration)


def find_z(h, tau, expiration, strike_price, initial_price, interest):
    forward = forward_price(initial_price, interest, expiration)
    tau_T = tau(expiration)

    def to_solve(z):
        return h(z + tau_T) - h(z) - math.log(forward/strike_price)
    return root_scalar(to_solve, x0=0, x1=1).root


def carr_pelts_price(h, tau, expiration, strike_price, initial_price, interest):
    forward = forward_price(initial_price, interest, expiration)
    discount = time_discount(interest, expiration)
    z = find_z(h, tau, expiration, strike_price, initial_price, interest)

    def omega(z):
        return quad(lambda x: math.exp(-1 * h(x)), -inf, z)[0]
    return discount * (forward * omega(z + tau(expiration)) - strike_price * omega(z))


def find_tau(expirations, volatilities):
    M = len(expirations) - 1
    precomputed_values = [0] * (M + 1)
    for i in range(M+1):
        for j in range(i):
            precomputed_values[i] += volatilities[j]**2 * (expirations[j+1] - expirations[j])

    def new_tau(t):
        for i in range(M):
            if t < expirations[i]:
                return math.sqrt(precomputed_values[i] +
                                 volatilities[i]**2 * (t - expirations[i]))
        return math.sqrt(precomputed_values[M] +
                         volatilities[M]**2 * (t - expirations[M]))
    return new_tau


def first_tau(T):
    return 0.3 * T**(1/2)


def first_h(x):
    return (x**2 + math.log(2 * math.pi)) / 2


second_tau = find_tau([0, 1], [0.3, 0.3])
print(carr_pelts_price(first_h, first_tau, 0.5, 35, 32, 0.05))
print(carr_pelts_price(first_h, second_tau, 0.5, 35, 32, 0.05))