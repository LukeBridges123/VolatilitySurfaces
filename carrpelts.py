from scipy.optimize import root_scalar
from scipy.integrate import quad
import numpy as np
from numpy import inf
import math
from blackscholes import black_scholes_implied_volatility


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
        return quad(lambda x: np.exp(-1 * h(x)), -inf, z)[0]
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
# N.B. We assume that nodes[0] = -inf and nodes[-1] = inf
def find_h(nodes, a, b, c_list):
    N = len(nodes) // 2
    a_list = [a] * (2 * N)
    b_list = [b] * (2 * N)

    for i in range(N, 2 * N - 1):
        a_list[i+1] = a_list[i] + b_list[i] * (nodes[i+1] - nodes[i]) + (1/(2*c_list[i])) * (nodes[i+1]-nodes[i])**2
        b_list[i+1] = b_list[i] + (1/c_list[i]) * (nodes[i+1] - nodes[i])
    for i in range(N-1, 0, -1):
        a_list[i-1] = a_list[i] + b_list[i] * (nodes[i] - nodes[i+1]) + (1/(2*c_list[i])) * (nodes[i]-nodes[i+1])**2
        b_list[i-1] = b_list[i] + (1/c_list[i]) * (nodes[i] - nodes[i+1])
    omega = 0
    for i in range(0, N):
        integral = quad(lambda x: np.exp(-1 * (a_list[i] + b_list[i] * (x - nodes[i+1]) + (1 / (2 * c_list[i])) * (x - nodes[i+1])**2)),
                        nodes[i], nodes[i+1])
        omega += integral[0]
    for i in range(N, 2*N):
        integral = quad(lambda x: np.exp(-1 * (a_list[i] + b_list[i] * (x - nodes[i]) + (1 / (2 * c_list[i])) * (x - nodes[i]) ** 2)),
                        nodes[i], nodes[i + 1])
        omega += integral[0]

    for i in range(2 * N):
        a_list[i] += math.log(omega)

    def h(x):
        for i in range(0, N):
            if x < nodes[i+1]:
                return a_list[i] + b_list[i] * (x - nodes[i+1]) + (1 / (2 * c_list[i])) * (x - nodes[i+1])**2
        for i in range(N, 2 * N):
            if x < nodes[i+1]:
                return a_list[i] + b_list[i] * (x - nodes[i]) + (1 / (2 * c_list[i])) * (x - nodes[i]) ** 2
        return a_list[2 * N - 1] + b_list[2 * N - 1] * (x - nodes[2 * N - 1]) + \
               (1 / (2 * c_list[2 * N - 1])) * (x - nodes[2 * N - 1]) ** 2
    return h

# use Black-Scholes implied volatilities as the "volatility like quantities"
def carr_pelts_model(initial_price, interest, expiries, prices, nodes, a, b, c_list):
    h = find_h(nodes, a, b, c_list)
    # get implied volatilities of at-the-money options with given maturities
    volatilities = [black_scholes_implied_volatility(expiries[i], initial_price, initial_price, interest, prices[i])
                    for i in range(len(expiries))]
    tau = find_tau(expiries, volatilities)

    def model(expiry, strike):
        return carr_pelts_price(h, tau, expiry, strike, initial_price, interest)
    return model

def first_tau(T):
    return 0.3 * T**(1/2)


def first_h(x):
    return (x**2 + math.log(2 * math.pi)) / 2

# version with bad numerical differentiation, just as a first attempt
def simple_deriv(f, x, dx = 0.001):
    return (f(x + dx) - f(x))/dx
def carr_pelts_volatility(h, tau, expiry, strike, initial_price, interest):
    z = find_z(h, tau, expiry, strike, initial_price, interest)
    tau_prime = simple_deriv(tau, expiry)
    h_prime_1 = simple_deriv(h, z + tau(expiry))
    h_prime_2 = simple_deriv(h, z)
    return math.sqrt(2 * tau_prime * (h_prime_1 - h_prime_2))




second_tau = find_tau([0, 1], [0.3, 0.3])
print(carr_pelts_price(first_h, first_tau, 0.5, 35, 32, 0.05))
# nodes etc. used to construct an h function mimicking the one in the Black-Scholes price
nodes_bs = [-1 * math.inf, 0, math.inf]
a_bs = 0
b_bs = 0
c_list_bs = [1, 1]
second_h = find_h(nodes_bs, a_bs, b_bs, c_list_bs)
print(carr_pelts_price(second_h, first_tau, 0.5, 35, 32, 0.05))
print(carr_pelts_volatility(second_h, first_tau, 0.5, 25, 32, 0.05))
# challenge problem 5
nodes = [-1 * math.inf, -1, 0, 1, math.inf]
a = 1/2
b = 0
c_list = [1/2, 1, 1, 1/2]
third_h = find_h(nodes, a, b, c_list)
print(carr_pelts_price(third_h, first_tau, 0.5, 35, 32, 0.05))

print(quad(lambda x : math.exp(-1 * third_h(x)), -1 * math.inf, math.inf)[0])
# challenge problem 3
test_model = carr_pelts_model(140, 0.05, [0.1, 0.25, 0.5], [1.94, 3.87, 6.5], nodes_bs, a_bs, b_bs, c_list_bs)
print(test_model(0.2, 160))
print(test_model(0.6, 130))

# challenge problem 6
test_model = carr_pelts_model(140, 0.05, [0.1, 0.25, 0.5], [1.94, 3.87, 6.5], nodes_bs, a, b, c_list)
print(test_model(0.2, 160))
print(test_model(0.6, 130))
