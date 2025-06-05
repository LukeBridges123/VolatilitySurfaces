from blackscholes import *
from carrpelts import *
import math

print(black_scholes_implied_volatility(1/12, 190, 248.25, 0.05, 57.25))
#print(black_scholes_call_price(0.5, 35, 32, 0.05, 0.3))
print(black_scholes_implied_volatility(0.5, 25, 32, 0.05, 7.9))
# 0.2616
print(black_scholes_implied_volatility(7/365, 5075, 4884, 0.05, 34.60))


def first_tau(T):
    return 0.3 * T**(1/2)


def first_h(x):
    return (x**2 + math.log(2 * math.pi)) / 2

# to do next: plot with constant strike price, varying expiration



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
