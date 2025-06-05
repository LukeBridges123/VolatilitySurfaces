from blackscholes import *
from carrpelts import *
import matplotlib.pyplot as plt
import numpy as np

strike = 25
initial_price = 32
interest = 0.05
option_price = 7.9
expirations = np.linspace(0.01, 0.5, 100)
volatilities = [black_scholes_implied_volatility(x, strike, initial_price, interest, option_price) for x in expirations]
plt.title("Black-Scholes Implied Volatility, Strike Price \${}, Current Stock Price \${}, Interest Rate {}, Option Price \${}".format(
            strike, initial_price, interest, option_price
))

plt.xlabel("Years to expiration")
plt.ylabel("Implied volatility")
plt.plot(expirations, volatilities)
plt.show()

nodes = [-1 * math.inf, -1, 0, 1, math.inf]
a = 1/2
b = 0
c_list = [1/2, 1, 1, 1/2]
expiries = [0.1, 0.25, 0.5]
prices = [1.94, 3.87, 6.5]
bs_volatilities = [black_scholes_implied_volatility(expiries[i], initial_price, initial_price, interest, prices[i])
                    for i in range(len(expiries))]

strike = 140
interest = 0.05
initial_price = 140
def bs_tau(T):
    return 0.3 * math.sqrt(T)
def bs_h(x):
    return (x**2 / 2) + 0.5 * math.log(2 * math.pi)
second_tau = find_tau(expiries, bs_volatilities)
third_h = find_h(nodes, a, b, c_list)
expirations = np.linspace(0.1, 0.5, 1000)
volatilities = [carr_pelts_volatility(bs_h, bs_tau, x, strike, initial_price, interest) for x in expirations]
plt.title("Carr-Pelts Implied Volatility, Strike Price \${}, Current Stock Price \${}, Interest Rate {}, Option Price \${}".format(
            strike, initial_price, interest, option_price
))
plt.xlabel("Years to expiration")
plt.ylabel("Implied volatility")
# plt.scatter(expiries, bs_volatilities)
plt.plot(expirations, volatilities)
plt.show()

nodes = [-1 * math.inf, -2, -1, 0, 1, 2, math.inf]
a = 1/2
b = 0
c_list = [1/2, 1/2, 1, 1, 1/2, 1/2]
third_h = find_h(nodes, a, b, c_list)
times = np.linspace(0.1, 0.5, 1000)
prices = [carr_pelts_price(third_h, second_tau, x, 35, 32, 0.05) for x in times]
plt.plot(times, prices)
plt.show()

