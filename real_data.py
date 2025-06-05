from data.search import search_options
# underlying price = 2368.06
from blackscholes import black_scholes_implied_volatility
from carrpelts import carr_pelts_model
import numpy as np
import matplotlib.pyplot as plt
import math
strike = 2365
rate = 0.0289

data = search_options(strike_price=strike)[["date", "expiration", "strike", "ask", "volume"]]
data = data.reset_index()
print(data)
# indexes 0, 3, 6
times = [1/365, 8/365, 14/365, 22/365, 29/365]
option_prices = [6.8, 13.1, 16.8, 21.5, 26.5]
times2 = [4/365, 6/365, 11/365, 18/365, 25/365]
prices2 = [8.6, 11.1, 14.2, 18, 24.4]

#c_list = [1, 2, 1, 1, 2, 1]
#c_list = [4, 1, 1, 1, 1, 4]
#c_list = [1, 1, 4, 4, 1, 1]
#c_list = [1/2] * 6
#c_list = [1, 1, 1/2, 1/2, 1, 1]
c_list = [1] * 6
f = carr_pelts_model(2368.06, 2365, rate, times, option_prices, [-1 * math.inf, -2, -1, 0, 1, 2, math.inf], 0.5 * math.log(2 * math.pi), 1, c_list)

xs = np.linspace(2/365, 31/365, 60)
ys = np.asarray([f(t, 2365) for t in xs])
plt.scatter(times, option_prices, label='\"Training\" points')
plt.scatter(times2, prices2, c='r', label='\"Test\" points')
plt.plot(xs, ys)
plt.xlabel("Time until expiration (fractional years)")
plt.ylabel("At-the-money call option price (USD)")
plt.title("Quadratic coefficients = {}".format(c_list))
plt.legend()
plt.show()


