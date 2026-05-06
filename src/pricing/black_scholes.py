import numpy as np
from scipy.stats import norm

# Black-Scholes option pricing model. also used to generate data for a model. 
# Inputs can either be a dataset as a NumPy array or just floats.
class BlackScholesOption:
    def __init__(self, S, K, T, r, sigma=None):
        self.S = S  # underlying asset price
        self.K = K  # strike price
        self.T = T  # time to expiration
        self.r = r  # risk-free interest rate
        self.sigma = sigma  # volatility

    def update(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def d1(self):
        return (self.moneyness()
                    + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return (self.d1()
                - self.sigma * np.sqrt(self.T))

    # returns a call price
    def call_price(self):
        return (self.S * norm.cdf(self.d1()) 
                - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2()))

    # returns a put price
    def put_price(self):
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) 
                - self.S * norm.cdf(-self.d1()))
    
    # moneyness
    def moneyness(self):
        return np.log(self.S / self.K)

    # a couple of greeks
    def delta_call(self):
        return norm.cdf(self.d1())

    def delta_put(self):
        return norm.cdf(self.d1()) - 1

    def gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S * norm.pdf(self.d1()) * np.sqrt(self.T)