import numpy as np
from scipy.integrate import quad

class HestonCFModel:
    def __init__(self, S0, v0, theta, rho, kappa, xi, r):
        self.S0 = S0
        self.v0 = v0
        self.theta = theta
        self.rho = rho
        self.kappa = kappa
        self.xi = xi
        self.r = r


    def update(self, S0, v0, theta, rho, kappa, xi, r):
        self.S0 = S0
        self.v0 = v0
        self.theta = theta
        self.rho = rho
        self.kappa = kappa
        self.xi = xi
        self.r = r
        

    def characteristic_fn(self, u, T):
        """Characteristic function for the Heston model."""
        i = 1j
        x0 = np.log(self.S0)
        
        # compute d, g as intermediaries.
        d = np.sqrt(
            (self.rho * self.xi * i * u - self.kappa) ** 2
            + self.xi ** 2 * (i * u + u ** 2)
        )
        g = (self.kappa - self.rho * self.xi * i * u - d) / (
            self.kappa - self.rho * self.xi * i * u + d)
        
        exp_dT = np.exp(-d * T)
        
        # then solve for C, D and plug into the final solver
        C = (i * u * (x0 + self.r * T)) + (self.kappa * self.theta / self.xi ** 2) * (
                (self.kappa - self.rho * self.xi * i * u - d) * T
                - 2 * np.log((1-g * exp_dT) / (1 - g))
            )

        D = (self.kappa - self.rho * self.xi * i * u - d) / self.xi ** 2 * \
                (1 - exp_dT) / (1 - g * exp_dT)
                
        return np.exp(C + D * self.v0)
    
    
class CarrMadanSolver:
    def __init__(self, model: HestonCFModel,
                    alpha = 1.5,
                    N = 4096,
                    eta = 0.25):
        self.model = model
        self.alpha = alpha
        self.N = N
        self.eta = eta
        

    def solve(self, K, T):
        """Solve for a single call price via direct numerical integration of the Carr-Madan formula."""
        k = np.log(K)
        alpha = self.alpha

        def integrand(u):
            cf = self.model.characteristic_fn(u - (alpha + 1) * 1j, T)
            denom = alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u
            return np.real(np.exp(-1j * u * k) * cf / denom)

        result, _ = quad(integrand, 0, 200, limit=200)
        price = np.exp(-alpha * k) / np.pi * result
        discount = np.exp(-self.model.r * T)
        return discount * price
    
    
    def solve_fft(self, T):
        """Solve for one expiry slice using Carr-Madan FFT.

        Returns:
            Tuple of strike grid K and call prices C on that grid.
        """
        i = 1j

        v_grid = np.arange(self.N) * self.eta
        lambd = 2 * np.pi / (self.N * self.eta)
        
        b = 0.5 * lambd * self.N
        k_grid = np.arange(self.N) * lambd - b
        K_grid = np.exp(k_grid)

        u = v_grid - i * (self.alpha + 1)

        phi = self.model.characteristic_fn(u, T)

        discount = np.exp(-self.model.r * T)
        integrand = discount * phi / (
            self.alpha**2 + self.alpha - v_grid**2 + 
            1j * (2 * self.alpha + 1) * v_grid)

        weights = np.ones(self.N)
        weights[0] = 0.5

        modified_integrand = np.exp(i * b * v_grid) * integrand * weights * self.eta
        fft = np.fft.fft(modified_integrand)
        C_grid = np.exp(-self.alpha * k_grid) / np.pi * np.real(fft)
        return K_grid, C_grid
