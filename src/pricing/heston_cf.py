import numpy as np
from scipy.integrate import quad
from src.util.synthetic_iv_config import config_value, resolve_config

class HestonCFModel:
    def __init__(self, S0=None, v0=None, theta=None, rho=None, kappa=None, xi=None, r=None, cfg=None, rng=None):
        if cfg is not None or any(x is None for x in (S0, v0, theta, rho, kappa, xi, r)):
            params = self._params_from_cfg(resolve_config(cfg), rng=rng)
            S0 = params["S0"] if S0 is None else S0
            v0 = params["v0"] if v0 is None else v0
            theta = params["theta"] if theta is None else theta
            rho = params["rho"] if rho is None else rho
            kappa = params["kappa"] if kappa is None else kappa
            xi = params["xi"] if xi is None else xi
            r = params["r"] if r is None else r

        self.S0 = S0
        self.v0 = v0
        self.theta = theta
        self.rho = rho
        self.kappa = kappa
        self.xi = xi
        self.r = r


    @classmethod
    def from_cfg(cls, cfg=None, rng=None, **overrides):
        params = cls._params_from_cfg(resolve_config(cfg), rng=rng)
        params.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**params)


    @staticmethod
    def _params_from_cfg(cfg, rng=None):
        cfg = resolve_config(cfg)
        heston_cfg = cfg["heston"]
        return {
            "S0": config_value(heston_cfg["s0"], rng=rng),
            "v0": config_value(heston_cfg["v0"], rng=rng),
            "theta": config_value(heston_cfg["theta"], rng=rng),
            "rho": config_value(heston_cfg["rho"], rng=rng),
            "kappa": config_value(heston_cfg["kappa"], rng=rng),
            "xi": config_value(heston_cfg["xi"], rng=rng),
            "r": config_value(heston_cfg["r"], rng=rng),
        }


    def update(self, S0=None, v0=None, theta=None, rho=None, kappa=None, xi=None, r=None, cfg=None):
        if cfg is not None:
            params = self._params_from_cfg(cfg)
            S0 = params["S0"] if S0 is None else S0
            v0 = params["v0"] if v0 is None else v0
            theta = params["theta"] if theta is None else theta
            rho = params["rho"] if rho is None else rho
            kappa = params["kappa"] if kappa is None else kappa
            xi = params["xi"] if xi is None else xi
            r = params["r"] if r is None else r

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
    def __init__(self, model: HestonCFModel = None,
                    alpha=None,
                    N=None,
                    eta=None,
                    cfg=None):
        cfg = resolve_config(cfg)
        fft_cfg = cfg["fft"]
        alpha = fft_cfg["alpha"] if alpha is None else alpha
        N = fft_cfg["N"] if N is None else N
        eta = fft_cfg["eta"] if eta is None else eta

        self.model = model if model is not None else HestonCFModel.from_cfg(cfg)
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
        
        b = 0.5 * lambd * self.N - np.log(self.model.S0)
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
