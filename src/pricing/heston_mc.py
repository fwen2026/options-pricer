import numpy as np
from src.util.synthetic_iv_config import config_value, resolve_config

class HestonMCSim:
    def __init__(self, S0=None, v0=None,
                    theta=None, rho=None, kappa=None, xi=None, r=None,
                    dt=None,
                    cfg=None):
        if cfg is not None or dt is None or any(x is None for x in (S0, v0, theta, rho, kappa, xi, r)):
            params = self._params_from_cfg(resolve_config(cfg))
            S0 = params["S0"] if S0 is None else S0
            v0 = params["v0"] if v0 is None else v0
            theta = params["theta"] if theta is None else theta
            rho = params["rho"] if rho is None else rho
            kappa = params["kappa"] if kappa is None else kappa
            xi = params["xi"] if xi is None else xi
            r = params["r"] if r is None else r
            dt = params["dt"] if dt is None else dt

        self.S0 = S0
        self.v0 = v0
        self.theta = theta
        self.rho = rho
        self.kappa = kappa
        self.xi = xi
        self.r = r
        self.dt = dt


    @classmethod
    def from_cfg(cls, cfg=None, **overrides):
        params = cls._params_from_cfg(resolve_config(cfg))
        params.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**params)


    @staticmethod
    def _params_from_cfg(cfg):
        cfg = resolve_config(cfg)
        heston_cfg = cfg["heston"]
        return {
            "S0": config_value(heston_cfg["s0"]),
            "v0": config_value(heston_cfg["v0"]),
            "theta": config_value(heston_cfg["theta"]),
            "rho": config_value(heston_cfg["rho"]),
            "kappa": config_value(heston_cfg["kappa"]),
            "xi": config_value(heston_cfg["xi"]),
            "r": config_value(heston_cfg["r"]),
            "dt": config_value(cfg.get("mc", {}).get("dt", 1e-3)),
        }
        
        
    def get_initial_conditions(self):
        """Gets the initial conditions for the model"""
        return self.v0, self.S0


    def update(self, S0=None, v0=None, theta=None, rho=None, kappa=None, xi=None, r=None, cfg=None):
        """Updates simulation params"""
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


    def get_random_variables(self, paths, rng):
        """Gets the delta of each Wiener process for our model"""
        Z1 = rng.standard_normal(paths)
        Z2 = rng.standard_normal(paths)
        
        dZ1 = Z1
        dZ2 = self.rho * dZ1 + np.sqrt(1 - self.rho**2) * Z2
        return dZ1, dZ2
    
    
    def get_dv(self, v, dZ2):
        """Try Truncated Euler on vol first. If it doesn't work, revert to Andersen QE"""
        dv = self.dt * (self.kappa * (self.theta - v)) + self.xi * np.sqrt(v * self.dt) * dZ2
        return dv
    
    
    def get_dS(self, S, dZ1, v):
        """Truncated Euler method."""
        dS = self.r * S * self.dt + S * np.sqrt(v * self.dt) * dZ1
        return dS
        
    
    def generate_prices(self,
                        seed=None,
                        steps=None,
                        paths=None,
                        cfg=None
        ):
            """Monte-Carlo simulation, 100000 paths as default."""
            cfg = resolve_config(cfg)
            seed = cfg["seed"] if seed is None else seed
            steps = cfg.get("mc", {}).get("steps", 1000) if steps is None else steps
            paths = cfg["dataset"]["num_samples"] if paths is None else paths
            rng = np.random.default_rng(seed=seed)

            S = np.empty((steps+1, paths))
            v = np.empty((steps+1, paths))

            S[0] = self.S0
            v[0] = self.v0

            for t in range(steps):
                dZ1, dZ2 = self.get_random_variables(paths, rng)

                v_pos = np.maximum(v[t], 0)

                S[t+1] = S[t] + self.get_dS(S[t], dZ1, v_pos)
                v[t+1] = v[t] + self.get_dv(v_pos, dZ2)

                v[t+1] = np.maximum(v[t+1], 0)

            return S, v

