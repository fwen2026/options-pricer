import numpy as np

class HestonModel:
    def __init__(self, S0, v0, 
                    theta, rho, kappa, xi, r,
                    dt=1e-3):
        self.S0 = S0
        self.v0 = v0
        self.theta = theta
        self.rho = rho
        self.kappa = kappa
        self.xi = xi
        self.r = r
        self.dt = dt
        
        
    def get_initial_conditions(self):
        """Gets the initial conditions for the model"""
        return self.v0, self.S0
    
    
    def update(self, nu_0, theta, rho, kappa, xi):
        """Updates simulation params"""
        self.nu_0 = nu_0
        self.theta = theta # long-run variance
        self.rho = rho 
        self.kappa = kappa # mean reversion
        self.xi = xi # vol of vol


    def get_random_variables(self, paths, seed):
        """Gets the delta of each Wiener process for our model"""
        rng = np.random.default_rng(seed=seed)
        Z1 = rng.standard_normal(paths)
        Z2 = rng.standard_normal(paths)
        
        dZ1 = Z1
        dZ2 = self.rho * dZ1 + np.sqrt(1 - self.rho**2) * Z2
        return dZ1, dZ2
    
    
    def get_dv(self, v, dZ2):
        """Try Naive Euler on vol first. If it doesn't work, revert to Andersen QE"""
        dv = self.dt * (self.kappa * (self.theta - v)) + self.xi * np.sqrt(v * self.dt) * dZ2
        return dv
    
    
    def get_dS(self, S, dZ1, v):
        """Truncated Euler method."""
        dS = self.r * S * self.dt + S * np.sqrt(v * self.dt) * dZ1
        
    
    def generate_prices(self,
                        seed=42,
                        steps=1000,
                        paths=100_000
        ):
            """Monte-Carlo simulation, 100000 paths as default."""
            S = np.empty((steps+1, paths))
            v = np.empty((steps+1, paths))
            
            S[0] = self.S0
            v[0] = self.v0
            
            for t in range(steps):
                dZ1, dZ2 = self.get_random_variables(paths, seed)
                v_pos = np.maximum(v[t], 0)
                
                S[t+1] = S[t] + self.get_dS(S[t], dZ1, v_pos)
                v[t+1] = v[t] + self.get_dv(v_pos, dZ2)

            return S, v
        

