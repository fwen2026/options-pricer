import numpy as np


def numerical_derivative(f, x, dx=1e-8):
    """Numerical derivative. Thank God it's vectorized"""
    return (f(x + dx) - f(x)) / (dx)


def vectorized_newton_raphson(f, x0, df=None,
                            tolerance=1e-6,
                            derivative_tolerance=1e-8,
                            iterations=50,
                            bounds=(1e-6, 5.0),
                            numerical_dx=1e-8):
    """
    Vectorized Newton-Raphson solver a function f. Notes: f and df should be vectorized in and of themselves.
    Also returns a mask of elements that did converge. 
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    converged = np.zeros_like(x, dtype=bool)
    lo, hi = bounds

    for _ in range(iterations):
        fx = f(x)
        dfx = df(x) if df else numerical_derivative(f, x, dx=numerical_dx)
        zero_derivative = np.abs(dfx) < derivative_tolerance
        out_of_bounds = (x <= lo) | (x > hi)
        frozen = out_of_bounds | converged | zero_derivative

        safe_dfx = np.where(zero_derivative, 0.67, dfx)  # the 0.67 is just purely arbitrary to prevent a div by zero. also SIX SEVENNN
        step = np.where(frozen, 0.0, fx / safe_dfx)
        x = x - step
        converged |= np.abs(fx) < tolerance

        if np.all(converged):
            break
    return x, converged


def newton_raphson_solver(f, x0, df=None,
                            tolerance=1e-6,
                            derivative_tolerance=1e-8,
                            iterations=50,
                            bounds=(1e-6, 5.0),
                            numerical_dx=1e-8,
                            clamp_to_bounds=True):
    """
    Custom scalar Newton-Raphson solver on a function f. If no derivative df is given, computes a numerical derivative.
    Solver is clamped to stay within bounds (idt scipy's solver does this?).
    """
    x = x0
    lo, hi = bounds

    for _ in range(iterations):
        fx = f(x)
        if abs(fx) < tolerance:
            return x
        dfx = df(x) if df else numerical_derivative(f, x, dx=numerical_dx)
        if abs(dfx) < derivative_tolerance:
            raise ValueError("Derivative is effectively zero. No solution found.")
        else:
            x = x - fx / dfx
            if clamp_to_bounds:
                if x <= lo:
                    x = 0.5 * (x + fx / dfx + lo)
                if x > hi:
                    raise ValueError("Solution diverged beyond upper bound.")
            elif x <= lo or x > hi:
                raise ValueError("Solution diverged beyond bounds.")
    raise ValueError("Max iterations exceeded.")
