def numerical_derivative(f, x, dx=1e-8):
    """Numerical derivative"""
    return (f(x + dx) - f(x)) / (dx)


def newton_raphson_solver(f, x0, df=None,
                            tolerance=1e-10,
                            derivative_tolerance=1e-12,
                            iterations=1000,
                            bounds=(1e-6, 5.0)):
    """
    Newton-Raphson solver on a function f. If no derivative df is given, computes a numerical derivative.
    Solver is clamped to stay within bounds.
    """
    x = x0
    lo, hi = bounds

    for _ in range(iterations):
        fx = f(x)
        if abs(fx) < tolerance:
            return x
        dfx = df(x) if df else numerical_derivative(f, x)
        if abs(dfx) < derivative_tolerance:
            raise ValueError("Derivative is effectively zero. No solution found.")
        else:
            x = x - fx / dfx
            if x <= lo:
                x = 0.5 * (x + fx / dfx + lo)  # Clamp to lower bound
            if x > hi:
                raise ValueError("Solution diverged beyond upper bound.")
    raise ValueError("Max iterations exceeded.")