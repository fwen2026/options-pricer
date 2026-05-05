def numerical_derivative(f, x, dx=1e-8):
    """Numerical derivative"""
    return (f(x + dx) - f(x)) / (dx)


def newton_raphson_solver(f, x0, df=None,
                            tolerance=1e-6,
                            derivative_tolerance=1e-12,
                            iterations=1000):
    """
    Newton-Raphson solver on a function f. If no derivative df is given, computes a numerical derivative.
    """
    x = x0
    fx = f(x)
    dfx = df(x) if df else numerical_derivative(f, x)

    if abs(dfx) < derivative_tolerance:
        raise ValueError("Derivative is effectively zero. No solution found.")
    for _ in range(iterations):
        if abs(fx) < tolerance:
            return x
        else:
            x = x - fx / dfx
            fx = f(x)
            dfx = df(x) if df else numerical_derivative(f, x)
    raise ValueError("Max iterations exceeded. No solution found.")
