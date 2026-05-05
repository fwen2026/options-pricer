def bisection(fx, x1, x2, 
                tolerance=1e-5, 
                max_iterations=100):
    """
    Bisection method for finding roots of a function.
    """
    for i in range(max_iterations):
        x_mid = (x1 + x2) / 2
        if fx(x_mid) == 0 or (x2 - x1) / 2 < tolerance:
            return x_mid
        if fx(x_mid) * fx(x1) < 0:
            x2 = x_mid
        else:
            x1 = x_mid
    return (x1 + x2) / 2