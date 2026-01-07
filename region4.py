import numpy as np
from scipy.optimize import shgo
from numpy.polynomial import Polynomial as P

a1, c1 = 0.34, 0.16
a2, c2 = 0.355, 0.145

THRESHOLD = 1e-15
TOLERANCE = 1e-20
R_LOWER = 1e-5


# Upper bound on the value log(x) when x is approaching 0
def logx_upper(x):
    if x < THRESHOLD:
      return np.log2(THRESHOLD)
    else:
      return np.log2(x)

# Lower bound on the value xlog(x) when x is approaching 0
def xlogx_lower(x):
    if x < THRESHOLD:
        return THRESHOLD * np.log2(THRESHOLD)
    return x * np.log2(x)


def solve_cubic_v(p, q, r, a, c):
    """
    Solves for the unique positive root vi of the cubic equation:
    4r v^3 + 2(2a - p - q) v^2 + 2(2c - q)v - q = 0
    """

    coeff = [
        -q,
        2 * (2*c - q),
        2 * (2*a - p - q),
        4 * r,
    ]

    poly = P(coeff)
    roots = poly.roots()
    real_roots = roots[np.isclose(roots.imag, 0, atol=TOLERANCE)].real
    positive_roots = real_roots[real_roots > 0]

    # The degree of poly is guaranteed to be 3 since r >= R_LOWER
    # So a real root exists and by direct arguments, it must be positive
    return positive_roots[0]


def get_asymptotic_rate(p, q, a, c):
    r = 1.0 - p - q

    # We have analytically handled this case already
    if r <= R_LOWER:
      return -1e6

    # Calculating u_i and v_i
    vi = solve_cubic_v(p, q, r, a, c)
    ui = (p * (2 * vi**2 + 1)) / (2*a - p)

    # Line 1 calculations
    # Since -xlogx contributes positively, we use lower bound on it
    line1 = -xlogx_lower(p) - xlogx_lower(q) - xlogx_lower(r) + q

    # Line 2 calculations
    line2 = -2 * c * np.log2(2 * vi + 1) - a * np.log2(ui + 2 * vi**2 + 1)

    # Line 3 calculations
    line3 = (p / 2.0) * logx_upper(ui) + q * logx_upper(vi)

    return line1 + line2 + line3

def objective(x):

    p, q = x

    val1 = get_asymptotic_rate(p, q, a1, c1)
    val2 = get_asymptotic_rate(p, q, a2, c2)

    return -min(val1, val2)


cons = [
    {'type': 'ineq', 'fun': lambda x: x[0]}, # p >= 0
    {'type': 'ineq', 'fun': lambda x: x[1]}, # q >= 0
    {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]}, # 1 - p - q >= 0
    {'type': 'ineq', 'fun': lambda x: 0.64 - x[0]}, # 0.64 - p >= 0
    {'type': 'ineq', 'fun': lambda x: 0.75*x[0] + x[1] - 0.8} # (3/4)p + q - (4/5) >= 0
]

# Bounds for p and q
bounds = [(0, 0.64), (0, 1.0)]

print("Running shgo optimizer...")
res = shgo(
          objective,
          bounds=bounds,
          constraints=cons,
          options={'f_tol': 1e-12},
          minimizer_kwargs={'method': 'COBYLA', 'tol': 1e-12})

print(res.message)
print("Was the optimization successful? {}".format(res.success))
max_val = -res.fun # Res is negative of the actual cost since we run a minimizer. So we negate again.

print("-" * 30)
print(f"Max Value:      {max_val:.6f}")
res
