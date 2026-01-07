import numpy as np
from scipy.optimize import shgo


b1, c1 = 0.34, 0.16
b2, c2 = 0.465, 0.035

THRESHOLD = 1e-15

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

# Upper bound on the value xlog(x) when x is approaching 0
def xlogx_upper(x):
    if x < THRESHOLD:
      return 0
    else:
      return x * np.log2(x)

def get_asymptotic_rate(p, r, b, c):
    q = 1.0 - p - r

    # Calculate v

    beta = 4*c + 2*b - p - 3*q
    # Discriminant: beta^2 + 8rq.
    discriminant = beta**2 + 8 * r * q
    # Since r >= constant, division is safe.
    v = (-beta + np.sqrt(discriminant)) / (4*r)


    # Line 1 calculations
    # Since -xlogx contributes positively, we use lower bound on it
    line1 = -xlogx_lower(p) - xlogx_lower(q) - xlogx_lower(r) + q


    # Line 2 calculations
    # Since log(v) contributes positively, we use upper bound on it
    line2 = q * logx_upper(v) - 2*c * np.log2(2*v + 1) - (2*b - p) * np.log2(v + 1)

    # Line 3 calculations
    # Since xlogx contributes positively in both places, we use upper bound on it
    line3 = 0.5 * xlogx_upper(2*b - p) + 0.5 * xlogx_upper(p) - b * np.log2(2*b)

    return line1 + line2 + line3

def objective(x):

    p, r = x

    val1 = get_asymptotic_rate(p, r, b1, c1)
    val2 = get_asymptotic_rate(p, r, b2, c2)
    return -min(val1, val2) # since this is a minimizing solver, we have to negate the cost

cons = (
    {'type': 'ineq', 'fun': lambda x: x[0]},          # Adding constraint p >= 0
    {'type': 'ineq', 'fun': lambda x: x[1]},          # Adding constraint r >= 0
    {'type': 'ineq', 'fun': lambda x: 0.64 - x[0]}, # Adding constraint p <= 0.64 n
    {'type': 'ineq', 'fun': lambda x: (1.0*25/32)*x[0] + 6.25*x[1] - 1.25}, # Adding constraint (25/32)p + (25/4)r - (5/4) >= 0
    {'type': 'ineq', 'fun': lambda x: 1.0 - x[0] - x[1]} # Adding constraint 1 - p - r >= 0
)

# Bounds for p and r
bounds = [(0.0, 0.64), (0.0, 1.0)]

print("Running shgo optimizer...")
res = shgo(objective, bounds,
            constraints=cons,
            options={'f_tol': 1e-12},
            n = 100,
            minimizer_kwargs={'method':'COBYLA', 'tol': 1e-12})

print(res.message)
print("Was the optimization successful? {}".format(res.success))
max_val = -res.fun # Res is negative of the actual cost since we run a minimizer. So we negate again.

print("-" * 30)
print(f"Max Value:      {max_val:.6f}")
res
