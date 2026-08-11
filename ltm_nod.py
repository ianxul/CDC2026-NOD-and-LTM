"""Linear Threshold Model and its Nonlinear Opinion Dynamics relaxation.

Reference implementation accompanying

    I. X. Belaustegui, H. Sinhmar, L.-W. Kong, A. M. Hein and N. E. Leonard,
    "A Continuous Time and State-Space Relaxation of the Linear Threshold
    Model with Nonlinear Opinion Dynamics."

Equation and result numbers in the docstrings refer to that paper.

Conventions
-----------
``A`` is a non-negative weighted adjacency matrix with zero diagonal, where
``A[i, j]`` is the weight with which node ``j`` influences node ``i``.  Cascades
are reported as activation-time arrays of length ``n``: entry ``i`` is the time
at which node ``i`` became active, or ``np.nan`` if it never did.
"""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

DATA_DIR = Path(__file__).parent / "data"


# --------------------------------------------------------------------------
# LTM instances
# --------------------------------------------------------------------------

def load_instance():
    """Return the ``(A, tau)`` LTM instance used for the figures in the paper.

    It is a 30-node fractional contagion model: ``tau = 0.6 * (A @ ones)``.
    """
    A = np.loadtxt(DATA_DIR / "A.csv", delimiter=",")
    tau = np.loadtxt(DATA_DIR / "tau.csv", delimiter=",")
    return A, tau


def fcm_thresholds(A, fraction):
    """Thresholds of the fractional contagion model, ``tau_i = fraction * (A 1)_i``.

    A node activates once the given `fraction` of the social input it can
    receive is active.
    """
    return fraction * A.sum(axis=1)


def random_instance(n=30, density=0.08, fraction=0.6, seed=0):
    """A random directed LTM instance, for exploring variants of the figures.

    Edge weights are drawn from ``{0.5, 0.6, ..., 1.1}`` and thresholds follow
    the fractional contagion model, matching the instance shipped in ``data/``.
    """
    rng = np.random.default_rng(seed)
    A = rng.choice(np.arange(0.5, 1.2, 0.1), size=(n, n))
    A *= rng.random((n, n)) < density
    np.fill_diagonal(A, 0.0)
    return A, fcm_thresholds(A, fraction)


# --------------------------------------------------------------------------
# Linear Threshold Model
# --------------------------------------------------------------------------

def simulate_ltm(A, tau, seeds):
    """Cascade of the LTM, Eq. (1).

    Every node is updated from the state at the previous step, exactly as
    written in Eq. (1).

    Returns
    -------
    ndarray of shape (n,)
        Activation times, ``np.nan`` for nodes that never activate.
    """
    n = len(tau)
    seeds = np.asarray(list(seeds), dtype=int)

    zeta = np.zeros(n)
    zeta[seeds] = 1.0
    times = np.full(n, np.nan)
    times[seeds] = 0.0

    # Monotone dynamics on a finite state space settle in at most n steps.
    for step in range(1, n + 1):
        newly = (zeta == 0) & (A @ zeta > tau)
        if not newly.any():
            break
        zeta[newly] = 1.0
        times[newly] = step

    return times


# --------------------------------------------------------------------------
# Nonlinear Opinion Dynamics
# --------------------------------------------------------------------------

def gamma_max(tau, k):
    """Largest social coupling admitting a valid relaxation (Definition 3).

    ``mu_i > 0`` for every node requires ``gamma * k * max(tau) < 1/4``.
    """
    return 1.0 / (4.0 * k * np.max(tau))


def nod_relaxation(tau, k, gamma):
    """Linear self-reinforcement gains of the NOD relaxation, Eq. (12).

    ``mu_i = 1 - 2 sqrt(gamma k tau_i)``, so that the LTM threshold ``tau_i``
    becomes the saddle-node bifurcation of node ``i``.
    """
    if gamma >= gamma_max(tau, k):
        raise ValueError(
            f"gamma={gamma:.4g} is not a valid relaxation of this instance; "
            f"it must be below {gamma_max(tau, k):.4g} so that every mu_i > 0."
        )
    return 1.0 - 2.0 * np.sqrt(gamma * k * tau)


def b_star(mu, k):
    """Per-node activation threshold on the external input (Proposition 3)."""
    return (1.0 - mu) ** 2 / (4.0 * k)


def nod_rhs(t, z, mu, k, gamma, A, b):
    """Right-hand side of the NOD dynamics, Eq. (5)."""
    return -z + np.clip(mu * z + k * z * z + gamma * (A @ z) + b(t), 0.0, 1.0)


def no_input(t):
    """The zero external input, ``b(t) = 0``."""
    return 0.0


def simulate_nod(A, mu, k, gamma, seeds=(), b=no_input, t_span=(0.0, 1000.0),
                 method="RK45", **kwargs):
    """Integrate Eq. (5) from ``z_i(0) = 1`` on `seeds` and 0 elsewhere."""
    z0 = np.zeros(A.shape[0])
    z0[list(seeds)] = 1.0
    return solve_ivp(nod_rhs, t_span, z0, args=(mu, k, gamma, A, b),
                     method=method, dense_output=True, **kwargs)


def activation_times(sol, seeds=(), level=0.95):
    """First time each opinion in `sol` rises above `level`, Definition 4.

    Seeds start active and are reported at time 0.  Resolution is that of the
    solver's own time grid, which is fine here because activation is fast
    compared to the spacing between activations.
    """
    times = np.full(sol.y.shape[0], np.nan)
    for i, z_i in enumerate(sol.y):
        crossings = np.flatnonzero(z_i > level)
        if crossings.size:
            times[i] = sol.t[crossings[0]]
    times[list(seeds)] = 0.0
    return times
