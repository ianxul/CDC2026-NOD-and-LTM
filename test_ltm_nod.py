"""Numerical checks of the two main results, on the instance in ``data/``.

Run with ``python test_ltm_nod.py`` (no test runner needed).
"""

import itertools

import numpy as np

import fig_distributed_input as f3
import ltm_nod as ln

K = 1.1
SEED_SETS = [(0,), (0, 1, 2, 3), (14,), (18, 19), (5, 6, 7)]

ACTIVE = 0.95            # z above this counts as an activation, Definition 4
STAGGER_TRIALS = 100     # draws of the Fig. 3c onset delays


def cascade_ltm(A, tau, seeds):
    return frozenset(np.flatnonzero(~np.isnan(ln.simulate_ltm(A, tau, seeds))))


def cascade_nod(A, mu, gamma, seeds, t_max=4000.0):
    sol = ln.simulate_nod(A, mu, K, gamma, seeds, t_span=(0.0, t_max))
    return frozenset(np.flatnonzero(sol.y[:, -1] > 0.95))


def run_fig3_case(A, mu, gamma, onsets):
    """One Fig. 3 run.  Returns ``(nodes that activate, peak opinion)``.

    Activation is judged over the whole trajectory, not just its end, so a node
    that rises above the line and falls back still counts.
    """
    sol = ln.simulate_nod(A, mu, f3.K, gamma, b=f3.step_input(onsets),
                          t_span=(0.0, f3.T_MAX), method="Radau",
                          t_eval=np.linspace(0.0, f3.T_MAX, 4001))
    peaks = sol.y.max(axis=1)
    return np.count_nonzero(peaks > ACTIVE), peaks.max()


def fig3_stagger_trials(A, n_trials=STAGGER_TRIALS):
    """Fig. 3c under `n_trials` independent draws of the onset delays.

    Fig. 3 makes its point with one draw: subthreshold input that cascades the
    group when it arrives at once (case b) does nothing when the same input is
    spread over ``MAX_DELAY`` (case c).  A single draw cannot say whether that
    is a property of the staggering or a lucky sample, so this repeats case c
    over many draws.  Trial 0 is the one the figure shows.
    """
    n = A.shape[0]
    tau = np.full(n, f3.THRESHOLD)               # Fig. 3 gives every node the same tau
    gamma = f3.GAMMA_FRACTION * ln.gamma_max(tau, f3.K)
    mu = ln.nod_relaxation(tau, f3.K, gamma)

    # Guard: the simultaneous case must still cascade under these parameters.
    # Without it, a mistake that silences the dynamics would make case c look
    # like a result rather than a broken setup.
    simultaneous, _ = run_fig3_case(A, mu, gamma, np.full(n, f3.ONSET))
    assert simultaneous > 0, "case b should cascade; the parameters are wrong"

    trials_with_activation, total_activations, peak = 0, 0, 0.0
    for trial in range(n_trials):
        delays = np.random.default_rng(trial).uniform(0.0, f3.MAX_DELAY, n)
        active, trial_peak = run_fig3_case(A, mu, gamma, f3.ONSET + delays)
        trials_with_activation += active > 0
        total_activations += active
        peak = max(peak, trial_peak)

    print(f"Fig. 3c: {trials_with_activation}/{n_trials} staggered trials show "
          f"any activation ({total_activations} nodes in total); the highest "
          f"opinion reached in any trial was z = {peak:.3f}, against the "
          f"activation line at {ACTIVE}")
    print(f"         for reference, case b activates {simultaneous}/{n} nodes "
          f"from the same input delivered at once")
    return trials_with_activation, total_activations


def main():
    A, tau = ln.load_instance()
    g_max = ln.gamma_max(tau, K)

    # The instance is the fractional contagion model with tau = 0.6 (A 1).
    assert np.allclose(tau, ln.fcm_thresholds(A, 0.6))

    # Theorem 1: C_LTM is contained in C_NOD for every relaxation and seed set.
    for fraction, seeds in itertools.product((0.05, 0.1, 0.4, 0.8), SEED_SETS):
        gamma = fraction * g_max
        mu = ln.nod_relaxation(tau, K, gamma)
        assert cascade_ltm(A, tau, seeds) <= cascade_nod(A, mu, gamma, seeds), \
            (fraction, seeds)
    print("Theorem 1: C_LTM is a subset of C_NOD in all cases checked")

    # Theorem 2: small enough gamma recovers the LTM cascade exactly.
    gamma = 0.05 * g_max
    mu = ln.nod_relaxation(tau, K, gamma)
    for seeds in SEED_SETS:
        assert cascade_ltm(A, tau, seeds) == cascade_nod(A, mu, gamma, seeds), seeds
    print(f"Theorem 2: gamma = {gamma:.4g} recovers C_LTM exactly")

    # Fig. 3c is not a lucky draw: staggering the same input never cascades.
    trials_with_activation, total_activations = fig3_stagger_trials(A)
    assert trials_with_activation == 0 and total_activations == 0

    # The relaxation is only defined below gamma_max.
    try:
        ln.nod_relaxation(tau, K, g_max)
    except ValueError:
        print(f"gamma_max = {g_max:.4g} is rejected as invalid, as it should be")
    else:
        raise AssertionError("gamma_max should not be a valid relaxation")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
