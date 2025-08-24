import math
import time
import numpy as np
from GUI_simulation_6DOF import SixDOFSimulator  # ← UPDATE THIS import


RNG = np.random.default_rng(42)

def random_state(sim: SixDOFSimulator) -> np.ndarray:
    """
    Create a plausible state vector around the initial trim,
    with small perturbations so dynamics stay well-behaved.
    x: [u,v,w,p,q,r, phi,theta,psi, N,E,D]
    """
    x0 = sim.x[:, -1].copy()
    # velocities +/- 10 m/s, rates +/- 5 deg/s, angles +/- 5 deg, position +/- 100 m
    x0[0] += float(RNG.normal(0.0, 5.0))   # u
    x0[1] += float(RNG.normal(0.0, 2.0))   # v
    x0[2] += float(RNG.normal(0.0, 5.0))   # w
    x0[3] += math.radians(RNG.normal(0.0, 1.0))  # p
    x0[4] += math.radians(RNG.normal(0.0, 1.0))  # q
    x0[5] += math.radians(RNG.normal(0.0, 1.0))  # r
    x0[6] += math.radians(RNG.normal(0.0, 3.0))  # phi
    x0[7] += math.radians(RNG.normal(0.0, 3.0))  # theta
    x0[8] += math.radians(RNG.normal(0.0, 3.0))  # psi
    x0[9] += float(RNG.normal(0.0, 50.0))        # N
    x0[10] += float(RNG.normal(0.0, 50.0))       # E
    x0[11] += float(RNG.normal(0.0, 20.0))       # D
    return x0


def set_random_controls(sim: SixDOFSimulator) -> None:
    # δa, δe, δr within small deflection; throttle 30–70%
    da = math.radians(float(RNG.uniform(-3.0, 3.0)))
    de = math.radians(float(RNG.uniform(-3.0, 3.0)))
    dr = math.radians(float(RNG.uniform(-2.0, 2.0)))
    thr = float(RNG.uniform(0.3, 0.7))
    sim.set_controls(da, de, dr, thr)


# ----------------------- tests: sanity -----------------------

def test_forces_moments_sanity(sim: SixDOFSimulator):
    set_random_controls(sim)
    x = random_state(sim)
    Vb = x[0:3]
    p, q, r = x[3:6]
    X, Y, Z, L, M, Nn = sim._forces_moments(Vb, p, q, r)

    arr = np.array([X, Y, Z, L, M, Nn], dtype=float)
    assert arr.shape == (6,)
    assert np.all(np.isfinite(arr)), "Forces/moments contain NaNs/inf"
    # Basic magnitudes (very loose): forces within ±5 MN, moments within ±5 GN·m
    assert np.all(np.abs(arr[:3]) < 5e6)
    assert np.all(np.abs(arr[3:]) < 5e9)


def test_rhs_sanity(sim: SixDOFSimulator):
    set_random_controls(sim)
    x = random_state(sim)
    dx = sim._rhs(x, t=0.0)
    assert dx.shape == (12,)
    assert np.all(np.isfinite(dx)), "State derivatives contain NaNs/inf"


def test_step_sanity_progress(sim: SixDOFSimulator):
    # Ensure step advances state and time, and remains finite
    set_random_controls(sim)
    t0 = sim.t_hist[-1]
    x0 = sim.x[:, -1].copy()
    sim.step()
    assert sim.t_hist[-1] > t0
    x1 = sim.x[:, -1]
    assert np.all(np.isfinite(x1))
    # Should not be identical
    assert np.linalg.norm(x1 - x0) > 0.0


# ----------------------- tests: performance -----------------------

def test_forces_moments_perf(sim: SixDOFSimulator):
    """
    10k calls should complete under FORCES_MOMENTS_BUDGET_S seconds.
    """
    set_random_controls(sim)

    xs = [random_state(sim) for _ in range(128)]
    idx = 0
    t0 = time.perf_counter()
    for _ in range(N_CALLS_INNER):
        x = xs[idx]
        idx = (idx + 1) % len(xs)
        Vb = x[0:3]
        p, q, r = x[3:6]
        sim._forces_moments(Vb, p, q, r)
    elapsed = time.perf_counter() - t0
    elapsed *= 1000
    print(f"_forces_moments: total {elapsed:.3f}ms,  each call {elapsed/N_CALLS_INNER:.3f}ms")


def test_rhs_perf(sim: SixDOFSimulator):
    """
    10k calls should complete under RHS_BUDGET_S seconds.
    """
    set_random_controls(sim)
    xs = [random_state(sim) for _ in range(128)]
    idx = 0
    t0 = time.perf_counter()
    for _ in range(N_CALLS_INNER):
        x = xs[idx]
        idx = (idx + 1) % len(xs)
        sim._rhs(x, 0.0)
    elapsed = time.perf_counter() - t0
    elapsed *= 1000
    print(f"_rhs: total {elapsed:.3f}ms,  each call {elapsed/N_CALLS_INNER:.3f}ms")


def test_step_perf(sim: SixDOFSimulator):
    """
    1k step() advances should complete under STEP_BUDGET_S seconds.
    """
    set_random_controls(sim)
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        # Slightly vary controls to avoid identical solver path
        set_random_controls(sim)
        sim.step()
    elapsed = time.perf_counter() - t0
    elapsed *= 1000
    print(f"step: total {elapsed:.3f}ms,  each call {elapsed/N_STEPS:.3f}ms")


if __name__ == "__main__":
    N_CALLS_INNER = 1_000
    N_STEPS       = 1_000

    sim = SixDOFSimulator(condition="CruiseLow", dt=0.1, t_max=30.0)
    test_forces_moments_sanity(sim)
    test_rhs_sanity(sim)
    test_step_sanity_progress(sim)

    test_forces_moments_perf(sim)
    test_rhs_perf(sim)
    test_step_perf(sim)
