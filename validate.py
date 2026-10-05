"""Physics validation. every scene here has a closed-form answer to compare to."""

from __future__ import annotations
import numpy as np
from vicontact import scenes, sim


def _fit_accel(t: np.ndarray, s: np.ndarray) -> float:
    """Quadratic fit s = s0 + v0 t + a t^2 / 2 -> return a."""
    A = np.stack([np.ones_like(t), t, 0.5 * t**2], axis=1)
    return float(np.linalg.lstsq(A, s, rcond=None)[0][2])


def check(name: str, got: float, want: float, tol: float) -> bool:
    rel = abs(got - want) / max(abs(want), 1e-12)
    ok = rel <= tol
    print(
        f"  {'PASS' if ok else 'FAIL'}  {name:34s} got {got: .5f}  want {want: .5f}  rel {rel:.2%}"
    )
    return ok


def test_slide() -> bool:
    mu, v0 = 0.3, 3.0
    h = 1.0 / 800
    rec = sim.run(scenes.slide(mu=mu, v0=v0), Tf=0.6, h=h, verbose=False)
    x = rec.q[:, 0]
    t = np.arange(len(x)) * h
    v = np.gradient(x, h)
    m = (t > 0.05) & (v > 0.4 * v0)
    a = _fit_accel(t[m], x[m])
    return check("sliding block  a = -mu g", a, -mu * 9.81, 0.02)


def test_incline() -> bool:
    ang, mu = 30.0, 0.25
    a_exp = 9.81 * (np.sin(np.radians(ang)) - mu * np.cos(np.radians(ang)))
    h = 1.0 / 800
    rec = sim.run(scenes.incline(ang, mu), Tf=0.7, h=h, verbose=False)
    s = np.hypot(rec.q[:, 0] - rec.q[0, 0], rec.q[:, 1] - rec.q[0, 1])
    t = np.arange(len(s)) * h
    m = t > 0.15
    return check("block on 30 deg ramp  a", _fit_accel(t[m], s[m]), a_exp, 0.02)


def test_rolling() -> bool:
    """A polygon is not a disc: it loses energy at every pivot."""
    ang, h, Tf = 25.0, 1.0 / 1500, 0.45
    ns = np.array([16.0, 24.0, 48.0, 96.0])
    a_disc = 9.81 * np.sin(np.radians(ang)) / 1.5  # I/(m r^2) = 1/2 for a disc
    acc = []
    for n in ns:
        rec = sim.run(scenes.rolling(ang, sides=int(n)), Tf=Tf, h=h, verbose=False)
        s = np.hypot(rec.q[:, 0] - rec.q[0, 0], rec.q[:, 1] - rec.q[0, 1])
        t = np.arange(len(s)) * h
        m = (t > 0.12) & (t < Tf - 0.02)
        acc.append(_fit_accel(t[m], s[m]))
    acc = np.array(acc)
    A = np.stack([np.ones_like(ns), 1.0 / ns], axis=1)
    coef = np.linalg.lstsq(A, acc, rcond=None)[0]
    resid = float(np.abs(A @ coef - acc).max())
    print("        " + "   ".join(f"a({int(n)})={a:.4f}" for n, a in zip(ns, acc)))
    print(
        f"        fit a(n) = {coef[0]:.4f} - {-coef[1]:.4f}/n  (max resid {resid:.1e})"
    )
    ok = check("rolling a extrapolated to n->inf", float(coef[0]), a_disc, 0.01)
    if resid > 0.02:
        print(f"  FAIL  deficit is not 1/n: fit residual {resid:.2e} > 2e-2")
        ok = False
    return bool(ok)


def test_stack() -> bool:
    n = 4
    h = 1.0 / 600
    rec = sim.run(scenes.stack(n), Tf=1.2, h=h, verbose=False)
    q = rec.q.reshape(len(rec.q), n, 3)
    drift = np.abs(q[-1, :, 0] - q[0, :, 0]).max()
    tilt = np.abs(q[-1, :, 2]).max()
    sink = np.abs(q[-1, :, 1] - q[0, :, 1]).max()
    ok = bool(drift < 2e-3 and tilt < 2e-3 and sink < 5e-3)
    print(
        f"  {'PASS' if ok else 'FAIL'}  {'stack stays put':34s} "
        f"drift {drift:.2e} m   tilt {tilt:.2e} rad   settle {sink:.2e} m"
    )
    return ok


def test_energy_freefall() -> bool:
    """Before any contact the integrator must conserve energy to O(h^2)."""
    h = 1.0 / 600
    rec = sim.run(scenes.tumble(), Tf=0.35, h=h, verbose=False)
    E = rec.kinetic[2:-2] + rec.potential[2:-2]
    rel = float((E.max() - E.min()) / abs(E.mean()))
    print(f"        free-flight energy spread = {rel:.2e}")
    return rel < 1e-3


def test_solver_health() -> bool:
    """No step may be integrated from an infeasible or unconverged solve."""
    h = 1.0 / 600
    ok = True
    for name, scene, Tf in (
        ("stack(4)", scenes.stack(4), 1.2),
        ("tumble", scenes.tumble(), 1.0),
    ):
        rec = sim.run(scene, Tf=Tf, h=h, verbose=False)
        nrej = int(rec.rejected.sum())
        worst = float(rec.feas_residual.max())
        good = nrej == 0 and worst < 1e-7
        ok &= good
        print(
            f"  {'PASS' if good else 'FAIL'}  {name:18s} rejected={nrej:4d}  "
            f"worst constraint violation={worst:.2e}"
        )
    return bool(ok)


if __name__ == "__main__":
    print("validation")
    results = {
        "slide": test_slide(),
        "incline": test_incline(),
        "rolling": test_rolling(),
        "stack": test_stack(),
        "energy": test_energy_freefall(),
        "solver": test_solver_health(),
    }
    print()
    bad = [k for k, v in results.items() if not v]
    print("all passed" if not bad else f"failed: {', '.join(bad)}")
    raise SystemExit(1 if bad else 0)
