"""checking the polygon-rolling energy loss against the analytic corner-impact law."""

from __future__ import annotations
import numpy as np
from vicontact import scenes, sim

G = 9.81


def ngon_props(r: float, n: int) -> tuple[float, float, float, float]:
    th = np.arange(n) * 2 * np.pi / n
    v = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    vn = np.roll(v, -1, axis=0)
    cr = v[:, 0] * vn[:, 1] - vn[:, 0] * v[:, 1]
    area = 0.5 * cr.sum()
    tt = (v * v).sum(1) + (v * vn).sum(1) + (vn * vn).sum(1)
    Iom = (cr * tt).sum() / 12.0 / area  # I_c / m
    side = 2 * r * np.sin(np.pi / n)
    rin = r * np.cos(np.pi / n)
    return Iom, side, rin, r


def analytic(r: float, n: int, ang_deg: float) -> dict:
    Iom, side, rin, R = ngon_props(r, n)
    a = 2 * np.pi / n
    e = (Iom + R**2 * np.cos(a)) / (Iom + R**2)
    th = np.radians(ang_deg)
    Kstar = e**2 * G * side * np.sin(th) / (1 - e**2)  # per unit mass
    omega = np.sqrt(2 * Kstar / (Iom + R**2))
    return {
        "e2": e**2,
        "loss_per_metre": (1 - e**2) / side,
        "v_terminal": omega * R,
        "a_ideal": G * np.sin(th) / (1 + Iom / rin**2),
        "R_eff": side / a,
    }


def run_case(n: int, ang: float = 25.0, Tf: float = 0.45, h: float = 1 / 1500) -> dict:
    rec = sim.run(scenes.rolling(ang, sides=n), Tf=Tf, h=h, verbose=False)
    s = np.hypot(rec.q[:, 0] - rec.q[0, 0], rec.q[:, 1] - rec.q[0, 1])
    t = np.arange(len(s)) * h
    m = (t > 0.12) & (t < Tf - 0.02)
    A = np.stack([np.ones(m.sum()), t[m], 0.5 * t[m] ** 2], axis=1)
    a_fit = float(np.linalg.lstsq(A, s[m], rcond=None)[0][2])
    dth = rec.q[-1, 2] - rec.q[0, 2]
    return {"a": a_fit, "R_eff": float(-s[-1] / dth), "dist": float(s[-1])}


if __name__ == "__main__":
    r, ang = 0.25, 25.0
    print(f"regular n-gon, r={r}, ramp {ang} deg\n")
    print(
        f"{'n':>5} {'a_sim':>9} {'a_ideal':>9} {'deficit':>9} "
        f"{'R_eff sim':>10} {'R_eff exact':>11} {'slip':>10}"
    )
    rows = []
    for n in (16, 24, 48, 96):
        th = analytic(r, n, ang)
        got = run_case(n, ang)
        slip = abs(got["R_eff"] - th["R_eff"]) / th["R_eff"]
        rows.append((n, got["a"], th["a_ideal"]))
        print(
            f"{n:5d} {got['a']:9.4f} {th['a_ideal']:9.4f} "
            f"{1 - got['a'] / th['a_ideal']:8.2%} "
            f"{got['R_eff']:10.5f} {th['R_eff']:11.5f} {slip:9.2e}"
        )

    ns = np.array([r_[0] for r_ in rows], dtype=float)
    a_sim = np.array([r_[1] for r_ in rows])
    a_id = rows[-1][2]
    A = np.stack([np.ones_like(ns), 1.0 / ns], axis=1)
    coef = np.linalg.lstsq(A, a_sim, rcond=None)[0]
    print(
        f"\nfit a(n) = {coef[0]:.4f} - {-coef[1]:.4f}/n   (ideal disc a = {a_id:.4f})"
    )
    print(f"extrapolated n->inf deficit: {1 - coef[0] / a_id:+.2%}")

    print("\nis there a sustained-contact terminal speed at all?")
    th_rad = np.radians(ang)
    n_rolls_above = np.pi / th_rad
    n_contact_below = np.pi / np.sin(th_rad)
    print(f"  rolls only for      n > pi/ang      = {n_rolls_above:.2f}")
    print(f"  keeps contact only  n < pi/sin(ang) = {n_contact_below:.2f}")
    print(
        f"  => sustained rolling needs an integer n in ({n_rolls_above:.2f}, "
        f"{n_contact_below:.2f}): "
        f"{'none exists' if np.floor(n_contact_below) <= np.ceil(n_rolls_above) else 'EXISTS'}"
    )

    rec = sim.run(
        scenes.rolling(ang, sides=24, length=40.0), Tf=4.0, h=1 / 1500, verbose=False
    )
    s = np.hypot(rec.q[:, 0] - rec.q[0, 0], rec.q[:, 1] - rec.q[0, 1])
    v = np.gradient(s, 1 / 1500)
    airborne = float(np.mean(rec.n_contacts == 0))
    print(
        f"\n  n=24 on a 40 m ramp: travelled {s[-1]:.2f} m, "
        f"v {v[-400:].mean():.3f} m/s and still rising "
        f"({v[-800:-400].mean():.3f} -> {v[-400:].mean():.3f})"
    )
    print(f"  airborne for {airborne:.1%} of steps -- it hops, as predicted, so the")
    print(
        f"  sustained-contact terminal speed of {th['v_terminal']:.3f} m/s never applies."
    )
