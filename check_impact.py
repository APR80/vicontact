"""Validate the impact law with a single, controlled corner slap."""

from __future__ import annotations

import numpy as np

from vicontact import World, ngon, scenes, sim

G = 9.81


def ngon_Iom(r: float, n: int) -> float:
    th = np.arange(n) * 2 * np.pi / n
    v = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    vn = np.roll(v, -1, axis=0)
    cr = v[:, 0] * vn[:, 1] - vn[:, 0] * v[:, 1]
    area = 0.5 * cr.sum()
    tt = (v * v).sum(1) + (v * vn).sum(1) + (vn * vn).sum(1)
    return float((cr * tt).sum()) / 12.0 / float(area)


def build(n: int, r: float, eps_deg: float) -> World:
    eps = np.radians(eps_deg)
    th0 = 1.5 * np.pi + eps
    body = ngon("poly", r, n, 0.0, 0.0, angle=th0, mu=0.9, color="#38bdf8")
    v = body.world_verts(np.array([0.0, 0.0, th0]))
    body.q0 = np.array([0.0, -float(v[:, 1].min()) + 5e-4, th0])
    floor = scenes._floor(8.0, 0.0, mu=0.9)
    return World([floor, body], name=f"slap{n}")


def run(n: int, r: float = 0.25, eps_deg: float = 6.0, h: float = 1 / 4000) -> dict:
    Iom = ngon_Iom(r, n)
    alpha = 2 * np.pi / n
    e = (Iom + r**2 * np.cos(alpha)) / (Iom + r**2)
    w_before = np.sqrt(
        2 * G * r * (np.cos(np.radians(eps_deg)) - np.cos(np.pi / n)) / (Iom + r**2)
    )
    Tf = 4.0 * (np.pi / n) / max(w_before, 1e-3)
    rec = sim.run(build(n, r, eps_deg), Tf=Tf, h=h, verbose=False, margin=0.01)

    th = rec.q[:, 2]
    w = np.gradient(th, h)
    dw = np.diff(w)
    k = int(np.argmin(dw))
    pre = float(np.median(w[max(0, k - 12) : k - 1]))
    post = float(np.median(w[k + 3 : k + 16]))
    return {
        "n": n,
        "e_exact": e,
        "e_sim": post / pre if abs(pre) > 1e-9 else np.nan,
        "w_before_exact": w_before,
        "w_before_sim": pre,
        "w_after_exact": e * w_before,
        "w_after_sim": post,
    }


if __name__ == "__main__":
    print("single corner slap: omega jump vs analytic angular-momentum transfer\n")
    print(
        f"{'n':>4} {'w- sim':>9} {'w- exact':>9} {'err':>7} | "
        f"{'w+ sim':>9} {'w+ exact':>9} | {'e sim':>8} {'e exact':>8} {'err':>7}"
    )
    worst = 0.0
    for n in (4, 5, 6, 8, 12):
        d = run(n)
        e_err = abs(d["e_sim"] - d["e_exact"]) / d["e_exact"]
        w_err = abs(d["w_before_sim"] - d["w_before_exact"]) / d["w_before_exact"]
        worst = max(worst, e_err, w_err)
        print(
            f"{n:4d} {d['w_before_sim']:9.4f} {d['w_before_exact']:9.4f} {w_err:6.2%} | "
            f"{d['w_after_sim']:9.4f} {d['w_after_exact']:9.4f} | "
            f"{d['e_sim']:8.4f} {d['e_exact']:8.4f} {e_err:6.2%}"
        )
    print(f"\nworst relative error: {worst:.2%}")
    raise SystemExit(0 if worst < 0.05 else 1)
