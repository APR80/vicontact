"""Time stepping loop and trajectory recording."""

from __future__ import annotations
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .stepper import ContactStepper
from .world import World

Array = np.ndarray


@dataclass
class Recording:
    name: str
    h: float
    q: Array  # (N, nq)
    poses: Array  # (N, nbody, 3) including static bodies, for rendering
    kinetic: Array
    potential: Array
    n_contacts: Array
    comp_residual: Array
    del_residual: Array
    feas_residual: Array
    rejected: Array
    contact_points: list[Array] = field(default_factory=list)
    contact_fn: list[Array] = field(default_factory=list)
    contact_ft: list[Array] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "Recording":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def run(
    world: World,
    Tf: float,
    h: float,
    margin: float = 0.02,
    progress_every: float = 0.5,
    verbose: bool = True,
    **stepper_kw,
) -> Recording:
    stepper = ContactStepper(world, h, margin=margin, **stepper_kw)
    N = int(round(Tf / h)) + 1
    nq = world.nq

    q = np.zeros((N, nq))
    q[0] = world.q_init()
    # Discrete Legendre transform of the initial momentum: p0 = -D1 Ld(q0, q1).
    q[1] = q[0] + h * world.v_init() - 0.5 * h**2 * (stepper.Minv @ stepper.grad_V)

    kinetic = np.zeros(N)
    potential = np.zeros(N)
    ncon = np.zeros(N, dtype=int)
    comp = np.zeros(N)
    dres = np.zeros(N)
    fres = np.zeros(N)
    rej = np.zeros(N, dtype=bool)
    pts: list[Array] = [np.zeros((0, 2)) for _ in range(N)]
    fns: list[Array] = [np.zeros(0) for _ in range(N)]
    fts: list[Array] = [np.zeros(0) for _ in range(N)]

    worst_comp = 0.0
    worst_feas = 0.0
    bad_steps: list[tuple[int, str]] = []
    t_start = time.perf_counter()
    next_report = progress_every

    for k in range(1, N - 1):
        r = stepper.step(q[k - 1], q[k])
        q[k + 1] = r.q
        ncon[k] = len(r.contacts)
        comp[k] = r.comp_residual
        dres[k] = r.del_residual
        fres[k] = r.feas_residual
        rej[k] = not r.accepted
        worst_comp = max(worst_comp, r.comp_residual)
        worst_feas = max(worst_feas, r.feas_residual)
        if not r.accepted and len(bad_steps) < 20:
            bad_steps.append((k, r.status))
        if r.contacts:
            pts[k] = np.array([c["point"] for c in r.contacts])
            fns[k] = np.array([c["fn"] for c in r.contacts])
            fts[k] = np.array([c["ft"] for c in r.contacts])
        kinetic[k], potential[k] = stepper.energy(q[k - 1], q[k])

        if verbose and (k * h) >= next_report:
            next_report += progress_every
            el = time.perf_counter() - t_start
            print(
                f"  t={k * h:6.3f}s  step {k:5d}/{N - 2}  contacts={ncon[k]:3d}  "
                f"comp={comp[k]:.2e}  solvers={stepper.stats['builds']:3d}  "
                f"elapsed={el:6.1f}s",
                flush=True,
            )

    kinetic[0], potential[0] = stepper.energy(q[0], q[0])
    kinetic[-1], potential[-1] = stepper.energy(q[-2], q[-1])

    poses = np.stack([world.all_poses(q[k]) for k in range(N)])
    elapsed = time.perf_counter() - t_start
    if verbose:
        print(
            f"  done in {elapsed:.1f}s  |  {stepper.stats['builds']} solvers built "
            f"({stepper.stats['build_time']:.1f}s)  |  {stepper.stats['fallbacks']} retries "
            f"|  worst complementarity residual {worst_comp:.2e}"
        )

    return Recording(
        name=world.name,
        h=h,
        q=q,
        poses=poses,
        kinetic=kinetic,
        potential=potential,
        n_contacts=ncon,
        comp_residual=comp,
        del_residual=dres,
        feas_residual=fres,
        rejected=rej,
        contact_points=pts,
        contact_fn=fns,
        contact_ft=fts,
        meta={
            "bodies": [
                {
                    "name": b.name,
                    "verts": b.verts,
                    "static": b.static,
                    "color": b.color,
                    "edge": b.edge,
                    "zorder": b.zorder,
                    "kind": b.kind,
                    "mass": b.mass,
                }
                for b in world.bodies
            ],
            "Tf": Tf,
            "margin": margin,
            "elapsed": elapsed,
            "solvers_built": stepper.stats["builds"],
            "retries": stepper.stats["fallbacks"],
            "worst_comp": worst_comp,
            "worst_feas": worst_feas,
            "rejected": int(rej.sum()),
            "bad_steps": bad_steps,
            "total_weight": world.total_weight,
        },
    )
