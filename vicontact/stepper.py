"""Variational integrator with frictional contact, solved step-by-step by IPOPT."""

from __future__ import annotations
import time
from dataclasses import dataclass, field
import casadi as ca
import numpy as np
from .world import World

Array = np.ndarray


@dataclass
class StepResult:
    q: Array
    contacts: list[dict] = field(default_factory=list)
    comp_residual: float = 0.0
    del_residual: float = 0.0
    feas_residual: float = 0.0
    iters: int = 0
    status: str = "ok"
    accepted: bool = True
    solve_time: float = 0.0


class ContactStepper:
    """One IPOPT solve per timestep, with a solver cached per contact topology."""

    def __init__(
        self,
        world: World,
        h: float,
        margin: float = 0.02,
        pen_frac: float = 0.3,
        pen_rate: float = 0.2,
        reg: float = 1e-7,
        max_iter: int = 400,
        tol: float = 1e-9,
        comp_tol: float = 1e-5,
        feas_tol: float = 1e-8,
        cache_limit: int = 800,
    ) -> None:
        self.world = world
        self.h = h
        self.margin = margin
        self.pen_frac = pen_frac
        self.pen_rate = pen_rate
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.comp_tol = comp_tol
        self.feas_tol = feas_tol
        self.cache_limit = cache_limit

        nq = world.nq
        self.nq = nq
        self.M = np.diag(world.mass_diag)
        self.Minv = np.diag(1.0 / world.mass_diag)
        self.grad_V = world.weight_vec.copy()
        self.F0 = max(world.total_weight, 1e-6)
        self.V0 = 1.0
        self.Lc = h * self.V0
        masses = np.asarray(world.mass_diag, dtype=float)[0::3]  # translational
        self.Mref = float(np.mean(masses)) if masses.size else 1.0
        self.Fc = self.Mref * self.V0 / h
        self.row_scale = 1.0 / (np.asarray(world.mass_diag, dtype=float) * self.V0)

        self._build_discrete_lagrangian()
        self._cache: dict[tuple, tuple[ca.Function, int]] = {}
        self._prev_key: tuple | None = None
        self._prev_sol: dict | None = None
        self._prev_imp: dict[tuple[int, int], list[tuple[Array, Array]]] = {}
        self.stats = {
            "builds": 0,
            "solves": 0,
            "attempts": 0,
            "fallbacks": 0,
            "rejected": 0,
            "build_time": 0.0,
        }

    # -- discrete mechanics ---
    def _build_discrete_lagrangian(self) -> None:
        h, nq = self.h, self.nq
        Mdiag = ca.DM(self.world.mass_diag)
        gvec = ca.DM(self.grad_V)

        def lagrangian(q: ca.MX, qdot: ca.MX) -> ca.MX:
            kinetic = 0.5 * ca.dot(Mdiag * qdot, qdot)
            potential = ca.dot(gvec, q)  # sum_i m_i g y_i, linear in q
            return kinetic - potential

        def Ld(q1: ca.MX, q2: ca.MX) -> ca.MX:
            return h * lagrangian(0.5 * (q1 + q2), (q2 - q1) / h)

        q1 = ca.MX.sym("q1", nq)
        q2 = ca.MX.sym("q2", nq)
        q3 = ca.MX.sym("q3", nq)
        self.f_D2 = ca.Function("D2Ld", [q1, q2], [ca.gradient(Ld(q1, q2), q2)])
        self.f_D1 = ca.Function("D1Ld", [q2, q3], [ca.gradient(Ld(q2, q3), q2)])

    def free_step(self, qkm1: Array, qk: Array) -> Array:
        """Contact-free update, i.e. the DEL equation with zero contact force."""
        return 2.0 * qk - qkm1 - self.h**2 * (self.Minv @ self.grad_V)

    def momentum(self, qkm1: Array, qk: Array) -> Array:
        return np.asarray(self.f_D2(qkm1, qk)).ravel()

    def energy(self, qkm1: Array, qk: Array) -> tuple[float, float]:
        p = self.momentum(qkm1, qk)
        kinetic = 0.5 * float(p @ (self.Minv @ p))
        potential = float(self.grad_V @ qk)
        return kinetic, potential

    # -- NLP construction --
    def _solver_for(self, topo: tuple) -> tuple[ca.Function, int]:
        hit = self._cache.get(topo)
        if hit is not None:
            return hit
        if len(self._cache) >= self.cache_limit:
            self._cache.clear()

        t0 = time.perf_counter()
        built = self._build_nlp(topo)
        self._cache[topo] = built
        self.stats["builds"] += 1
        self.stats["build_time"] += time.perf_counter() - t0
        return built

    def _build_nlp(self, topo: tuple) -> tuple[ca.Function, int]:
        h, nq = self.h, self.nq
        C = len(topo)
        Fc, V0, Lc = self.Fc, self.V0, self.Lc

        qn = ca.MX.sym("qn", nq)
        # Decision variables for the contact forces are held in scaled units:
        z = ca.MX.sym("z", 4 * C)
        par = ca.MX.sym("p", 2 * nq + 14 * C)
        qkm1, qk = par[:nq], par[nq : 2 * nq]

        # Forced discrete Euler-Lagrange residual, accumulated slot by slot
        base = self.f_D2(qkm1, qk) + self.f_D1(qk, qn)
        res = [base[i] for i in range(nq)]

        gaps, vels, cones = [], [], []
        obj = ca.MX(0)
        for c, (ka, kb) in enumerate(topo):
            o = 2 * nq + 14 * c
            Jna, Jnb = par[o : o + 3], par[o + 3 : o + 6]
            Jta, Jtb = par[o + 6 : o + 9], par[o + 9 : o + 12]
            d_eff, mu = par[o + 12], par[o + 13]

            nh, bph, bmh, lh = z[4 * c], z[4 * c + 1], z[4 * c + 2], z[4 * c + 3]
            n, bp, bm = Fc * nh, Fc * bph, Fc * bmh
            lam = V0 * lh
            bt = bp - bm

            gap = d_eff
            tau = ca.MX(0)
            for k, Jn, Jt in ((ka, Jna, Jta), (kb, Jnb, Jtb)):
                if k < 0:
                    continue
                for i in range(3):
                    res[3 * k + i] = res[3 * k + i] + h * (Jn[i] * n + Jt[i] * bt)
                dq = qn[3 * k : 3 * k + 3] - qk[3 * k : 3 * k + 3]
                gap = gap + ca.dot(Jn, dq)
                tau = tau + ca.dot(Jt, dq)
            vt = tau / h
            cone = mu * n - bp - bm

            # Every constraint row is reported in its own natural unit
            gaps.append(gap / Lc)
            vels += [(lam + vt) / V0, (lam - vt) / V0]
            cones.append(cone / Fc)

            obj = obj + (n * gap) / (Fc * Lc)
            obj = obj + (bp * (lam + vt) + bm * (lam - vt) + lam * cone) / (Fc * V0)
            obj = obj + self.reg * (nh * nh + bph * bph + bmh * bmh + lh * lh)

        del_rows = [res[i] * float(self.row_scale[i]) for i in range(nq)]

        g = ca.vertcat(*del_rows, *gaps, *vels, *cones)
        nlp = {"x": ca.vertcat(qn, z), "p": par, "f": obj, "g": g}
        opts = {
            "expand": True,
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.max_iter,
            "ipopt.tol": self.tol,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.acceptable_iter": 5,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.honor_original_bounds": "yes",
            # we dnot let IPOPT relax g >= 0
            "ipopt.bound_relax_factor": 0.0,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.warm_start_bound_push": 1e-6,
            "ipopt.warm_start_mult_bound_push": 1e-6,
        }
        return ca.nlpsol("step", "ipopt", nlp, opts), int(g.numel())

    # -- warm start --
    def _guess_impulses(self, contacts: list[dict]) -> Array:
        """Reuse the previous step's impulses for contacts at nearby points."""
        x = np.zeros(4 * len(contacts))
        nominal = float(np.clip((self.F0 / max(len(contacts), 1)) / self.Fc, 1e-3, 1.0))
        for c, rec in enumerate(contacts):
            key = (rec["ia"], rec["ib"])
            best, bestd = None, np.inf
            for pt, val in self._prev_imp.get(key, ()):
                d = float(np.linalg.norm(pt - rec["point"]))
                if d < bestd:
                    best, bestd = val, d
            if best is not None and bestd < 0.15:
                x[4 * c : 4 * c + 4] = best
            else:
                x[4 * c : 4 * c + 4] = (nominal, 0.0, 0.0, 0.0)
        # Interior-point methods behave badly when started exactly on a bound.
        return np.maximum(x, 1e-4)

    def _store_impulses(self, contacts: list[dict], x: Array) -> None:
        store: dict[tuple[int, int], list[tuple[Array, Array]]] = {}
        for c, rec in enumerate(contacts):
            store.setdefault((rec["ia"], rec["ib"]), []).append(
                (rec["point"].copy(), x[self.nq + 4 * c : self.nq + 4 * c + 4].copy())
            )
        self._prev_imp = store

    # -- one step --
    def _assess(self, sol: dict, stats: dict) -> dict:
        """Score one candidate solve: feasibility first, complementarity second."""
        nq = self.nq
        x = np.asarray(sol["x"]).ravel()
        gval = np.asarray(sol["g"]).ravel()
        comp = float(sol["f"])
        del_res = float(np.abs(gval[:nq]).max()) if nq else 0.0
        viol = float(max(0.0, -gval[nq:].min())) if gval.size > nq else 0.0
        status = str(stats.get("return_status", "?"))
        finite = bool(np.all(np.isfinite(x)) and np.isfinite(comp))
        ok = (
            finite
            and viol <= self.feas_tol
            and del_res <= 1e-6
            and comp <= self.comp_tol
            and status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        )
        # Constraint violation dominates the score
        score = np.inf if not finite else 1e6 * viol + 1e3 * del_res + max(comp, 0.0)
        return {
            "x": x,
            "g": gval,
            "comp": comp,
            "del_res": del_res,
            "viol": viol,
            "status": status,
            "iters": int(stats.get("iter_count", 0)),
            "lam_x": np.asarray(sol["lam_x"]).ravel(),
            "lam_g": np.asarray(sol["lam_g"]).ravel(),
            "ok": ok,
            "score": float(score),
        }

    def step(self, qkm1: Array, qk: Array) -> StepResult:
        nq = self.nq
        contacts = self.world.find_contacts(qk, self.margin)

        if not contacts:
            self._prev_key, self._prev_sol, self._prev_imp = None, None, {}
            return StepResult(q=self.free_step(qkm1, qk))

        topo = tuple((r["ka"], r["kb"]) for r in contacts)
        solver, ng = self._solver_for(topo)
        C = len(contacts)

        pvals = np.empty(2 * nq + 14 * C)
        pvals[:nq] = qkm1
        pvals[nq : 2 * nq] = qk
        for c, r in enumerate(contacts):
            o = 2 * nq + 14 * c
            pvals[o : o + 3] = r["Jna"]
            pvals[o + 3 : o + 6] = r["Jnb"]
            pvals[o + 6 : o + 9] = r["Jta"]
            pvals[o + 9 : o + 12] = r["Jtb"]
            d = r["sep"]
            # Recover from penetration gradually
            pvals[o + 12] = (
                d if d >= 0.0 else -min(-d * self.pen_frac, self.h * self.pen_rate)
            )
            pvals[o + 13] = r["mu"]

        nx = nq + 4 * C
        common = {
            "p": pvals,
            "lbx": np.concatenate([np.full(nq, -np.inf), np.zeros(4 * C)]),
            "ubx": np.full(nx, np.inf),
            "lbg": np.zeros(ng),
            "ubg": np.concatenate([np.zeros(nq), np.full(ng - nq, np.inf)]),
        }
        qfree = self.free_step(qkm1, qk)

        # Starting points, tried in order until one yields an acceptable step.
        starts: list[dict] = []
        if self._prev_key == topo and self._prev_sol is not None:
            warm = dict(common)
            x0 = self._prev_sol["x"].copy()
            x0[:nq] = qfree
            x0[nq:] = np.maximum(x0[nq:], 1e-4)
            warm["x0"] = x0
            warm["lam_x0"] = self._prev_sol["lam_x"]
            warm["lam_g0"] = self._prev_sol["lam_g"]
            starts.append(warm)
        for guess in (
            self._guess_impulses(contacts),
            np.full(4 * C, 1e-2),
            np.full(4 * C, 1.0),
        ):
            starts.append(dict(common, x0=np.concatenate([qfree, guess])))

        t0 = time.perf_counter()
        best: dict | None = None
        for i, kwargs in enumerate(starts):
            cand = self._assess(solver(**kwargs), solver.stats())
            self.stats["attempts"] += 1
            if best is None or cand["score"] < best["score"]:
                best = cand
            if cand["ok"]:
                break
            if i + 1 < len(starts):
                self.stats["fallbacks"] += 1
        assert best is not None
        dt = time.perf_counter() - t0
        self.stats["solves"] += 1
        if not best["ok"]:
            self.stats["rejected"] += 1

        x = best["x"]
        self._prev_key = topo
        self._prev_sol = {"x": x.copy(), "lam_x": best["lam_x"], "lam_g": best["lam_g"]}
        self._store_impulses(contacts, x)

        Fc, V0, Lc = self.Fc, self.V0, self.Lc
        for c, r in enumerate(contacts):
            r["fn"] = float(x[nq + 4 * c]) * Fc
            r["ft"] = float(x[nq + 4 * c + 1] - x[nq + 4 * c + 2]) * Fc
            r["slip"] = float(x[nq + 4 * c + 3]) * V0
            r["gap_end"] = float(best["g"][nq + c]) * Lc

        return StepResult(
            q=x[:nq],
            contacts=contacts,
            comp_residual=best["comp"],
            del_residual=best["del_res"],
            feas_residual=best["viol"],
            iters=best["iters"],
            status=best["status"],
            accepted=bool(best["ok"]),
            solve_time=dt,
        )
