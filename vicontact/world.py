"""Rigid bodies and the scene graph."""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
from . import geometry as geo

Array = np.ndarray


@dataclass
class Body:
    name: str
    verts: Array  # body frame, recentred on the centroid in __post_init__
    static: bool = False
    density: float = 700.0
    mu: float = 0.5
    q0: Array = field(default_factory=lambda: np.zeros(3))
    v0: Array = field(default_factory=lambda: np.zeros(3))
    color: str = "#7dd3fc"
    edge: str | None = None
    zorder: float = 2.0
    kind: str = "body"

    mass: float = field(init=False, default=0.0)
    inertia: float = field(init=False, default=0.0)
    area: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        v = geo.as_ccw(np.asarray(self.verts, dtype=float))
        a, c = geo.area_centroid(v)
        v = v - c  # body frame origin == centre of mass
        if not geo.is_convex(v):
            raise ValueError(f"body {self.name!r} is not convex")
        self.verts = v
        self.area = a
        self.mass = self.density * a
        self.inertia = self.density * geo.polar_second_moment(v)
        self.q0 = np.asarray(self.q0, dtype=float).copy()
        self.v0 = np.asarray(self.v0, dtype=float).copy()
        if self.edge is None:
            self.edge = self.color

    @property
    def radius(self) -> float:
        return float(np.linalg.norm(self.verts, axis=1).max())

    def world_verts(self, q: Array) -> Array:
        return geo.transform(self.verts, q)


# Convenience constructors
def box(
    name: str, w: float, h: float, cx: float, cy: float, angle: float = 0.0, **kw
) -> Body:
    hw, hh = 0.5 * w, 0.5 * h
    v = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    return Body(name=name, verts=v, q0=np.array([cx, cy, angle]), **kw)


def polygon(name: str, pts, **kw) -> Body:
    """Polygon given in world coordinates; the pose is set to its centroid."""
    v = geo.as_ccw(np.asarray(pts, dtype=float))
    _, c = geo.area_centroid(v)
    return Body(name=name, verts=v, q0=np.array([c[0], c[1], 0.0]), **kw)


def ngon(
    name: str, radius: float, n: int, cx: float, cy: float, angle: float = 0.0, **kw
) -> Body:
    """Regular n-gon."""
    th = np.arange(n) * (2.0 * np.pi / n)
    v = np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1)
    return Body(name=name, verts=v, q0=np.array([cx, cy, angle]), **kw)


# World
class World:
    def __init__(
        self, bodies: list[Body], gravity: float = 9.81, name: str = "scene"
    ) -> None:
        self.bodies = bodies
        self.gravity = gravity
        self.name = name

        self.dyn = [i for i, b in enumerate(bodies) if not b.static]
        self.dyn_of = {bi: k for k, bi in enumerate(self.dyn)}
        self.nq = 3 * len(self.dyn)

        self.mass_diag = np.concatenate(
            [[bodies[i].mass, bodies[i].mass, bodies[i].inertia] for i in self.dyn]
        )
        self.weight_vec = np.concatenate(
            [[0.0, bodies[i].mass * gravity, 0.0] for i in self.dyn]
        )
        self.total_weight = float(sum(bodies[i].mass for i in self.dyn) * gravity)

        self.exclude: set[tuple[int, int]] = set()

        self._static_verts = {
            i: bodies[i].world_verts(bodies[i].q0)
            for i, b in enumerate(bodies)
            if b.static
        }
        self._static_aabb = {i: geo.aabb(v) for i, v in self._static_verts.items()}

    # -- state helpers ---

    def q_init(self) -> Array:
        return np.concatenate([self.bodies[i].q0 for i in self.dyn])

    def v_init(self) -> Array:
        return np.concatenate([self.bodies[i].v0 for i in self.dyn])

    def pose(self, body_index: int, q: Array) -> Array:
        k = self.dyn_of.get(body_index)
        if k is None:
            return self.bodies[body_index].q0
        return q[3 * k : 3 * k + 3]

    def all_poses(self, q: Array) -> Array:
        return np.stack([self.pose(i, q) for i in range(len(self.bodies))])

    def exclude_pair(self, a: str, b: str) -> None:
        ia = next(i for i, x in enumerate(self.bodies) if x.name == a)
        ib = next(i for i, x in enumerate(self.bodies) if x.name == b)
        self.exclude.add((min(ia, ib), max(ia, ib)))

    # -- collision detection --
    def find_contacts(self, q: Array, margin: float) -> list[dict]:
        """Broad phase + narrow phase for the configuration q."""
        verts: dict[int, Array] = {}
        boxes: dict[int, tuple[float, float, float, float]] = {}
        for i, b in enumerate(self.bodies):
            if b.static:
                verts[i] = self._static_verts[i]
                boxes[i] = self._static_aabb[i]
            else:
                verts[i] = b.world_verts(self.pose(i, q))
                boxes[i] = geo.aabb(verts[i], margin)

        out: list[dict] = []
        for ia, ib in combinations(range(len(self.bodies)), 2):
            if self.bodies[ia].static and self.bodies[ib].static:
                continue
            if (ia, ib) in self.exclude:
                continue
            if not geo.aabb_overlap(boxes[ia], boxes[ib]):
                continue

            mu = float(np.sqrt(self.bodies[ia].mu * self.bodies[ib].mu))
            for c in geo.collide(verts[ia], verts[ib], margin):
                n = c.n
                that = np.array([-n[1], n[0]])
                rec: dict = {
                    "ia": ia,
                    "ib": ib,
                    "ka": self.dyn_of.get(ia, -1),
                    "kb": self.dyn_of.get(ib, -1),
                    "n": n,
                    "t": that,
                    "sep": c.sep,
                    "mu": mu,
                    "point": 0.5 * (c.pa + c.pb),
                }
                # Jacobian rows: dg/dq = [n, cross(r, n)] for a, negated for b.
                if rec["ka"] >= 0:
                    r = c.pa - self.pose(ia, q)[:2]
                    rec["Jna"] = np.array([n[0], n[1], r[0] * n[1] - r[1] * n[0]])
                    rec["Jta"] = np.array(
                        [that[0], that[1], r[0] * that[1] - r[1] * that[0]]
                    )
                else:
                    rec["Jna"] = np.zeros(3)
                    rec["Jta"] = np.zeros(3)
                if rec["kb"] >= 0:
                    r = c.pb - self.pose(ib, q)[:2]
                    rec["Jnb"] = -np.array([n[0], n[1], r[0] * n[1] - r[1] * n[0]])
                    rec["Jtb"] = -np.array(
                        [that[0], that[1], r[0] * that[1] - r[1] * that[0]]
                    )
                else:
                    rec["Jnb"] = np.zeros(3)
                    rec["Jtb"] = np.zeros(3)
                out.append(rec)

        # Canonical order
        out.sort(
            key=lambda r: (
                r["ka"],
                r["kb"],
                round(r["point"][0], 6),
                round(r["point"][1], 6),
            )
        )
        return out
