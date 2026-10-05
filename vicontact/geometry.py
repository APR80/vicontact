"""Convex-polygon geometry: mass properties and SAT contact manifolds."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray


# Basic polygon utilities
def rot(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform(verts: Array, q: Array) -> Array:
    """Body-frame vertices -> world frame for pose q = (x, y, theta)."""
    return verts @ rot(q[2]).T + q[:2]


def area_centroid(v: Array) -> tuple[float, Array]:
    x, y = v[:, 0], v[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a = 0.5 * float(cr.sum())
    cx = float(((x + xn) * cr).sum()) / (6.0 * a)
    cy = float(((y + yn) * cr).sum()) / (6.0 * a)
    return a, np.array([cx, cy])


def as_ccw(v: Array) -> Array:
    a, _ = area_centroid(v)
    return np.array(v, dtype=float) if a > 0 else np.array(v[::-1], dtype=float)


def polar_second_moment(v: Array) -> float:
    """Unit-density second moment about the origin."""
    vn = np.roll(v, -1, axis=0)
    cr = v[:, 0] * vn[:, 1] - vn[:, 0] * v[:, 1]
    t = (v * v).sum(1) + (v * vn).sum(1) + (vn * vn).sum(1)
    return float((cr * t).sum()) / 12.0


def edge_normals(v: Array) -> Array:
    """Outward unit normals of a CCW polygon, one per edge ``i -> i+1``."""
    e = np.roll(v, -1, axis=0) - v
    n = np.stack([e[:, 1], -e[:, 0]], axis=1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def is_convex(v: Array, tol: float = 1e-9) -> bool:
    e = np.roll(v, -1, axis=0) - v
    en = np.roll(e, -1, axis=0)
    cr = e[:, 0] * en[:, 1] - en[:, 0] * e[:, 1]
    return bool(np.all(cr > -tol))


# Contact manifold
@dataclass(slots=True)
class Contact:
    """One contact point between polygon a and b."""

    pa: Array
    pb: Array
    n: Array
    sep: float


def _best_axis(ref: Array, refn: Array, inc: Array) -> tuple[float, int]:
    """Largest separation of inc behind any face of ref."""
    # d[j, i] = refn[i] . (inc[j] - ref[i])
    d = inc @ refn.T - (ref * refn).sum(1)
    s = d.min(axis=0)
    i = int(np.argmax(s))
    return float(s[i]), i


def _closest_on_segment(p: Array, a: Array, b: Array) -> Array:
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-18:
        return a
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return a + t * ab


def collide(va: Array, vb: Array, margin: float) -> list[Contact]:
    """Contact manifold between two convex CCW polygons in world coordinates."""
    na, nb = edge_normals(va), edge_normals(vb)
    sa, ia = _best_axis(va, na, vb)
    sb, ib = _best_axis(vb, nb, va)

    if max(sa, sb) > margin:
        return []
    ref_is_a = sa >= sb - 1e-9
    if ref_is_a:
        ref, ei, inc, incn = va, ia, vb, nb
    else:
        ref, ei, inc, incn = vb, ib, va, na
    n = (na if ref_is_a else nb)[ei]

    r1 = ref[ei]
    r2 = ref[(ei + 1) % len(ref)]
    tvec = r2 - r1
    length = float(np.linalg.norm(tvec))
    that = tvec / length

    # Incident face
    j = int(np.argmin(incn @ n))
    p, qv = inc[j], inc[(j + 1) % len(inc)]

    # Cliping incident segment to the tangential extent of the reference face.
    s0 = float(that @ (p - r1))
    ds = float(that @ (qv - p))
    lo, hi = 0.0, 1.0
    if abs(ds) > 1e-12:
        u1, u2 = (0.0 - s0) / ds, (length - s0) / ds
        lo = max(lo, min(u1, u2))
        hi = min(hi, max(u1, u2))
    elif not (-1e-9 <= s0 <= length + 1e-9):
        lo, hi = 1.0, 0.0  # force the vertex fallback below

    out: list[Contact] = []
    if hi >= lo:
        pts = [p + lo * (qv - p)]
        if hi - lo > 1e-9:
            pts.append(p + hi * (qv - p))
        for x in pts:
            s = float(n @ (x - r1))
            if s > margin:
                continue
            proj = x - s * n
            if ref_is_a:
                out.append(Contact(pa=proj, pb=x, n=-n, sep=s))
            else:
                out.append(Contact(pa=x, pb=proj, n=n, sep=s))
    else:
        # No tangential overlap
        x = p if abs(s0) < abs(s0 + ds - length) else qv
        y = _closest_on_segment(x, r1, r2)
        d = x - y
        dist = float(np.linalg.norm(d))
        nn = d / dist if dist > 1e-12 else n
        if dist <= margin:
            if ref_is_a:
                out.append(Contact(pa=y, pb=x, n=-nn, sep=dist))
            else:
                out.append(Contact(pa=x, pb=y, n=nn, sep=dist))
    return out


def aabb(verts: Array, pad: float = 0.0) -> tuple[float, float, float, float]:
    lo = verts.min(axis=0) - pad
    hi = verts.max(axis=0) + pad
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def aabb_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
