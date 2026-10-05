"""Scenarios: a validation ladder plus the scene used for the video."""

from __future__ import annotations
import numpy as np
from .world import Body, World, box, ngon, polygon

# Palette
BG = "#080b14"
STATIC_FILL = "#1b2334"
STATIC_EDGE = "#33415c"
ACCENT = "#f5a524"


def _lerp_hex(c0: str, c1: str, t: float) -> str:
    a = np.array([int(c0[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    b = np.array([int(c1[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    c = np.clip(a + t * (b - a), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(c)


def ramp_gradient(n: int) -> list[str]:
    """Teal -> indigo -> magenta, one colour per domino."""
    anchors = ["#22d3ee", "#818cf8", "#f472b6"]
    out = []
    for i in range(n):
        u = i / max(n - 1, 1) * (len(anchors) - 1)
        j = min(int(u), len(anchors) - 2)
        out.append(_lerp_hex(anchors[j], anchors[j + 1], u - j))
    return out


def _floor(w: float, cx: float, mu: float = 0.55, thickness: float = 0.6) -> Body:
    return box(
        "floor",
        w,
        thickness,
        cx,
        -0.5 * thickness,
        static=True,
        mu=mu,
        color=STATIC_FILL,
        edge=STATIC_EDGE,
        zorder=1.0,
        kind="static",
    )


# Validation scenes
def slide(mu: float = 0.3, v0: float = 3.0) -> World:
    """Block launched horizontally on flat ground: should decelerate at mu*g."""
    bodies = [
        _floor(14.0, 5.0, mu=mu),
        box(
            "block",
            0.4,
            0.4,
            0.0,
            0.2005,
            mu=mu,
            v0=np.array([v0, 0.0, 0.0]),
            color="#38bdf8",
        ),
    ]
    return World(bodies, name="slide")


def incline(angle_deg: float = 30.0, mu: float = 0.25) -> World:
    """Block released on a ramp: should accelerate at g(sin a - mu cos a)."""
    a = np.radians(angle_deg)
    L = 6.0
    surf = np.array([[0.0, 0.0], [L * np.cos(a), -L * np.sin(a)]])
    thick = 0.5
    down = np.array([np.sin(a), np.cos(a)]) * -thick
    pts = [surf[0], surf[1], surf[1] + down, surf[0] + down]
    n = np.array([np.sin(a), np.cos(a)])
    c = surf[0] + 0.8 * (surf[1] - surf[0]) + n * (0.2 + 0.0005)
    bodies = [
        polygon(
            "ramp",
            pts,
            static=True,
            mu=mu,
            color=STATIC_FILL,
            edge=STATIC_EDGE,
            kind="static",
        ),
        box("block", 0.4, 0.4, c[0], c[1], angle=-a, mu=mu, color="#38bdf8"),
    ]
    return World(bodies, name="incline")


def rolling(
    angle_deg: float = 25.0, mu: float = 0.8, sides: int = 48, length: float = 6.0
) -> World:
    """Disc on a ramp: should accelerate at g sin(a)/(1 + I/(m r^2)) = (2/3) g sin a."""
    a = np.radians(angle_deg)
    L = length
    surf = np.array([[0.0, 0.0], [L * np.cos(a), -L * np.sin(a)]])
    down = np.array([np.sin(a), np.cos(a)]) * -0.5
    pts = [surf[0], surf[1], surf[1] + down, surf[0] + down]
    r = 0.25
    n = np.array([np.sin(a), np.cos(a)])
    c = surf[0] + 0.15 * (surf[1] - surf[0]) + n * (r * np.cos(np.pi / sides) + 0.0005)
    bodies = [
        polygon(
            "ramp",
            pts,
            static=True,
            mu=mu,
            color=STATIC_FILL,
            edge=STATIC_EDGE,
            kind="static",
        ),
        ngon("disc", r, sides, c[0], c[1], mu=mu, color=ACCENT, kind="ball"),
    ]
    return World(bodies, name="rolling")


def stack(n: int = 4) -> World:
    """A tower of crates that must simply stand still."""
    s, gap = 0.3, 0.001
    bodies = [_floor(6.0, 1.0, mu=0.7)]
    for i in range(n):
        bodies.append(
            box(
                f"crate{i}",
                s,
                s,
                0.0,
                0.5 * s + i * (s + gap) + gap,
                mu=0.7,
                density=500.0,
                color="#c98a4b",
                edge="#e2b07a",
            )
        )
    return World(bodies, name="stack")


def tumble() -> World:
    """A brick thrown with spin onto the floor: corner impacts, slide, settle."""
    bodies = [
        _floor(14.0, 4.0, mu=0.5),
        box(
            "brick",
            0.8,
            0.35,
            0.0,
            1.6,
            angle=0.35,
            mu=0.5,
            v0=np.array([3.5, 0.0, -7.0]),
            color="#38bdf8",
        ),
    ]
    return World(bodies, name="tumble")


# Hero scene
def cascade(n_dom: int = 16) -> World:
    """Ball rolls down a ramp, topples a domino run, the last domino
    demolishes a crate stack."""
    dom_w, dom_h = 0.09, 0.62
    spacing = 0.40
    x0 = 0.25
    gap = 0.001

    apex = np.array([-2.2, 1.1])
    foot = np.array([-0.6, 0.0])
    ramp_pts = [[-2.2, 0.0], [foot[0], foot[1]], [apex[0], apex[1]]]

    last_x = x0 + (n_dom - 1) * spacing
    pivot = last_x + 0.5 * dom_w
    crate_s = 0.26
    crate_cx = pivot + 0.69 * dom_h + 0.5 * crate_s
    floor_hi = crate_cx + 1.8

    bodies: list[Body] = [
        _floor(floor_hi - (-3.2), 0.5 * (floor_hi + -3.2), mu=0.55),
        polygon(
            "ramp",
            ramp_pts,
            static=True,
            mu=0.7,
            color=STATIC_FILL,
            edge=STATIC_EDGE,
            kind="static",
        ),
    ]

    d = foot - apex
    d = d / np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])
    if nrm[1] < 0:
        nrm = -nrm
    r_ball, sides = 0.145, 48
    s_contact = apex + 0.20 * (foot - apex)
    cball = s_contact + nrm * (r_ball * np.cos(np.pi / sides) + 0.002)
    bodies.append(
        ngon(
            "ball",
            r_ball,
            sides,
            cball[0],
            cball[1],
            mu=0.8,
            density=760.0,
            color=ACCENT,
            edge="#ffd08a",
            kind="ball",
            zorder=4.0,
        )
    )

    colors = ramp_gradient(n_dom)
    for i in range(n_dom):
        bodies.append(
            box(
                f"dom{i}",
                dom_w,
                dom_h,
                x0 + i * spacing,
                0.5 * dom_h + gap,
                mu=0.45,
                density=1150.0,
                color=colors[i],
                edge=_lerp_hex(colors[i], "#ffffff", 0.45),
                zorder=3.0,
                kind="domino",
            )
        )

    for i in range(3):
        bodies.append(
            box(
                f"crate{i}",
                crate_s,
                crate_s,
                crate_cx,
                0.5 * crate_s + i * (crate_s + gap) + gap,
                mu=0.6,
                density=480.0,
                color="#c98a4b",
                edge="#e8bd8b",
                zorder=3.5,
                kind="crate",
            )
        )

    w = World(bodies, name="cascade")
    return w


SCENES = {
    "slide": slide,
    "incline": incline,
    "rolling": rolling,
    "stack": stack,
    "tumble": tumble,
    "cascade": cascade,
}
