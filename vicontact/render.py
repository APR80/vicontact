"""Video rendering for recorded simulations."""

from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle

from .geometry import transform
from .scenes import BG
from .sim import Recording

Array = np.ndarray

HUD = "#8fa3c8"
HUD_BRIGHT = "#e6edf9"
IMPULSE = "#ffd166"
PANEL = "#0d1322"
PANEL_EDGE = "#1e2942"


def _limits(rec: Recording, pad: float = 0.35) -> tuple[float, float, float, float]:
    lo = np.array([np.inf, np.inf])
    hi = -lo
    for k in range(0, len(rec.poses), max(1, len(rec.poses) // 120)):
        for b, meta in enumerate(rec.meta["bodies"]):
            v = transform(meta["verts"], rec.poses[k, b])
            lo = np.minimum(lo, v.min(axis=0))
            hi = np.maximum(hi, v.max(axis=0))
    return (
        float(lo[0]) - pad,
        float(hi[0]) + pad,
        float(lo[1]) - pad,
        float(hi[1]) + pad,
    )


def _dynamic_bounds(rec: Recording) -> tuple[float, float, float, float]:
    """Bounds of the moving bodies only"""
    dyn = [b for b, m in enumerate(rec.meta["bodies"]) if not m["static"]]
    lo = np.array([np.inf, np.inf])
    hi = -lo
    for k in range(0, len(rec.poses), max(1, len(rec.poses) // 200)):
        for b in dyn:
            v = transform(rec.meta["bodies"][b]["verts"], rec.poses[k, b])
            lo = np.minimum(lo, v.min(axis=0))
            hi = np.maximum(hi, v.max(axis=0))
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _smooth(x: Array, sigma: float) -> Array:
    """Zero-phase Gaussian blur along a signal, with edge padding."""
    x = np.asarray(x, dtype=float)
    if sigma <= 0.5 or x.size < 3:
        return x
    r = int(np.ceil(3.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def _fill_gaps(x: Array) -> Array:
    """Hold the last known value across NaNs (and the first one backwards)."""
    x = np.asarray(x, dtype=float)
    good = np.flatnonzero(~np.isnan(x))
    if good.size == 0:
        return np.zeros_like(x)
    idx = np.clip(
        np.searchsorted(good, np.arange(x.size), side="right") - 1, 0, good.size - 1
    )
    return x[good[idx]]


def _camera_path(
    rec: Recording,
    frames: list[int],
    aspect: float,
    bounds: tuple[float, float, float, float],
    ground: float,
    min_w: float,
    max_w: float,
    pad: float,
    e_frac: float,
    sigma_pan: float,
    sigma_zoom: float,
    hold_open: float,
    floor_frac: float,
) -> tuple[Array, Array, Array, Array]:
    """Follow the action: returns per-frame (cx, cy, width, height)."""
    meta = rec.meta["bodies"]
    dyn = np.array([b for b, m in enumerate(meta) if not m["static"]], dtype=int)
    nf = len(frames)
    cx = np.full(nf, np.nan)
    wid = np.full(nf, float(max_w))

    if dyn.size:
        rad = np.array(
            [float(np.linalg.norm(meta[b]["verts"], axis=1).max()) for b in dyn]
        )
        mass = np.array([float(meta[b]["mass"]) for b in dyn])
        vel = np.gradient(rec.poses[:, dyn, :], rec.h, axis=0)
        tip = np.hypot(vel[:, :, 0], vel[:, :, 1]) + rad * np.abs(vel[:, :, 2])
        score = mass * tip**2
        floor = e_frac * float(score.max())

        for i, k in enumerate(frames):
            sc = score[k]
            if sc.max() <= floor:
                continue
            keep = sc >= max(floor, 0.25 * sc.max())
            act, wgt = dyn[keep], sc[keep]
            lo, hi = np.inf, -np.inf
            for b in act:
                vx = transform(meta[b]["verts"], rec.poses[k, b])[:, 0]
                lo, hi = min(lo, float(vx.min())), max(hi, float(vx.max()))
            cx[i] = float(np.dot(wgt, rec.poses[k, act, 0]) / wgt.sum())
            wid[i] = float(np.clip((hi - lo) + 2.0 * pad, min_w, max_w))

    dt = (frames[1] - frames[0]) * rec.h if nf > 1 else rec.h
    t = np.asarray(frames, dtype=float) * rec.h
    wid[t < hold_open] = max_w  # establishing shot
    cx = _smooth(_fill_gaps(cx), sigma_pan / dt)
    wid = _smooth(wid, sigma_zoom / dt)

    span = bounds[1] - bounds[0]
    wid = np.minimum(wid, span)
    half = 0.5 * wid
    cx = np.clip(cx, bounds[0] + half, bounds[1] - half)

    hgt = wid / aspect
    cy = ground + (0.5 - floor_frac) * hgt
    return cx, cy, wid, hgt


def render(
    rec: Recording,
    path: str | Path,
    fps: int = 60,
    stride: int = 6,
    dpi: int = 120,
    title: str | None = None,
    subtitle: str | None = None,
    trail_frames: int = 55,
    show_impulses: bool = True,
    figsize: tuple[float, float] = (16.0, 9.0),
    bitrate: int = -1,
    camera: bool = True,
    min_w: float = 3.9,
    max_w: float = 6.6,
    cam_pad: float = 0.85,
    e_frac: float = 0.006,
    locator: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bodies = rec.meta["bodies"]
    frames = list(range(0, len(rec.poses), stride))

    sx0, sx1, sy0, sy1 = _limits(rec)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG)

    box = (0.0, 0.145, 1.0, 0.66)
    ax = fig.add_axes(box)
    ax.set_facecolor(BG)
    ax.axis("off")
    aspect = (figsize[0] * box[2]) / (figsize[1] * box[3])

    dx0, dx1, dy0, _dy1 = _dynamic_bounds(rec)
    if camera:
        cx, cy, cw, ch = _camera_path(
            rec,
            frames,
            aspect=aspect,
            bounds=(max(sx0, dx0 - 0.55), min(sx1, dx1 + 0.55), sy0, sy1),
            ground=dy0,
            min_w=min_w,
            max_w=max_w,
            pad=cam_pad,
            e_frac=e_frac,
            sigma_pan=0.28,
            sigma_zoom=0.50,
            hold_open=0.8,
            floor_frac=0.07,
        )
    else:
        w = sx1 - sx0
        cx = np.full(len(frames), 0.5 * (sx0 + sx1))
        cw = np.full(len(frames), w)
        ch = cw / aspect
        cy = np.full(len(frames), 0.5 * (sy0 + sy1))
    ax.set_xlim(cx[0] - 0.5 * cw[0], cx[0] + 0.5 * cw[0])
    ax.set_ylim(cy[0] - 0.5 * ch[0], cy[0] + 0.5 * ch[0])

    grad = np.linspace(1.0, 0.0, 256).reshape(-1, 1)
    wash = ax.imshow(
        grad,
        extent=(
            cx[0] - 0.5 * cw[0],
            cx[0] + 0.5 * cw[0],
            cy[0] - 0.5 * ch[0],
            cy[0] + 0.5 * ch[0],
        ),
        aspect="auto",
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "bgwash", ["#0e1526", BG]
        ),
        zorder=0,
        interpolation="bilinear",
    )

    patches: list[MplPolygon | None] = []
    for b, meta in enumerate(bodies):
        v = transform(meta["verts"], rec.poses[0, b])
        p = MplPolygon(
            v,
            closed=True,
            facecolor=meta["color"],
            edgecolor=meta["edge"],
            linewidth=1.6 if not meta["static"] else 1.2,
            alpha=1.0,
            zorder=meta["zorder"],
            joinstyle="round",
        )
        ax.add_patch(p)
        patches.append(None if meta["static"] else p)

    ball_idx = [b for b, m in enumerate(bodies) if m["kind"] == "ball"]
    trails = LineCollection([], colors=IMPULSE, linewidths=1.4, alpha=0.5, zorder=3.9)
    ax.add_collection(trails)

    impulse_scatter = ax.scatter(
        [], [], s=[], c=IMPULSE, alpha=0.85, zorder=9, edgecolors="none"
    )

    # ---- HUD ----------------------------------------------------------
    if title is None:
        title = rec.name
    fig.text(0.035, 0.955, title, color=HUD_BRIGHT, fontsize=25, weight="600", va="top")
    if subtitle:
        fig.text(0.035, 0.915, subtitle, color=HUD, fontsize=13.5, va="top")

    hud = fig.text(
        0.965,
        0.955,
        "",
        color=HUD,
        fontsize=13,
        va="top",
        ha="right",
        family="monospace",
        linespacing=1.7,
    )

    # --- locator map ---
    mini_patches: list[MplPolygon | None] = []
    view_rect = None
    if locator and camera:
        mbox = (0.035, 0.028, 0.345, 0.086)
        mini = fig.add_axes(mbox)
        mini.set_facecolor(PANEL)
        for s in mini.spines.values():
            s.set_color(PANEL_EDGE)
        mini.set_xticks([])
        mini.set_yticks([])
        mini.set_xlim(sx0, sx1)
        m_h = (sx1 - sx0) * (mbox[3] * figsize[1]) / (mbox[2] * figsize[0])
        mini.set_ylim(dy0 - 0.12 * m_h, dy0 - 0.12 * m_h + m_h)
        for b, meta in enumerate(bodies):
            p = MplPolygon(
                transform(meta["verts"], rec.poses[0, b]),
                closed=True,
                facecolor=meta["color"],
                edgecolor="none",
                lw=0.0,
                alpha=0.55 if meta["static"] else 1.0,
                zorder=meta["zorder"],
            )
            mini.add_patch(p)
            mini_patches.append(None if meta["static"] else p)
        view_rect = Rectangle(
            (cx[0] - 0.5 * cw[0], dy0 - 0.12 * m_h),
            cw[0],
            m_h,
            facecolor=HUD_BRIGHT,
            alpha=0.10,
            edgecolor=HUD_BRIGHT,
            lw=0.9,
            zorder=20,
        )
        mini.add_patch(view_rect)
        mini.text(
            0.006,
            0.9,
            "view",
            transform=mini.transAxes,
            color=HUD,
            fontsize=7.5,
            va="top",
            family="monospace",
        )
        strip_box = (0.42, 0.035, 0.545, 0.072)
    else:
        strip_box = (0.035, 0.035, 0.93, 0.072)

    # ---energy strip ---
    strip = fig.add_axes(strip_box)
    strip.set_facecolor(PANEL)
    for s in strip.spines.values():
        s.set_color(PANEL_EDGE)
    t = np.arange(len(rec.poses)) * rec.h
    total = rec.kinetic + rec.potential
    strip.plot(t, rec.kinetic, color="#22d3ee", lw=1.1, label="kinetic")
    strip.plot(t, total, color="#f472b6", lw=1.1, label="kinetic + potential")
    strip.set_xlim(t[0], t[-1])
    ymax = float(np.nanmax(total) * 1.1 + 1e-9)
    strip.set_ylim(0.0, ymax)
    strip.tick_params(colors=HUD, labelsize=9, length=2)
    strip.set_yticks([])
    strip.set_xlabel("time (s)", color=HUD, fontsize=9, labelpad=1)
    leg = strip.legend(
        loc="upper right",
        fontsize=8.5,
        frameon=False,
        ncol=2,
        labelcolor=HUD,
        handlelength=1.4,
    )
    for txt in leg.get_texts():
        txt.set_color(HUD)
    cursor = strip.axvline(0.0, color=HUD_BRIGHT, lw=1.2, alpha=0.8)

    fmax = max(
        (float(np.abs(f).max()) for f in rec.contact_fn if len(f)),
        default=1.0,
    )

    writer = FFMpegWriter(
        fps=fps,
        bitrate=bitrate,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "17", "-preset", "slow"],
    )

    with writer.saving(fig, str(path), dpi=dpi):
        for fi, k in enumerate(frames):
            for b, meta in enumerate(bodies):
                p = patches[b]
                if p is not None:
                    xy = transform(meta["verts"], rec.poses[k, b])
                    p.set_xy(xy)
                    mp = mini_patches[b] if mini_patches else None
                    if mp is not None:
                        mp.set_xy(xy)

            xlo, xhi = cx[fi] - 0.5 * cw[fi], cx[fi] + 0.5 * cw[fi]
            ylo, yhi = cy[fi] - 0.5 * ch[fi], cy[fi] + 0.5 * ch[fi]
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            wash.set_extent((xlo, xhi, ylo, yhi))
            if view_rect is not None:
                view_rect.set_x(xlo)
                view_rect.set_width(cw[fi])

            if ball_idx and trail_frames > 0:
                segs = []
                lo = max(0, k - trail_frames * stride)
                for b in ball_idx:
                    pts = rec.poses[lo : k + 1 : max(1, stride // 2), b, :2]
                    if len(pts) > 1:
                        segs.append(pts)
                trails.set_segments(segs)

            if show_impulses and len(rec.contact_points[k]):
                mag = np.abs(rec.contact_fn[k])
                keep = mag > 1e-3 * fmax
                pts = rec.contact_points[k][keep]
                mag = mag[keep]
                impulse_scatter.set_offsets(pts if len(pts) else np.zeros((0, 2)))
                impulse_scatter.set_sizes(18.0 + 320.0 * np.sqrt(mag / fmax))
            else:
                impulse_scatter.set_offsets(np.zeros((0, 2)))
                impulse_scatter.set_sizes(np.zeros(0))

            cursor.set_xdata([t[k], t[k]])
            hud.set_text(
                f"t        {t[k]:6.3f} s\n"
                f"contacts {rec.n_contacts[k]:6d}\n"
                f"energy   {total[k]:6.1f} J\n"
                f"compl.   {rec.comp_residual[k]:.1e}"
            )
            writer.grab_frame(facecolor=BG)

    plt.close(fig)
    return path


def snapshot(
    rec: Recording, path: str | Path, times: list[float], dpi: int = 110
) -> Path:
    """Contact-sheet PNG at the requested times: a fast visual sanity check."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bodies = rec.meta["bodies"]
    x0, x1, y0, y1 = _limits(rec)
    n = len(times)
    fig, axes = plt.subplots(n, 1, figsize=(13.0, 2.9 * n), dpi=dpi)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes)
    for ax, tt in zip(axes, times):
        k = min(int(round(tt / rec.h)), len(rec.poses) - 1)
        ax.set_facecolor(BG)
        for b, meta in enumerate(bodies):
            ax.add_patch(
                MplPolygon(
                    transform(meta["verts"], rec.poses[k, b]),
                    closed=True,
                    facecolor=meta["color"],
                    edgecolor=meta["edge"],
                    lw=1.2,
                    zorder=meta["zorder"],
                )
            )
        if len(rec.contact_points[k]):
            ax.scatter(*rec.contact_points[k].T, s=12, c=IMPULSE, zorder=9)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(
            0.005,
            0.94,
            f"t = {k * rec.h:.3f} s   contacts = {rec.n_contacts[k]}",
            transform=ax.transAxes,
            color=HUD,
            fontsize=10,
            family="monospace",
        )
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path
