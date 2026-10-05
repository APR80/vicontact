"""Simulate a scene and render it to video."""

from __future__ import annotations
import argparse
from pathlib import Path
from vicontact import render, scenes, sim

OUT = Path("out")

# Per-scene defaults: (Tf, h, stride, title, subtitle)
PRESETS = {
    "cascade": (
        5.70,
        1 / 480,
        6,
        "Variational integrator with frictional contact",
        "Ball → 16 dominoes → 3-crate tower  ·  Stewart–Trinkle friction, "
        "one IPOPT solve per step  ·  h = 1/480 s, 0.75× speed",
    ),
    "tumble": (1.60, 1 / 600, 8, "Brick tumble", "Corner impacts, slide, settle"),
    "stack": (
        1.50,
        1 / 600,
        8,
        "Crate stack",
        "Four crates that must simply stand still",
    ),
    "rolling": (
        0.90,
        1 / 1500,
        20,
        "Polygon on a ramp",
        "48-gon rolling under Coulomb friction",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="cascade", choices=sorted(PRESETS))
    ap.add_argument("--Tf", type=float, default=None)
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--fresh", action="store_true", help="ignore the cached recording")
    ap.add_argument(
        "--no-camera", action="store_true", help="fixed wide shot instead of tracking"
    )
    args = ap.parse_args()

    Tf, h, stride, title, subtitle = PRESETS[args.scene]
    Tf = args.Tf if args.Tf is not None else Tf
    h = args.h if args.h is not None else h
    stride = args.stride if args.stride is not None else stride

    pkl = OUT / f"{args.scene}.pkl"
    if pkl.exists() and not args.fresh and args.Tf is None and args.h is None:
        print(f"loading {pkl}")
        rec = sim.Recording.load(pkl)
    else:
        print(f"simulating {args.scene}: Tf={Tf}s  h=1/{round(1 / h)}")
        rec = sim.run(scenes.SCENES[args.scene](), Tf=Tf, h=h)
        rec.save(pkl)

    m = rec.meta
    print(
        f"  steps {len(rec.poses)}  rejected {m['rejected']}  "
        f"worst constraint violation {m['worst_feas']:.2e}  worst comp {m['worst_comp']:.2e}"
    )

    mp4 = OUT / f"{args.scene}.mp4"
    n = len(range(0, len(rec.poses), stride))
    print(f"rendering {n} frames at {args.fps} fps -> {n / args.fps:.1f}s of video")
    render.render(
        rec,
        mp4,
        fps=args.fps,
        stride=stride,
        dpi=args.dpi,
        title=title,
        subtitle=subtitle,
        camera=not args.no_camera,
    )
    print(f"wrote {mp4}  ({mp4.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
