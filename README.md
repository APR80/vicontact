# vicontact

**V**ariational **i**ntegrators for planar multibody systems with frictional **contact**.

`vicontact` simulates 2D convex rigid bodies (boxes, polygons, n-gons) that collide, slide, roll and stack under gravity. Each timestep is a small nonlinear program solved with **IPOPT** (via **CasADi**): a discrete Euler-Lagrange equation for the dynamics, plus complementarity conditions for non-penetration and Coulomb friction. Results can be rendered to a polished MP4 with a tracking camera, energy plot and contact-impulse overlay.

## Highlights

- **Variational integrator.** A midpoint discrete Lagrangian `Ld(q1, q2) = h L((q1+q2)/2, (q2-q1)/h)` is differentiated symbolically, and every step enforces the forced discrete Euler-Lagrange equation
  `D2 Ld(q[k-1], q[k]) + D1 Ld(q[k], q[k+1]) + h * sum_c J_c^T lambda_c = 0`.
- **Stewart-Trinkle friction.** Coulomb friction in linear complementarity form (no `sign()` smoothing), which behaves much better with many simultaneous contacts.
- **MPCC relaxation.** Complementarity products are minimised as the objective while all bounds stay hard constraints, so every step remains feasible.
- **SAT collision detection** outside the optimiser: up to two contact points per polygon pair, with *speculative* contacts (within a margin) to prevent tunnelling.
- **Careful non-dimensionalisation** of positions, impulses and residuals so one tolerance is meaningful across scenes and timesteps.
- **Solver caching per contact topology** plus warm starts from the previous step's impulses, with fallback starting points when a solve is rejected.
- **Guarded step acceptance.** A step is never integrated from an infeasible or unconverged solve — IPOPT can report the per-step NLP infeasible even when a solution provably exists, and a small complementarity residual says nothing on its own.
- **Gentle penetration recovery** instead of a one-step correction that injects energy.
- **Cinematic renderer**: energy-weighted tracking camera, ball trails, impulse markers, energy strip, locator minimap and a live HUD.

## Project layout

| File | What it does |
| --- | --- |
| `vicontact/geometry.py` | Convex polygon utilities, mass properties, SAT contact manifolds, AABBs |
| `vicontact/world.py` | `Body`, `World`, shape helpers (`box`, `polygon`, `ngon`) and broad/narrow phase contact search |
| `vicontact/stepper.py` | `ContactStepper`: builds and solves the per-step NLP with CasADi + IPOPT |
| `vicontact/sim.py` | `run()` time loop and the `Recording` trajectory container (save/load via pickle) |
| `vicontact/scenes.py` | Validation scenes and the `cascade` hero scene |
| `vicontact/render.py` | MP4 rendering (`render`) and contact-sheet PNGs (`snapshot`) |
| `validate.py` | Physics validation against closed-form answers |
| `check_impact.py` | Impact law vs. analytic angular-momentum transfer |
| `check_rolling.py` | Polygon-rolling energy loss vs. the corner-impact law |
| `make_video.py` | CLI: simulate a scene and render it to video |

## Requirements

- Python 3.10+
- `numpy`
- `casadi` (ships with IPOPT)
- `matplotlib`
- `ffmpeg` on your `PATH` (only needed for video rendering)

## Installation

```bash
git clone https://github.com/<you>/vicontact.git
cd vicontact
pip install -e .
```

Or just install the dependencies and work from the repository root:

```bash
pip install numpy casadi matplotlib
```

## Quick start

```bash
python make_video.py                 # cascade, reusing out/cascade.pkl
python make_video.py --fresh         # re-simulate from scratch
python make_video.py tumble --Tf 1.2
```

The same thing from Python:

```python
from vicontact import run, scenes, render

world = scenes.cascade()                 # ball -> ramp -> 16 dominoes -> crate stack
rec = run(world, Tf=5.0, h=1 / 360)      # simulate 5 s
rec.save("out/cascade.pkl")

render.snapshot(rec, "out/cascade.png", times=[0.0, 1.0, 2.5, 4.0])
render.render(rec, "out/cascade.mp4", title="Cascade",
              subtitle="Variational integrator + Stewart-Trinkle friction")
```

With the default `fps=60` and `stride=6`, a timestep of `h = 1/360` plays back in real time (`fps * stride * h = 1`).

Reload a saved run later without re-simulating:

```python
from vicontact import Recording, render
rec = Recording.load("out/cascade.pkl")
```
