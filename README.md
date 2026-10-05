# vicontact

**V**ariational **i**ntegrators for planar multibody systems with frictional **contact**.

`vicontact` simulates 2D convex rigid bodies (boxes, polygons, n-gons) that collide, slide, roll and stack under gravity. Each timestep is a small nonlinear program solved with **IPOPT** (via **CasADi**): a discrete Euler-Lagrange equation for the dynamics, plus complementarity conditions for non-penetration and Coulomb friction. Results can be rendered to a polished MP4 with a tracking camera, energy plot and contact-impulse overlay.

<p align="center">
  <a href="https://youtu.be/7Z9DUAu6hnU">
    <img src="https://img.youtube.com/vi/7Z9DUAu6hnU/maxresdefault.jpg" alt="Project Demo" width="75%">
    <br>
    <sub><b>Project Demo</b></sub>
  </a>
</p>
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
## Validation

```bash
python validate.py        # exits non-zero if any check fails
```

Six checks, each against a closed-form answer:

| Check | Asserts |
| --- | --- |
| `slide` | Sliding block decelerates at `mu g`, within 2% |
| `incline` | Block on a 30° ramp accelerates at `g (sin a - mu cos a)`, within 2% |
| `rolling` | `a(n)` over n = 16, 24, 48, 96 fits `a_inf - c/n`; the extrapolation to n → ∞ recovers the ideal disc `(2/3) g sin a` within 1%, with 1/n fit residual < 2e-2 |
| `stack` | A 4-crate tower drifts < 2 mm, tilts < 2e-3 rad, settles < 5 mm |
| `energy` | Free-flight energy spread < 1e-3 relative, before any contact |
| `solver` | `stack(4)` and `tumble` integrate with zero rejected steps and worst constraint violation < 1e-7 |

Two focused diagnostics go further:

```bash
python check_impact.py    # exits non-zero if worst relative error > 5%
python check_rolling.py
```

## Scenes

All scenes live in `scenes.SCENES`:

| Scene | Setup | What to check |
| --- | --- | --- |
| `slide` | Block launched along flat ground | Decelerates at `mu * g` |
| `incline` | Block released on a ramp | Accelerates at `g (sin a - mu cos a)` |
| `rolling` | Regular n-gon on a ramp (48 sides by default) | Acceleration sits below the ideal disc by an O(1/n) pivot loss |
| `stack` | Tower of crates | Stays perfectly still |
| `tumble` | Spinning brick thrown at the floor | Corner impacts, slide, settle |
| `cascade` | Ball rolls down a ramp, topples a domino run, last domino smashes a crate stack | The hero shot |

## Building your own scene

```python
import numpy as np
from vicontact import World, box, ngon, run

floor = box("floor", 10.0, 0.5, 0.0, -0.25, static=True, mu=0.6)
crate = box("crate", 0.4, 0.4, 0.0, 1.0, angle=0.3, mu=0.5, v0=np.array([1.0, 0.0, 2.0]))
wheel = ngon("wheel", 0.2, 32, -1.0, 0.21, mu=0.8, kind="ball")

world = World([floor, crate, wheel], name="my_scene")
rec = run(world, Tf=3.0, h=1 / 360)
```

Bodies must be **convex**. Poses are `(x, y, theta)`; `v0` is the initial `(vx, vy, omega)`. Static bodies have no degrees of freedom and only take part in collisions. Mass and inertia come from `density` and the polygon area; friction between two bodies uses `sqrt(mu_a * mu_b)`.

## Tuning

`run()` forwards extra keyword arguments to `ContactStepper`:

| Argument | Default | Meaning |
| --- | --- | --- |
| `margin` | `0.02` | Speculative contact distance |
| `pen_frac`, `pen_rate` | `0.3`, `0.2` | How quickly existing penetration is recovered |
| `reg` | `1e-7` | Tiny regulariser on the contact impulses |
| `max_iter` | `400` | IPOPT iteration cap |
| `tol` | `1e-9` | IPOPT tolerance |
| `comp_tol` | `1e-5` | Max complementarity residual for an accepted step |
| `feas_tol` | `1e-8` | Max constraint violation for an accepted step |
| `cache_limit` | `800` | Max cached solvers (one per contact topology) |

## Diagnostics

Every `Recording` stores per-step kinetic and potential energy, contact count, complementarity / DEL / feasibility residuals, a rejected-step flag, plus contact points and normal/tangential forces. `rec.meta` summarises the run: wall time, solvers built, retries, worst residuals and the first rejected steps.
