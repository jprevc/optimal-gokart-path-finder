# Optimal Gokart Path Finder

A Monte Carlo–based tool that finds a near-optimal racing line for a go-kart on a user-defined track. The track is represented by border points; paths are interpolated with splines and evaluated using a simple physics model (acceleration, drag, and lateral grip).

---

## Description

The program:

1. Lets you define a track by clicking border points on a track image (or by loading saved points).
2. Discretizes the track into segments and samples waypoints along each segment.
3. Searches over path combinations via **random sampling (Monte Carlo)**.
4. For each candidate path: builds a **smooth spline**, computes curvature and maximum cornering speed, then **simulates** lap time with a go-kart model.
5. Keeps the path with the **shortest simulated lap time** and optionally animates the kart driving along it.

### Physics model (brief)

- **Longitudinal:** Motor force `F_motor` vs quadratic drag `F_drag = k_drag * v^2`. Acceleration: `a = (F_motor - F_drag) / m`.
- **Lateral:** Max cornering speed at each point is limited by grip: `v_max = sqrt(F_grip * R / m)`, where `R` is the local radius of curvature from the path spline. The simulation never exceeds this speed on corners.

---

## Screenshot / animation

![Track example](docs/brnik-track-snip.PNG)
![GA path finder](docs/ga-path-finder.gif)
![Gokart simulation](docs/gokart-simulation.gif)

---

## Installation

1. Clone the repository and enter the project directory:

   ```bash
   git clone <repo-url>
   cd optimal-gokart-path-finder
   ```

2. Install the package and dependencies with **uv**:

   ```bash 
   uv sync
   ```
---

## Usage

### Running the example

From the project root, run:

```bash
python examples/brnik_track_example.py
```

Place the track image at the project root: `brnik_track_snip.png` (or change the path in the script). The script also looks for `points.npy` at the project root.

### Interactive track definition

- If **`points.npy`** does not exist, the script opens the track image and asks you to **click points** along the track borders.
  - Click in order along one border, then the other, so that consecutive pairs of points define line segments across the track.
  - When finished, **click the middle mouse button (wheel)** to stop input. The points are saved to `points.npy`.
- The **first two clicked points** are used as a scale reference: they should correspond to a known distance on the track (e.g. 20 m). The script uses this to set the pixel-to-meter ratio.
- On the next run, if `points.npy` exists, the script loads it and skips the clicking step.

### What the script does

1. Loads or collects track border points and builds a `Track`.
2. Runs a **Monte Carlo search**: repeatedly samples random paths on the track, simulates lap time for each, and keeps the path with the **minimum lap time**.
3. Redraws the plot whenever a better path is found.
4. At the end, plays an **animation** of the go-kart driving the best path.

---

## Gokart parameters

You can tune the go-kart by passing these arguments to `Gokart(mass, f_grip, f_motor, k_drag)`:

| Parameter   | Meaning                         | Unit   | Example / note                          |
|------------|----------------------------------|--------|----------------------------------------|
| `mass`     | Total mass (kart + driver)       | kg     | e.g. 200                               |
| `f_grip`   | Lateral grip force (cornering)   | N      | Limits v_max in corners                 |
| `f_motor`  | Motor thrust (forward force)    | N      | e.g. 2000                              |
| `k_drag`   | Drag coefficient (F = k_drag * v^2) | N·s²/m² | e.g. 0.6125                    |

Higher `f_grip` allows higher cornering speed; higher `f_motor` and lower `k_drag` give higher straight-line speed.

---

## Algorithm overview

1. **Track representation**  
   Border points define line segments across the track. Each segment is discretized into `points_on_line` waypoints. A **path** is one waypoint chosen on each segment, in order.

2. **Monte Carlo path search**  
   Instead of enumerating all paths (exponential in the number of segments), the code **samples random paths** by picking a random waypoint on each segment. It runs a fixed number of iterations (e.g. 100) and keeps the path with the **shortest simulated lap time**.

3. **Spline interpolation**  
   The chosen waypoints are interpolated with a **smooth spline** (e.g. `scipy.interpolate.splprep` / `splev`). The spline gives:
   - interpolated positions,
   - derivatives for curvature,
   - and curvature-based **maximum cornering speed** at each point.

4. **Physics simulation**  
   Lap time is computed by stepping in time with a fixed `dt`:
   - Update speed: `v <- v + dt * a(v)`, with `a = (F_motor - k_drag * v^2) / m`.
   - Cap speed at the current point’s `v_max` (grip limit).
   - Advance position along the path based on \(v\) and path length.
   The simulation stops when the kart has completed one lap; total time is the lap time used for comparison.

5. **Visualization**  
   The best path is drawn on the track image, and `GokartDriveAnimation` replays the simulated positions as an animation.

---

## How it works (flowchart)

```text
┌─────────────────────────────────────────────────────────────────┐
│  Load track image & border points (click or load points.npy)     │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Build Track: segments → waypoints per segment                  │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Monte Carlo loop (e.g. N = 100 iterations)                     │
│    ├─ Sample random path (one waypoint per segment)             │
│    ├─ Path → spline interpolation → curvature → v_max(s)        │
│    ├─ Simulate lap: integrate speed with motor/drag + grip cap  │
│    └─ If lap time < best so far → save path, redraw             │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Show best path on image; animate go-kart along path            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project structure

```text
optimal-gokart-path-finder/
├── src/
│   └── optimal_gokart/
│       ├── __init__.py
│       ├── models.py          # Gokart, Path, Track
│       └── visualization.py  # GokartDriveAnimation
├── tests/
│   ├── test_models.py
│   └── test_optimal_path_classes.py
├── examples/
│   └── brnik_track_example.py
├── pyproject.toml
├── uv.lock
└── README.md
```
