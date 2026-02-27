"""Command-line interface for the optimal gokart path finder.

This module exposes a console script entry point that can run either the
Genetic Algorithm (``ga``) or Monte Carlo (``montecarlo``) path-finding
algorithms against a given track image.
"""

from __future__ import annotations

import argparse
import importlib.resources
from collections.abc import Sequence
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np

from .algorithms import GeneticAlgorithmPathFinder, MonteCarloPathFinder, PathFinder
from .models import Gokart, Track
from .models import Path as ModelPath
from .visualization import GokartDriveAnimation


def _bundled(filename: str) -> Path:
    """Return the path to a file bundled inside the package's data/ directory."""
    return Path(
        str(importlib.resources.files("optimal_gokart").joinpath(f"data/{filename}"))
    )


MAX_CLICK_POINTS = 500
CLICK_TIMEOUT_SEC = 150
ANIMATION_PAUSE_SEC = 0.3


def _add_global_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run with the bundled Brnik track image and pre-selected points. "
            "Mutually exclusive with --track-image / --points-file / --save-points."
        ),
    )
    parser.add_argument(
        "--track-image",
        type=Path,
        default=None,
        help="Path to the track image (required unless --demo is used).",
    )
    parser.add_argument(
        "--points-file",
        "--points_file",
        dest="points_file",
        type=Path,
        default=None,
        help="Path to a .npy file with pre-saved border points. If omitted, points are selected interactively.",
    )
    parser.add_argument(
        "--save-points",
        type=Path,
        default=None,
        help="Save interactively clicked points to this .npy file.",
    )
    parser.add_argument(
        "--ref-dist",
        type=float,
        default=20.0,
        help=(
            "Physical distance in metres between the first two clicked (reference) points. "
            "Used to compute the pixel-to-metre ratio. Default: 20.0."
        ),
    )
    parser.add_argument(
        "--points-on-line",
        type=int,
        default=5,
        help="Number of candidate positions per cross-section line of the track.",
    )
    parser.add_argument(
        "--gokart-mass",
        type=float,
        default=300.0,
        help="Gokart mass in kg (including driver).",
    )
    parser.add_argument(
        "--gokart-f-grip",
        type=float,
        default=400.0,
        help="Grip force in Newtons.",
    )
    parser.add_argument(
        "--gokart-f-motor",
        type=float,
        default=2000.0,
        help="Motor force in Newtons.",
    )
    parser.add_argument(
        "--gokart-k-drag",
        type=float,
        default=0.6125,
        help="Drag coefficient (air resistance proportional to v^2).",
    )


def _add_ga_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser_ga = subparsers.add_parser(
        "ga",
        help="Run the Genetic Algorithm path finder.",
    )
    parser_ga.set_defaults(algorithm="ga")

    parser_ga.add_argument(
        "--population-size",
        type=int,
        default=25,
        help="Population size for the genetic algorithm.",
    )
    parser_ga.add_argument(
        "--num-generations",
        type=int,
        default=200,
        help="Number of generations.",
    )
    parser_ga.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        help="Per-gene mutation probability.",
    )
    parser_ga.add_argument(
        "--elite-fraction",
        type=float,
        default=0.1,
        help="Fraction of the best individuals carried over unchanged.",
    )
    parser_ga.add_argument(
        "--tournament-size",
        type=int,
        default=10,
        help="Tournament size for parent selection.",
    )
    parser_ga.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Simulation time step in seconds for lap-time evaluation.",
    )
    parser_ga.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of parallel workers (1=serial, -1=all CPUs, -2=all minus one).",
    )
    parser_ga.add_argument(
        "--eval-num-interp-pts",
        type=int,
        default=200,
        help="Spline resolution used during fitness evaluation.",
    )


def _add_montecarlo_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser_mc = subparsers.add_parser(
        "montecarlo",
        help="Run the Monte Carlo path finder.",
    )
    parser_mc.set_defaults(algorithm="montecarlo")

    parser_mc.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of random paths to evaluate.",
    )
    parser_mc.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Simulation time step in seconds for lap-time evaluation.",
    )
    parser_mc.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers (1=serial, -1=all CPUs, -2=all minus one).",
    )
    parser_mc.add_argument(
        "--eval-num-interp-pts",
        type=int,
        default=200,
        help="Spline resolution used during evaluation.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optimal-gokart",
        description="Optimal gokart path finder with GA and Monte Carlo algorithms.",
    )
    _add_global_options(parser)

    subparsers = parser.add_subparsers(
        title="algorithms",
        dest="algorithm",
        required=True,
        help="Which optimisation algorithm to use.",
    )
    _add_ga_subcommand(subparsers)
    _add_montecarlo_subcommand(subparsers)

    return parser


def _load_track_image(track_image_path: Path) -> np.ndarray:
    return imageio.imread(track_image_path)


def _is_valid_points(arr: np.ndarray) -> bool:
    """Return True if *arr* looks like a usable (N, 2) border-points array."""
    return bool(arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3)


def _load_or_collect_points(
    track_image_arr: np.ndarray,
    points_file: Path | None,
    save_points: Path | None,
) -> np.ndarray:
    if points_file is not None and points_file.is_file():
        pts: np.ndarray = np.load(points_file)
        if _is_valid_points(pts):
            return pts
        print(
            f"Warning: '{points_file}' does not contain valid border points "
            f"(shape {pts.shape}). Falling back to interactive selection."
        )

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.imshow(track_image_arr)
    plt.title("Click border points. Middle-click (or wheel) when done.")

    points: np.ndarray = np.array(
        plt.ginput(MAX_CLICK_POINTS, timeout=CLICK_TIMEOUT_SEC),
        dtype=float,
    )
    plt.close(fig)

    if not _is_valid_points(points):
        raise ValueError(
            f"Too few points selected ({len(points)}). "
            "At least 3 points are required (2 for scale + at least 1 border pair)."
        )

    dest = save_points or points_file
    if dest is not None:
        np.save(dest, points)

    return points


def _prepare_track_and_gokart(
    points: np.ndarray,
    points_on_line: int,
    ref_dist_m: float,
    gokart_mass: float,
    gokart_f_grip: float,
    gokart_f_motor: float,
    gokart_k_drag: float,
) -> tuple[Track, Gokart]:
    # first two points define the reference distance (scale)
    unit_pts = points[:2, :]
    pix_per_m = float(np.linalg.norm(unit_pts[0, :] - unit_pts[1, :]) / ref_dist_m)

    track = Track(
        points[2:, :], points_on_line=points_on_line, pix_to_m_ratio=pix_per_m
    )
    gokart = Gokart(
        mass=gokart_mass,
        f_grip=gokart_f_grip,
        f_motor=gokart_f_motor,
        k_drag=gokart_k_drag,
    )

    return track, gokart


def _build_finder(args: argparse.Namespace) -> PathFinder:
    common_kwargs = {
        "dt": args.dt,
        "n_jobs": args.n_jobs,
        "eval_num_interp_pts": args.eval_num_interp_pts,
    }

    if args.algorithm == "ga":
        return GeneticAlgorithmPathFinder(
            population_size=args.population_size,
            num_generations=args.num_generations,
            mutation_rate=args.mutation_rate,
            elite_fraction=args.elite_fraction,
            tournament_size=args.tournament_size,
            **common_kwargs,
        )

    if args.algorithm == "montecarlo":
        return MonteCarloPathFinder(
            num_iterations=args.num_iterations,
            **common_kwargs,
        )

    raise ValueError(f"Unknown algorithm: {args.algorithm}")


def _run(args: argparse.Namespace) -> None:
    if args.demo:
        track_image_path: Path = _bundled("brnik-track-snip.PNG")
        points_file: Path | None = _bundled("points.npy")
        save_points: Path | None = None
    else:
        track_image_path = args.track_image  # guaranteed non-None by main()
        points_file = args.points_file  # None → interactive selection
        save_points = args.save_points

    track_image_arr = _load_track_image(track_image_path)

    points = _load_or_collect_points(
        track_image_arr=track_image_arr,
        points_file=points_file,
        save_points=save_points,
    )

    track, gokart = _prepare_track_and_gokart(
        points=points,
        points_on_line=args.points_on_line,
        ref_dist_m=args.ref_dist,
        gokart_mass=args.gokart_mass,
        gokart_f_grip=args.gokart_f_grip,
        gokart_f_motor=args.gokart_f_motor,
        gokart_k_drag=args.gokart_k_drag,
    )

    all_pts = np.array(track.interpolated_track_points_mat)
    all_pts = all_pts.reshape([all_pts.shape[0] * track.points_on_line, 2])

    plt.ion()
    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.show(block=False)

    def on_new_best(best_path: ModelPath, best_time: float, generation: int) -> None:
        interpolated_path = best_path.get_interpolated_path(metric=False)

        axes.cla()
        axes.imshow(track_image_arr)
        axes.plot(all_pts[:, 0], all_pts[:, 1], "r+")
        axes.plot(interpolated_path[:, 0], interpolated_path[:, 1])

        plt.draw()
        plt.pause(ANIMATION_PAUSE_SEC)

        print(
            f"Generation {generation}: new best path found! "
            f"Duration of driving: {best_time:.3f} seconds"
        )

    finder = _build_finder(args)
    opt_path = finder.find_optimal_path(track, gokart, progress_callback=on_new_best)

    animation = GokartDriveAnimation(track_image_arr, opt_path, gokart)
    animation.show()


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.demo:
        if args.track_image or args.points_file or args.save_points:
            parser.error(
                "--demo cannot be combined with --track-image, --points-file, or --save-points."
            )
    else:
        if args.track_image is None:
            parser.error("--track-image is required unless --demo is used.")

    _run(args)


if __name__ == "__main__":
    main()
