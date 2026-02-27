"""Command-line interface for the optimal gokart path finder.

This module exposes a console script entry point that can run either the
Genetic Algorithm (``ga``) or Monte Carlo (``montecarlo``) path-finding
algorithms against a given track image.
"""

from __future__ import annotations

import argparse
import importlib.resources
from pathlib import Path
from typing import Sequence

import imageio
import matplotlib.pyplot as plt
import numpy as np

from .algorithms import GeneticAlgorithmPathFinder, MonteCarloPathFinder, PathFinder
from .models import Gokart, Track
from .visualization import GokartDriveAnimation


def _bundled(filename: str) -> Path:
    """Return the path to a file bundled inside the package's data/ directory."""
    return Path(str(importlib.resources.files("optimal_gokart").joinpath(f"data/{filename}")))

MAX_CLICK_POINTS = 500
CLICK_TIMEOUT_SEC = 150
ANIMATION_PAUSE_SEC = 0.3


def _add_global_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--track-image",
        type=Path,
        default=None,
        help="Path to the track image. Defaults to the bundled Brnik track image.",
    )
    parser.add_argument(
        "--points-file",
        type=Path,
        default=None,
        help="Path to a .npy file with pre-saved border points. If omitted, points are selected interactively.",
    )
    parser.add_argument(
        "--save-points",
        type=Path,
        default=None,
        help="If provided and interactive selection is used, save the clicked points to this .npy file.",
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
        default=200.0,
        help="Gokart mass in kg (including driver).",
    )
    parser.add_argument(
        "--gokart-f-grip",
        type=float,
        default=200.0,
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


def _add_ga_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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


def _load_or_collect_points(
    track_image_arr: np.ndarray,
    points_file: Path,
    save_points: Path | None,
) -> np.ndarray:
    if points_file.is_file():
        return np.load(points_file)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.imshow(track_image_arr)

    points = np.array(
        plt.ginput(MAX_CLICK_POINTS, timeout=CLICK_TIMEOUT_SEC),
        dtype=float,
    )

    if save_points is not None:
        np.save(save_points, points)
    else:
        np.save(points_file, points)

    plt.close(fig)
    return points


def _prepare_track_and_gokart(
    points: np.ndarray,
    points_on_line: int,
    gokart_mass: float,
    gokart_f_grip: float,
    gokart_f_motor: float,
    gokart_k_drag: float,
) -> tuple[Track, Gokart, float]:
    fig_pts = plt.figure()
    axes_pts = fig_pts.add_subplot(111)

    # first two points define the reference distance (scale)
    unit_pts = points[:2, :]
    ref_dist_m = 20.0
    pix_per_m = float(np.linalg.norm(unit_pts[0, :] - unit_pts[1, :]) / ref_dist_m)

    track = Track(points[2:, :], points_on_line=points_on_line, pix_to_m_ratio=pix_per_m)
    gokart = Gokart(
        mass=gokart_mass,
        f_grip=gokart_f_grip,
        f_motor=gokart_f_motor,
        k_drag=gokart_k_drag,
    )

    return track, gokart, pix_per_m


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
    track_image_path: Path = args.track_image or _bundled("brnik-track-snip.PNG")
    # Explicit --points-file takes precedence; otherwise use the bundled default.
    # When doing interactive selection and no explicit path is given, save to the
    # current working directory instead of the (possibly read-only) package data dir.
    bundled_points = _bundled("points.npy")
    points_file: Path = args.points_file or bundled_points
    save_points: Path | None = args.save_points or (
        Path("points.npy") if points_file == bundled_points else None
    )

    track_image_arr = _load_track_image(track_image_path)

    points = _load_or_collect_points(
        track_image_arr=track_image_arr,
        points_file=points_file,
        save_points=save_points,
    )

    track, gokart, _ = _prepare_track_and_gokart(
        points=points,
        points_on_line=args.points_on_line,
        gokart_mass=args.gokart_mass,
        gokart_f_grip=args.gokart_f_grip,
        gokart_f_motor=args.gokart_f_motor,
        gokart_k_drag=args.gokart_k_drag,
    )

    all_pts = np.array(track.interpolated_track_points_mat)
    all_pts = all_pts.reshape([all_pts.shape[0] * track.points_on_line, 2])

    fig = plt.figure()
    axes = fig.add_subplot(111)

    def on_new_best(best_path, best_time, generation):
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
    _run(args)


if __name__ == "__main__":
    main()

