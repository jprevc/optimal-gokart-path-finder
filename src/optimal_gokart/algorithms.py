"""Path-finding algorithms for optimal gokart paths."""

import os
import random
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Optional

from .models import Gokart, Path, Track

ProgressCallback = Callable[[Path, float, int], None]
"""Callback invoked when a new best path is found.

Arguments: ``(best_path, best_time, iteration)`` where *iteration* is the
zero-based index of the current generation (GA) or trial (Monte Carlo).
"""


def _resolve_n_jobs(n_jobs: int) -> int:
    """Return the actual worker count for a given ``n_jobs`` value.

    Follows the same convention as scikit-learn:
    ``1`` → serial, ``-1`` → all CPUs, ``-2`` → all CPUs minus one, etc.
    """
    if n_jobs == 1:
        return 1
    cpu_count = os.cpu_count() or 1
    if n_jobs < 0:
        return max(1, cpu_count + n_jobs + 1)
    return min(n_jobs, cpu_count)


class PathFinder(ABC):
    """Abstract base class for path-finding algorithms.

    Subclasses implement :meth:`find_optimal_path` to search for the path on
    *track* that minimises the lap time for *gokart*.
    """

    @abstractmethod
    def find_optimal_path(
        self,
        track: Track,
        gokart: Gokart,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        """Return the best :class:`Path` found on *track* for *gokart*.

        :param track: Track on which to find the path.
        :param gokart: Gokart whose physics are used to evaluate lap time.
        :param progress_callback: Optional callable invoked whenever a new
            best path is found. Receives ``(best_path, best_time, iteration)``.
        :return: The best :class:`Path` found.
        """


# ---------------------------------------------------------------------------
# Module-level helpers (must be top-level so ProcessPoolExecutor can pickle them)
# ---------------------------------------------------------------------------

def _chromosome_to_path(
    chromosome: list[int],
    track: Track,
    smooth_coef: float = 0,
    num_interp_points: int = 1000,
) -> Path:
    path_point_mat = track._get_path_point_mat(chromosome)
    return Path(
        path_point_mat,
        smooth_coef=smooth_coef,
        num_interp_pts=num_interp_points,
        pix_to_m_ratio=track.pix_to_m_ratio,
    )


def _evaluate(
    chromosome: list[int],
    track: Track,
    gokart: Gokart,
    dt: float,
    num_interp_pts: int = 1000,
) -> float:
    path = _chromosome_to_path(chromosome, track, num_interp_points=num_interp_pts)
    tvec, _ = path.get_time_track(gokart, dt)
    return float(tvec[-1])


def _evaluate_batch(
    chromosomes: list[list[int]],
    track: Track,
    gokart: Gokart,
    dt: float,
    num_interp_pts: int,
    n_workers: int,
) -> list[float]:
    """Evaluate a batch of chromosomes, in parallel when *n_workers* > 1."""
    if n_workers == 1:
        return [_evaluate(c, track, gokart, dt, num_interp_pts) for c in chromosomes]

    n = len(chromosomes)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        return list(
            executor.map(
                _evaluate,
                chromosomes,
                [track] * n,
                [gokart] * n,
                [dt] * n,
                [num_interp_pts] * n,
            )
        )


def _tournament_select(
    population: list[list[int]],
    fitnesses: list[float],
    tournament_size: int,
) -> list[int]:
    contestants = random.sample(range(len(population)), tournament_size)
    winner = min(contestants, key=lambda i: fitnesses[i])
    return population[winner]


def _uniform_crossover(parent_a: list[int], parent_b: list[int]) -> list[int]:
    return [a if random.random() < 0.5 else b for a, b in zip(parent_a, parent_b)]


def _mutate(chromosome: list[int], points_on_line: int, mutation_rate: float) -> list[int]:
    return [
        random.randint(0, points_on_line - 1) if random.random() < mutation_rate else gene
        for gene in chromosome
    ]


# ---------------------------------------------------------------------------
# Concrete finders
# ---------------------------------------------------------------------------

class MonteCarloPathFinder(PathFinder):
    """Path finder that samples random paths and keeps the fastest one.

    :param num_iterations: Number of random paths to evaluate.
    :param dt: Simulation time step in seconds for lap-time evaluation.
    :param n_jobs: Number of parallel workers. ``1`` = serial,
        ``-1`` = all CPUs, ``-2`` = all CPUs minus one (sklearn convention).
    :param eval_num_interp_pts: Spline resolution used during evaluation.
        Lower values are faster; the final returned path is always built at
        full resolution (1000 points).
    """

    def __init__(
        self,
        num_iterations: int = 1000,
        dt: float = 0.1,
        n_jobs: int = 1,
        eval_num_interp_pts: int = 200,
    ) -> None:
        self.num_iterations = num_iterations
        self.dt = dt
        self.n_jobs = n_jobs
        self.eval_num_interp_pts = eval_num_interp_pts

    def find_optimal_path(
        self,
        track: Track,
        gokart: Gokart,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        n_workers = _resolve_n_jobs(self.n_jobs)
        points_on_line = track.points_on_line
        num_lines = track.num_lines

        # Generate all random chromosomes up front so they can be evaluated in
        # parallel.  Progress is reported after evaluating each chromosome (in
        # order), which preserves the same semantics as the serial version.
        chromosomes = [
            [random.randint(0, points_on_line - 1) for _ in range(num_lines)]
            for _ in range(self.num_iterations)
        ]

        fitnesses = _evaluate_batch(
            chromosomes, track, gokart, self.dt, self.eval_num_interp_pts, n_workers
        )

        best_time: float = float("inf")
        best_path: Optional[Path] = None

        for i, (chrom, t) in enumerate(zip(chromosomes, fitnesses)):
            if t < best_time:
                best_time = t
                best_path = _chromosome_to_path(chrom, track)  # full resolution

                if progress_callback is not None:
                    progress_callback(best_path, best_time, i)

        if best_path is None:
            best_path = track.get_random_path()

        return best_path


class GeneticAlgorithmPathFinder(PathFinder):
    """Path finder based on a genetic algorithm.

    :param population_size: Number of individuals in each generation.
    :param num_generations: Number of generations to evolve.
    :param mutation_rate: Per-gene probability of random mutation.
    :param elite_fraction: Fraction of the best individuals carried over unchanged.
    :param tournament_size: Number of contestants in each tournament selection.
    :param dt: Simulation time step in seconds for lap-time evaluation.
    :param n_jobs: Number of parallel workers for fitness evaluation.
        ``1`` = serial, ``-1`` = all CPUs, ``-2`` = all CPUs minus one.
    :param eval_num_interp_pts: Spline resolution used during fitness evaluation.
        Lower values are faster; the final returned path and any path passed to
        *progress_callback* are always built at full resolution (1000 points).
    """

    def __init__(
        self,
        population_size: int = 50,
        num_generations: int = 50,
        mutation_rate: float = 0.1,
        elite_fraction: float = 0.1,
        tournament_size: int = 5,
        dt: float = 0.1,
        n_jobs: int = 1,
        eval_num_interp_pts: int = 200,
    ) -> None:
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size
        self.dt = dt
        self.n_jobs = n_jobs
        self.eval_num_interp_pts = eval_num_interp_pts

    def find_optimal_path(
        self,
        track: Track,
        gokart: Gokart,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        n_workers = _resolve_n_jobs(self.n_jobs)
        num_lines = track.num_lines
        points_on_line = track.points_on_line
        num_elites = max(1, int(self.population_size * self.elite_fraction))

        population = [
            [random.randint(0, points_on_line - 1) for _ in range(num_lines)]
            for _ in range(self.population_size)
        ]

        best_chromosome: list[int] = population[0]
        best_time: float = float("inf")
        best_path: Optional[Path] = None

        for generation in range(self.num_generations):
            fitnesses = _evaluate_batch(
                population, track, gokart, self.dt, self.eval_num_interp_pts, n_workers
            )

            gen_best_idx = min(range(self.population_size), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] < best_time:
                best_time = fitnesses[gen_best_idx]
                best_chromosome = population[gen_best_idx]
                best_path = _chromosome_to_path(best_chromosome, track)  # full resolution

                if progress_callback is not None:
                    progress_callback(best_path, best_time, generation)

            sorted_indices = sorted(range(self.population_size), key=lambda i: fitnesses[i])
            elites = [population[i] for i in sorted_indices[:num_elites]]

            new_population: list[list[int]] = list(elites)
            while len(new_population) < self.population_size:
                parent_a = _tournament_select(population, fitnesses, self.tournament_size)
                parent_b = _tournament_select(population, fitnesses, self.tournament_size)
                child = _uniform_crossover(parent_a, parent_b)
                child = _mutate(child, points_on_line, self.mutation_rate)
                new_population.append(child)

            population = new_population

        if best_path is None:
            best_path = _chromosome_to_path(best_chromosome, track)

        return best_path
