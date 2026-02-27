"""Genetic algorithm for finding the optimal gokart path on a track."""

import random
from typing import Callable, Optional

from .models import Gokart, Path, Track


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


def _evaluate(chromosome: list[int], track: Track, gokart: Gokart, dt: float) -> float:
    path = _chromosome_to_path(chromosome, track)
    tvec, _ = path.get_time_track(gokart, dt)
    return float(tvec[-1])


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


def find_optimal_path_ga(
    track: Track,
    gokart: Gokart,
    population_size: int = 50,
    num_generations: int = 50,
    mutation_rate: float = 0.1,
    elite_fraction: float = 0.1,
    tournament_size: int = 5,
    dt: float = 0.1,
    progress_callback: Optional[Callable[[Path, float, int], None]] = None,
) -> Path:
    """Find the optimal gokart path using a genetic algorithm.

    :param track: Track on which to find the path.
    :param gokart: Gokart whose physics are used to evaluate lap time.
    :param population_size: Number of individuals in each generation.
    :param num_generations: Number of generations to evolve.
    :param mutation_rate: Per-gene probability of random mutation.
    :param elite_fraction: Fraction of the best individuals carried over unchanged.
    :param tournament_size: Number of contestants in each tournament selection.
    :param dt: Simulation time step in seconds for lap-time evaluation.
    :param progress_callback: Optional callable invoked whenever a new best path is
        found. Receives ``(best_path, best_time, generation)`` as arguments.
    :return: The best :class:`Path` found.
    """
    num_lines = track.num_lines
    points_on_line = track.points_on_line
    num_elites = max(1, int(population_size * elite_fraction))

    # initialise population with random chromosomes
    population = [
        [random.randint(0, points_on_line - 1) for _ in range(num_lines)]
        for _ in range(population_size)
    ]

    best_chromosome: list[int] = population[0]
    best_time: float = float("inf")
    best_path: Optional[Path] = None

    for generation in range(num_generations):
        fitnesses = [_evaluate(chrom, track, gokart, dt) for chrom in population]

        # track the global best
        gen_best_idx = min(range(population_size), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] < best_time:
            best_time = fitnesses[gen_best_idx]
            best_chromosome = population[gen_best_idx]
            best_path = _chromosome_to_path(best_chromosome, track)

            if progress_callback is not None:
                progress_callback(best_path, best_time, generation)

        # elitism: carry the top individuals unchanged into the next generation
        sorted_indices = sorted(range(population_size), key=lambda i: fitnesses[i])
        elites = [population[i] for i in sorted_indices[:num_elites]]

        # fill the rest of the new population via selection, crossover and mutation
        new_population: list[list[int]] = list(elites)
        while len(new_population) < population_size:
            parent_a = _tournament_select(population, fitnesses, tournament_size)
            parent_b = _tournament_select(population, fitnesses, tournament_size)
            child = _uniform_crossover(parent_a, parent_b)
            child = _mutate(child, points_on_line, mutation_rate)
            new_population.append(child)

        population = new_population

    if best_path is None:
        best_path = _chromosome_to_path(best_chromosome, track)

    return best_path
