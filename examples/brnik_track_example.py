"""Brnik track example: interactive border selection and path search."""

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np

from optimal_gokart import (
    GeneticAlgorithmPathFinder,
    Gokart,
    GokartDriveAnimation,
    Track,
)

# Project root (parent of examples/) for asset paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Named constants for interactive point selection
MAX_CLICK_POINTS = 500
CLICK_TIMEOUT_SEC = 150
ANIMATION_PAUSE_SEC = 0.3


def main() -> None:
    # read image on which track is visible
    track_image_path = PROJECT_ROOT / "docs" / "brnik-track-snip.PNG"
    track_image_arr = imageio.imread(track_image_path)

    # load predefined track points from a .npy file, if they are not already defined, open track image
    # for user to define them by clicking on the track borders
    pts_filename = PROJECT_ROOT / "points.npy"
    if pts_filename.is_file():
        points = np.load(pts_filename)
    else:
        # here, the user is prompted to click border points. When finished, click the wheel button on the mouse to stop
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.imshow(track_image_arr)
        points = np.array(plt.ginput(MAX_CLICK_POINTS, timeout=CLICK_TIMEOUT_SEC))
        np.save(pts_filename, points)
        plt.close()

    # show image with defined border points
    fig_pts = plt.figure()
    axes_pts = fig_pts.add_subplot(111)
    axes_pts.imshow(track_image_arr)
    axes_pts.plot(points[:, 0], points[:, 1], "r+")

    # the first two clicked points are used as a reference to determine the scale of image (bottom right corner of image)
    unit_pts = points[:2, :]

    ref_dist = (
        20  # distance of reference marking on image in meters (bottom right corner)
    )
    pix_per_m = np.linalg.norm(unit_pts[0, :] - unit_pts[1, :]) / ref_dist

    # create Gokart instance
    gokart = Gokart(mass=200, f_grip=200, f_motor=2000, k_drag=0.6125)

    # create track from defined points, ignore the first two points which were used for reference distance
    track = Track(points[2:, :], points_on_line=5, pix_to_m_ratio=pix_per_m)

    # get all waypoints by interpolating additional points between each two border points on track
    all_pts = np.array(track.interpolated_track_points_mat)
    all_pts = all_pts.reshape([all_pts.shape[0] * track.points_on_line, 2])

    # open figure to draw found paths
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

    # find optimal path using a genetic algorithm
    finder = GeneticAlgorithmPathFinder(population_size=25, num_generations=200, mutation_rate=0.1, tournament_size=10, elite_fraction=0.1, n_jobs=8)
    opt_path = finder.find_optimal_path(track, gokart, progress_callback=on_new_best)

    animation = GokartDriveAnimation(track_image_arr, opt_path, gokart)
    animation.show()


if __name__ == "__main__":
    main()
