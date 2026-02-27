from optimal_gokart import Gokart, Path, Track


def test_import():
    """Package exports Gokart, Path, Track."""
    assert Gokart is not None
    assert Path is not None
    assert Track is not None


def test_track_get_random_path_returns_path():
    """Track.get_random_path returns a Path instance."""
    import numpy as np

    border_pts = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 2.5],
            [10.0, 5.0],
            [5.0, 5.0],
            [0.0, 5.0],
            [0.0, 2.5],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    track = Track(border_pts, points_on_line=3, pix_to_m_ratio=1.0)
    path = track.get_random_path(smooth_coef=0, num_interp_points=100)
    assert isinstance(path, Path)
    assert path.pts.shape[0] == track.num_lines
    assert path.pts.shape[1] == 2
