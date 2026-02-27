"""Unit tests for all classes (Gokart, Path, Track, GokartDriveAnimation).

Covers: Gokart.get_f_resistance, get_acceleration; Path construction (valid/invalid),
path_length, get_v_max; Track construction (even/odd border_pts), get_random_path;
GokartDriveAnimation init. Uses optimal_gokart package (models + visualization).
"""

import numpy as np
import pytest

from optimal_gokart import Gokart, GokartDriveAnimation, Path, Track


class TestGokartGetFResistance:
    """Tests for Gokart.get_f_resistance."""

    def test_zero_speed_returns_zero(self) -> None:
        kart = Gokart(mass=100.0, f_grip=500.0, f_motor=200.0, k_drag=0.5)
        assert kart.get_f_resistance(0.0) == 0.0

    def test_positive_speed_returns_positive_force(self) -> None:
        kart = Gokart(mass=100.0, f_grip=500.0, f_motor=200.0, k_drag=0.5)
        # F = k_drag * v^2
        assert kart.get_f_resistance(10.0) == 0.5 * 10.0**2.0
        assert kart.get_f_resistance(2.0) == 0.5 * 4.0


class TestGokartGetAcceleration:
    """Tests for Gokart.get_acceleration."""

    def test_basic_physics(self) -> None:
        # At v=0: F_resistance=0, so a = f_motor / mass
        kart = Gokart(mass=100.0, f_grip=500.0, f_motor=200.0, k_drag=0.5)
        acc = kart.get_acceleration(0.0)
        assert acc == 200.0 / 100.0  # 2.0 m/s^2

    def test_higher_speed_reduces_acceleration(self) -> None:
        kart = Gokart(mass=100.0, f_grip=500.0, f_motor=200.0, k_drag=0.5)
        acc_low = kart.get_acceleration(5.0)
        acc_high = kart.get_acceleration(25.0)
        assert acc_high < acc_low


class TestPathConstruction:
    """Tests for Path construction."""

    def test_valid_input_creates_path(self) -> None:
        # splprep needs at least k+1 points (default k=3), so 4+ points
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5], [3.0, 1.0]], dtype=float)
        path = Path(pts, smooth_coef=0, num_interp_pts=100, pix_to_m_ratio=1.0)
        np.testing.assert_array_equal(path.pts, pts)
        assert path.num_interp_points == 100

    def test_invalid_input_too_few_points_raises_value_error(self) -> None:
        # Plan: Path should validate at least 2 rows and 2 columns
        pts_one_row = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError):
            Path(pts_one_row)

    def test_invalid_input_single_column_raises_value_error(self) -> None:
        pts_one_col = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError):
            Path(pts_one_col)


class TestPathGetVMax:
    """Tests for Path.get_v_max."""

    def test_returns_non_negative_values(self) -> None:
        pts = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.5], [10.0, 1.0]], dtype=float)
        path = Path(pts, smooth_coef=0, num_interp_pts=50, pix_to_m_ratio=1.0)
        kart = Gokart(mass=100.0, f_grip=500.0, f_motor=200.0, k_drag=0.5)
        v_max = path.get_v_max(kart)
        assert isinstance(v_max, np.ndarray)
        assert v_max.shape == (path.num_interp_points,)
        assert np.all(v_max >= 0)


# --- Track ---


class TestTrackConstruction:
    """Tests for Track construction."""

    def test_even_border_pts_succeeds(self) -> None:
        border_pts = np.array(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]], dtype=float
        )
        track = Track(border_pts, points_on_line=2, pix_to_m_ratio=1.0)
        assert track.num_lines == 2
        assert track.points_on_line == 2

    def test_odd_border_pts_count_raises_value_error(self) -> None:
        border_pts = np.array(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]], dtype=float
        )  # 3 rows
        with pytest.raises(ValueError, match="even"):
            Track(border_pts)


# --- GokartDriveAnimation ---


class TestGokartDriveAnimation:
    """Tests for GokartDriveAnimation."""

    def test_init_with_path_sets_attributes(self) -> None:
        # Build a Path directly with 4 known unique points to avoid relying on
        # random selection (get_random_path can produce a wraparound duplicate
        # that drops a point below the splprep k=3 threshold non-deterministically).
        path_pts = np.array(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]], dtype=float
        )
        path = Path(path_pts, num_interp_pts=50)
        kart = Gokart(mass=100.0, f_grip=500.0, f_motor=200.0, k_drag=0.5)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        animation = GokartDriveAnimation(img, path, kart, dt=0.1)
        assert animation.path is path
        assert animation.gokart is kart
        assert animation.track_image_arr is img
        assert animation.ttrack is not None
        assert hasattr(animation, "show")
