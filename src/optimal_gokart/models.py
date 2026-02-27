"""Physics and track models: Gokart, Path, and Track."""

from __future__ import annotations

import random
from collections.abc import Generator, Sequence
from itertools import product

import numpy as np
from scipy import interpolate


class Gokart:
    """
    Gokart class, specifies parameters of gokart.

    :param mass: mass of gokart in kg, including the driver
    :type mass: float

    :param f_grip: Grip force in N (Newtons)
    :type f_grip: float

    :param f_motor: Motor force in N (Newtons)
    :type f_motor: float

    :param k_drag: Drag coefficient (air resistance proportional to v^2)
    :type k_drag: float

    """

    def __init__(
        self, mass: float, f_grip: float, f_motor: float, k_drag: float
    ) -> None:
        self.mass = mass
        self.f_grip = f_grip
        self.f_motor = f_motor
        self.k_drag = k_drag

    def get_f_resistance(self, v: float) -> float:
        """
        Calculates resistance force according to gokart's speed.

        :param v: Gokart's speed in m/s
        :type v: float

        :return: Resistive force in Newtons
        :rtype: float
        """

        return float(self.k_drag * v**2.0) if v > 0 else 0.0

    def get_acceleration(self, v: float) -> float:
        """
        Calculates acceleration of the gokart by subtracting force of resistance
        with pull force of the motor. The result is then divided by gokart's
        mass.

        :param v: Gokart's speed in m/s
        :type v: float

        :return: Gokart acceleration in m^2/s
        :rtype: float
        """

        acc = (self.f_motor - self.get_f_resistance(v)) / self.mass

        return acc


class Path:
    """
    Path class, defines a specific path on the track

    :param pts: Points on the path, N by 2 array
    :type pts: np.array

    :param smooth_coef: smoothing coeffiecient for spline interpolation.
    :type smooth_coef: float

    :param num_interp_pts: number of interpolation points to be used in spline interpolation.
    :type num_interp_pts: int

    :param pix_to_m_ratio: Pixels to meter ratio coefficient for units of points
                           specified in pts parameter.
    :type pix_to_m_ratio: float
    """

    def __init__(
        self,
        pts: np.ndarray,
        smooth_coef: float = 0,
        num_interp_pts: int = 1000,
        pix_to_m_ratio: float = 1,
    ) -> None:
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
            raise ValueError(
                "pts must be a 2D array with at least 2 rows and 2 columns"
            )
        self._pts = pts
        self._smooth_coef = smooth_coef
        self._num_interp_pts = num_interp_pts
        self._u = np.linspace(0, 1, self._num_interp_pts)
        self._pix_to_m_ratio = pix_to_m_ratio
        self._set_spline_interpolation_rep()
        self._set_radius()

    @property
    def num_interp_points(self) -> int:
        return self._num_interp_pts

    @num_interp_points.setter
    def num_interp_points(self, val: int) -> None:
        self._num_interp_pts = val
        self._u = np.linspace(0, 1, self._num_interp_pts)

    @property
    def pts(self) -> np.ndarray:
        return self._pts

    @pts.setter
    def pts(self, val: np.ndarray) -> None:
        self._pts = val
        self._set_spline_interpolation_rep()

    @property
    def smooth_coef(self) -> float:
        return self._smooth_coef

    @smooth_coef.setter
    def smooth_coef(self, val: float) -> None:
        self._smooth_coef = val
        self._set_spline_interpolation_rep()

    @property
    def pix_to_m_ratio(self) -> float:
        return self._pix_to_m_ratio

    @pix_to_m_ratio.setter
    def pix_to_m_ratio(self, val: float) -> None:
        self._pix_to_m_ratio = val
        self._set_spline_interpolation_rep()

    def _set_spline_interpolation_rep(self) -> None:
        pts_scaled = self.pts / self.pix_to_m_ratio
        # Remove consecutive duplicate points (including the wraparound between
        # last and first point) to avoid zero chord-lengths that cause splprep
        # to fail with "Invalid inputs".
        dists = np.linalg.norm(np.diff(pts_scaled, axis=0), axis=1)
        keep = np.concatenate(([True], dists > 0))
        pts_scaled = pts_scaled[keep]
        # Also drop the last point if it coincides with the first (periodic).
        if np.linalg.norm(pts_scaled[-1] - pts_scaled[0]) == 0:
            pts_scaled = pts_scaled[:-1]
        self._spline_interpolation_rep = interpolate.splprep(
            [pts_scaled[:, 0], pts_scaled[:, 1]],
            s=self.smooth_coef,
            per=True,
        )[0]

    def get_interpolated_path(self, metric: bool = True) -> np.ndarray:
        """
        Returns interpolated path as a N by 2 array.

        :param metric: True, if calculated point's units should be in meters
                       (applicable only if pix_to_m_ratio was defined),
        :type metric: bool

        :return: N by 2 array of interpolated points, where N is specified
                 by num_interp_pts field.
        :rtype: np.array

        """
        interpolated_path: np.ndarray = np.array(
            interpolate.splev(self._u, self._spline_interpolation_rep)
        ).T

        result: np.ndarray = (
            interpolated_path if metric else interpolated_path * self.pix_to_m_ratio
        )
        return result

    def _get_interp_path_der(self, order: int = 1) -> np.ndarray:
        """
        Calculates derivative of the interpolated path.

        :param order: Order of the derivative.
        :type order: int

        :return: N by 2 array of points which lie on the derivation path.
        """
        der = interpolate.spalde(self._u, self._spline_interpolation_rep)

        der_arr: np.ndarray = np.array(der)

        return der_arr[:, :, order].T

    @property
    def radius(self) -> np.ndarray:
        return self._radius

    def _set_radius(self) -> None:
        """
        Computes turning radius for each point on the track.
        """
        der1 = self._get_interp_path_der(order=1)
        der2 = self._get_interp_path_der(order=2)

        xd = der1[:, 0]
        yd = der1[:, 1]

        xdd = der2[:, 0]
        ydd = der2[:, 1]

        denom = (xd * ydd - yd * xdd) ** 2
        self._radius = np.where(denom > 1e-10, (xd**2 + yd**2) ** 3 / denom, np.inf)

    def get_v_max(self, gokart: Gokart) -> np.ndarray:
        """
        Calculates maximum possible (theoretical) gokart speed at each point on the path.
        This speed represents maximum speed at which gokart still doesn't slid.

        :param gokart: An instance of Gokart class
        :type gokart: Gokart

        :return: 1D array of size self.num_interp_points, each value represents
                 maximum possible speed of gokart at point on track.
        :rtype: np.array
        """

        v_max_arr: np.ndarray = np.sqrt(gokart.f_grip * self.radius / gokart.mass)

        return v_max_arr

    @property
    def path_length(self) -> float:
        """
        Returns path length (in meters if pix_to_m_ratio was defined).

        :return: Calculated length of the path.
        :rtype: float
        """
        interp_path = self.get_interpolated_path(True)

        path_len: float = float(
            np.sum(np.linalg.norm(np.diff(interp_path, axis=0), axis=1))
        )

        return path_len

    def get_time_track(
        self,
        gokart: Gokart,
        dt: float,
        v0: float = 0,
        max_loop_cnt: int = 100000,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Gets time track of the path, which is a sequence of gokart's positions
        on the track at regular intervals. The positions are calculated by
        running a simulation.

        :param gokart: Gokart object
        :type gokart: Gokart

        :param dt: Time interval in seconds.
        :type dt: float

        :param v0: Starting speed of the gokart in m/s.
        :type v0:  float

        :param max_loop_cnt: Maximum loop count for the simulation loop.
                             If loop counter increases beyond this value,
                             an error will be raised.
        :type max_loop_cnt: int

        :return: (time_vec, time_track), where time_vec is a time vector of
                 length N and time_track is an N by 2 array which defines
                  gokart's positions.
        :rtype: tuple of np.array

        """
        p_norm_lst = []
        len_path = self.path_length

        # Pre-compute the v_max array; because self._u is a uniform linspace(0,1)
        # grid we can convert p_norm → array index directly, avoiding the overhead
        # of calling a scipy interp1d object on every simulation step.
        v_max_arr = self.get_v_max(gokart)
        n_pts = len(v_max_arr)
        n_pts_m1 = n_pts - 1

        v = v0
        p_norm = 0.0
        loop_cnt = 0
        total_time = 0.0
        time_vec = []
        while True:

            # if normalised position is more than 1 that means that path is finished
            if p_norm > 1:
                break

            # get speed for next iteration
            v = v + dt * gokart.get_acceleration(v)

            # check if calculated speed is more than theoretically fastest for
            # current path radius — linear interpolation on the uniform grid
            idx_f = p_norm * n_pts_m1
            idx_lo = int(idx_f)
            if idx_lo >= n_pts_m1:
                v_max_at_p = float(v_max_arr[n_pts_m1])
            else:
                frac = idx_f - idx_lo
                v_max_at_p = float(
                    v_max_arr[idx_lo]
                    + frac * (v_max_arr[idx_lo + 1] - v_max_arr[idx_lo])
                )
            if v > v_max_at_p:
                v = v_max_at_p

            # get normalised position for next iteration
            p_norm = p_norm + dt * v / len_path
            p_norm_lst.append(p_norm)

            # increment loop counter and check if max number of loops is exceeded
            loop_cnt += 1
            if loop_cnt > max_loop_cnt:
                raise RuntimeError("Maximum number of loops exceeded.")

            time_vec.append(total_time)
            total_time += dt

        time_arr: np.ndarray = np.array(time_vec)
        p_norm_arr: np.ndarray = np.array(p_norm_lst)

        return time_arr, list(
            interpolate.splev(p_norm_arr, self._spline_interpolation_rep)
        )


class Track:
    """
    Track class, defines the track which contains all path combinations on
    which the gokart could drive.

    :param border_pts: Border points of the track. An array of N by 2,
                       where N should be an even number. Border points define
                       lines which run along and perpendicular to track.
    :type: border_pts: np.array

    :param points_on_line: Defines how many points should be put to each defined line.
    :type points_on_line: int

    :param pix_to_m_ratio: Pixels to meter ratio coefficient for units of points
                           specified in pts parameter.
    :type pix_to_m_ratio: float


    """

    def __init__(
        self,
        border_pts: np.ndarray,
        points_on_line: int = 2,
        pix_to_m_ratio: float = 1,
    ) -> None:
        self.border_pts = border_pts
        self.pix_to_m_ratio = pix_to_m_ratio

        if border_pts.shape[0] % 2 != 0:
            raise ValueError("Number of points in 'border_pts' array should be even.")

        self._num_lines = int(border_pts.shape[0] / 2)

        self._points_on_line = points_on_line

        self._set_interpolated_track_points_mat()

    @property
    def num_lines(self) -> int:
        return self._num_lines

    @num_lines.setter
    def num_lines(self, val: int) -> None:
        # when num_lines attribute gets changed, track points need to be recalculated
        self._num_lines = val
        self._set_interpolated_track_points_mat()

    @property
    def points_on_line(self) -> int:
        return self._points_on_line

    @points_on_line.setter
    def points_on_line(self, val: int) -> None:
        # when points_on_line attribute gets changed, track points need to be recalculated
        self._points_on_line = val
        self._set_interpolated_track_points_mat()

    def _set_interpolated_track_points_mat(self) -> None:
        points_arr = []
        for i in range(self.num_lines):
            line_rep, u = interpolate.splprep(
                [
                    self.border_pts[i * 2 : i * 2 + 2, 0],
                    self.border_pts[i * 2 : i * 2 + 2, 1],
                ],
                k=1,
                s=0,
            )

            line_pts = interpolate.splev(
                np.linspace(0, 1, self.points_on_line), line_rep
            )
            points_arr.append(np.array(line_pts).T)

        self._interpolated_track_points_mat = points_arr

    @property
    def interpolated_track_points_mat(self) -> list[np.ndarray]:
        """
        Returns point positions for each of the defined line on a track.

        :return: Point positions in a list, where length of list is equal to
                 number of lines on a track. Each list entry contains M points,
                 where M is specified by points_on_line object field.
        """

        return self._interpolated_track_points_mat

    def _get_path_point_mat(self, point_inds: Sequence[int]) -> np.ndarray:
        """
        Returns sequence of points, which define one possible path on the track.

        :param point_inds: List of length N, where N is equal to num_lines
                           object field. For each index i, the following should
                           be true: 0 <= i < M, where M is equal to
                           points_on_line object field value.
        :type point_inds: list

        :return: N by 2 matrix of points, which define a path.
        :rtype: np.array
        """
        path_point_mat = np.array(self.interpolated_track_points_mat)

        return path_point_mat[np.arange(self.num_lines), point_inds, :]

    def get_random_path(
        self, smooth_coef: float = 0, num_interp_points: int = 1000
    ) -> Path:
        """
        Returns one random path from all of the possible paths on the track.

        :param smooth_coef: Smoothing coefficient used in interpolation.
        :type smooth_coef: float

        :param num_interp_points: Number of interpolation points.
        :type num_interp_points: int

        :return: Path object
        :rtype: Path
        """

        # generate list of random point indexes for each line
        rand_point_inds = [
            random.randint(0, self.points_on_line - 1) for _ in range(self.num_lines)
        ]

        # get matrix of points on the path based on the random indexes
        path_point_mat = self._get_path_point_mat(rand_point_inds)

        return Path(
            path_point_mat,
            smooth_coef=smooth_coef,
            num_interp_pts=num_interp_points,
            pix_to_m_ratio=self.pix_to_m_ratio,
        )

    def paths(
        self, smooth_coef: float = 0, num_interp_points: int = 1000
    ) -> Generator[Path, None, None]:
        """
        This function is a generetor, which yields all possible paths on the
        track.

        :param smooth_coef: Smoothing coefficient in spline interpolation.
        :type smooth_coef: float

        :param num_interp_points: Number of interpolation points for the path.
        :type num_interp_points: int

        :return: Path object
        :rtype: Path
        """

        # get list of all combinations of point indexes on each line
        point_combs = [list(range(self.points_on_line)) for _ in range(self.num_lines)]

        # create all possible combinations of indexes of points on the track lines
        combs = product(*point_combs)

        # yield each path, defined by a index combination, one by one
        for comb in combs:
            path_point_mat = self._get_path_point_mat(comb)

            yield Path(
                path_point_mat,
                smooth_coef=smooth_coef,
                num_interp_pts=num_interp_points,
                pix_to_m_ratio=self.pix_to_m_ratio,
            )
