import numpy as np
from scipy import interpolate
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from itertools import product
import random

class Gokart:
    """
    Gokart class, specifies parameters of gokart.

    :param mass: mass of gokart in kg, including the driver
    :type mass: float

    :param f_grip: Grip force in N (Newtons)
    :type f_grip: float

    :param max_speed: maximum possible speed of gokart in m/s
    :type max_speed: float

    """
    def __init__(self, mass, f_grip, f_motor, k_drag):
        self.mass = mass
        self.f_grip = f_grip
        self.f_motor = f_motor
        self.k_drag = k_drag

    def get_f_resistance(self, v):
        """
        Calculates resistance force according to gokart's speed.

        :param v: Gokart's speed in m/s
        :type v: float

        :return: Resistive force in Newtons
        :rtype: float
        """

        return self.k_drag*v**2 if v > 0 else 0

    def get_acceleration(self, v):
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

    def __init__(self, pts, smooth_coef:float=0, num_interp_pts:int=1000, pix_to_m_ratio=1):
        self.pts = pts
        self.smooth_coef = smooth_coef
        self.num_interp_pts = num_interp_pts
        self._u = np.linspace(0, 1, self.num_interp_pts)
        self.pix_to_m_ratio = pix_to_m_ratio

    @property
    def _spline_interpolation_rep(self):
        """
        Calculates spline interpolation representation.
        """
        return interpolate.splprep([self.pts[:, 0] / self.pix_to_m_ratio,
                                    self.pts[:, 1] / self.pix_to_m_ratio],
                                    s=self.smooth_coef)[0]


    def get_interpolated_path(self, metric=True):
        """
        Returns interpolated path as a N by 2 array.

        :param metric: True, if calculated point's units should be in meters
                       (applicable only if pix_to_m_ratio was defined),
        :type metric: bool

        :return: N by 2 array of interpolated points, where N is specified
                 by num_interp_pts field.
        :rtype: np.array

        """
        interpolated_path = interpolate.splev(self._u, self._spline_interpolation_rep)
        interpolated_path = np.array(interpolated_path).T

        return interpolated_path if metric else interpolated_path * self.pix_to_m_ratio

    def _get_interp_path_der(self, order:int=1):
        """
        Calculates derivative of the interpolated path.

        :param order: Order of the derivative.
        :type order: int

        :return: N by 2 array of points which lie on the derivation path.
        """
        der = interpolate.spalde(self._u, self._spline_interpolation_rep)

        der_arr = np.zeros((len(self._u), 2))
        for i in range(len(self._u)):
            der_arr[i, :] = np.array([der[0][i][order], der[1][i][order]])

        return der_arr

    @property
    def radius(self):
        """
        Computes turning radius for each point on the track.
        """
        der1 = self._get_interp_path_der(order=1)
        der2 = self._get_interp_path_der(order=2)

        xd = der1[:, 0]
        yd = der1[:, 1]

        xdd = der2[:, 0]
        ydd = der2[:, 1]

        return ((xd**2 + yd**2)**3) / ((xd*ydd - yd*xdd)**2)

    def get_v_max(self, gokart: Gokart):
        """
        Calculates maximum possible (theoretical) gokart speed at each point on the path.
        This speed represents maximum speed at which gokart still doesn't slid.

        :param gokart: An instance of Gokart class
        :type gokart: Gokart

        :return: 1D array of size self.num_interp_points, each value represents
                 maximum possible speed of gokart at point on track.
        :rtype: np.array
        """

        v_max_arr = np.sqrt(gokart.f_grip * self.radius / gokart.mass)

        return v_max_arr

    @property
    def path_length(self, metric=True):
        """
        Returns path length.

        :param metric: True, if calculated point's units should be in meters
                       (applicable only if pix_to_m_ratio was defined),
        :type metric: bool

        :return: Calculated length of the path.
        :rtype: float
        """
        interp_path = self.get_interpolated_path(metric)

        path_len = 0
        for i in range(len(self._u)-1):
            path_len += np.linalg.norm(interp_path[i+1,:] - interp_path[i,:])

        return path_len

    def get_time_track(self, gokart: Gokart, dt, v0=0, max_loop_cnt=100000):
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

        # get interpolation function for maximum speed
        v_max_interp = interpolate.interp1d(self._u, self.get_v_max(gokart))

        v = v0
        p_norm = 0
        loop_cnt = 0
        total_time = 0
        time_vec = []
        while True:

            # if normalised position is more than 1 that means that path is finished
            if p_norm > 1:
                break

            # get speed for next iteration
            v = v + dt*gokart.get_acceleration(v)

            # check if calculated speed is more than theoretically fastest for
            # current path radius
            if v > float(v_max_interp(p_norm)):
                v = float(v_max_interp(p_norm))


            # get normalised position for next iteration
            p_norm = p_norm + dt * v / len_path
            p_norm_lst.append(p_norm)

            # increment loop counter and check if max number of loops is exceeded
            loop_cnt += 1
            if loop_cnt > max_loop_cnt:
                raise RuntimeError("Maximum number of loops exceeded.")

            time_vec.append(total_time)
            total_time += dt

        time_vec = np.array(time_vec)
        p_norm_arr = np.array(p_norm_lst)

        return time_vec, interpolate.splev(p_norm_arr, self._spline_interpolation_rep)


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
    def __init__(self, border_pts, points_on_line=2, pix_to_m_ratio=1):
        self.border_pts = border_pts
        self.pix_to_m_ratio = pix_to_m_ratio

        if border_pts.shape[0] % 2 != 0:
            raise ValueError("Number of points in 'border_pts' array should be even.")

        self.num_lines = int(border_pts.shape[0] / 2)

        self.points_on_line = points_on_line

    @property
    def interpolated_track_points_mat(self):
        """
        Returns point positions for each of the defined line on a track,

        :return: Point positions in a list, where length of list is equal to
                 number of lines on a track. Each list entry contains M points,
                 where M is specified by points_on_line object field.
        """
        points_arr = []
        for i in range(self.num_lines):
            line_rep,u = interpolate.splprep([self.border_pts[i*2:i*2+2,0],
                                             self.border_pts[i*2:i*2+2,1]],
                                             k=1,
                                             s=0)

            line_pts = interpolate.splev(np.linspace(0, 1, self.points_on_line), line_rep)
            points_arr.append(np.array(line_pts).T)

        return points_arr

    def _get_path_point_mat(self, point_inds):
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
        path_point_mat = np.zeros((self.num_lines, 2))
        for i, point_ind in enumerate(point_inds):
            path_point_mat[i, :] = self.interpolated_track_points_mat[i][point_ind, :]

        return path_point_mat

    def get_random_path(self, smooth_coef=0, num_interp_points=1000):
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
        rand_point_inds = [random.randint(0, self.points_on_line-1) for _ in range(self.num_lines)]

        # get matrix of points on the path based on the random indexes
        path_point_mat = self._get_path_point_mat(rand_point_inds)

        return Path(path_point_mat,
                    smooth_coef=smooth_coef,
                    num_interp_pts=num_interp_points,
                    pix_to_m_ratio=self.pix_to_m_ratio)

    def paths(self, smooth_coef=0, num_interp_points=1000):
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

            yield Path(path_point_mat,
                       smooth_coef=smooth_coef,
                       num_interp_pts=num_interp_points,
                       pix_to_m_ratio=self.pix_to_m_ratio)


class GokartDriveAnimation:
    """
    Class which contains useful functions for showing animations of gokart's
    driving.

    :param track_image_arr: Array of image of the track.
    :type track_image_arr: np.array

    :param path: Gokart's driving path.
    :type path: Path

    :param gokart: Gokart which is driving on the path.
    :type gokart: Gokart

    :param dt: Time interval for the driving simulation
    :type dt: float

    :param pstyle: Style of the point which represents the gokart in the animation.
    :type pstyle: str

    :param interval: refresh interval for the animation in ms.
    :type interval: int
    """
    def __init__(self, track_image_arr, path: Path, gokart: Gokart, dt=0.1, pstyle='ro', interval=10):
        self.track_image_arr = track_image_arr
        self.path = path
        self.gokart = gokart
        self.pstyle = pstyle
        self.interval = interval

        if path:
            _, self.ttrack = path.get_time_track(gokart, dt=dt)

    def show(self):
        """
        Shows animation.
        """
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.imshow(self.track_image_arr)

        point, = axes.plot([self.ttrack[0][0] * self.path.pix_to_m_ratio],
                           [self.ttrack[1][0] * self.path.pix_to_m_ratio],
                           self.pstyle)

        def ani(coords):
            point.set_data([coords[0] * self.path.pix_to_m_ratio],
                           [coords[1] * self.path.pix_to_m_ratio])
            return point

        def frames():
            for x_tt, y_tt in zip(self.ttrack[0], self.ttrack[1]):
                yield x_tt, y_tt

        ani = FuncAnimation(fig, ani, frames=frames, interval=self.interval)

        plt.show()


if __name__ == '__main__':
    pass
