"""Animation and visualization: GokartDriveAnimation."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .models import Gokart, Path


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

    def __init__(
        self,
        track_image_arr,
        path: Path,
        gokart: Gokart,
        dt=0.1,
        pstyle="ro",
        interval=10,
    ):
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

        (point,) = axes.plot(
            [self.ttrack[0][0] * self.path.pix_to_m_ratio],
            [self.ttrack[1][0] * self.path.pix_to_m_ratio],
            self.pstyle,
        )

        def ani(coords):
            point.set_data(
                [coords[0] * self.path.pix_to_m_ratio],
                [coords[1] * self.path.pix_to_m_ratio],
            )
            return point

        def frames():
            yield from zip(self.ttrack[0], self.ttrack[1])

        ani = FuncAnimation(fig, ani, frames=frames, interval=self.interval)

        plt.show()
