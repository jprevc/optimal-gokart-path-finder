"""Animation and visualization: GokartDriveAnimation."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
        track_image_arr: np.ndarray,
        path: Path,
        gokart: Gokart,
        dt: float = 0.1,
        pstyle: str = "ro",
        interval: int = 10,
    ) -> None:
        self.track_image_arr = track_image_arr
        self.path = path
        self.gokart = gokart
        self.pstyle = pstyle
        self.interval = interval

        if path:
            _, self.ttrack = path.get_time_track(gokart, dt=dt)

    def show(self) -> None:
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

        def ani(coords: tuple[Any, Any]) -> Any:
            point.set_data(
                [coords[0] * self.path.pix_to_m_ratio],
                [coords[1] * self.path.pix_to_m_ratio],
            )
            return point

        def frames() -> Generator[tuple[Any, Any], None, None]:
            yield from zip(self.ttrack[0], self.ttrack[1])

        _anim = FuncAnimation(fig, ani, frames=frames, interval=self.interval)

        plt.show()
