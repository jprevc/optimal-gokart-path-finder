import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import os
from optimal_path_classes import Gokart, Track, GokartDriveAnimation

# read image on which track is visible
track_image_arr = misc.imread('brnik_track_snip.png')

# load predefined track points from a .npy file, if they are not already defined, open track image
# for user to define them by clicking on the track borders
pts_filename = 'points.npy'
if os.path.isfile(pts_filename):
    points = np.load(pts_filename)
else:
    # here, user is prompted to click border points. When finished, click wheel button on mouse to stop
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.imshow(track_image_arr)
    points = np.array(plt.ginput(500, timeout=150))
    np.save(pts_filename, points)
    plt.close()

# show image with defined border points
fig_pts = plt.figure()
axes_pts = fig_pts .add_subplot(111)
axes_pts.imshow(track_image_arr)
axes_pts.plot(points[:,0], points[:,1], 'r+')

# first two clicked points are used as a reference to determine the scale of image (bottom right corner of image)
unit_pts = points[:2,:]

ref_dist = 20  # distance of reference marking on image in meters (bottom right corner)
pix_per_m = np.linalg.norm(unit_pts[0,:] - unit_pts[1,:]) / ref_dist

# create Gokart instance
gokart = Gokart(mass=200, f_grip=200, f_motor=2000, k_drag=0.6125)

# create track from defined points, ignore first two points which were used for reference distance
track = Track(points[2:,:], points_on_line=5, pix_to_m_ratio=pix_per_m)

# get all waypoints by interpolating additional points between each two border points on track
all_pts = np.array(track.interpolated_track_points_mat)
all_pts = all_pts.reshape([all_pts.shape[0]*track.points_on_line,2])

# find optimal path using Monte Carlo method, redraw path each time better one is found
N = int(100)
t_min = np.inf

# open figure to draw found paths
fig = plt.figure()
axes = fig.add_subplot(111)
for _ in range(N):
    path = track.get_random_path()

    # calculate time needed to complete the track using defined gokart
    tvec,_ = path.get_time_track(gokart, 0.1)

    # check if this path is better in terms of time needed to complete it
    path_duration = tvec[-1]
    if path_duration < t_min:

        t_min = path_duration
        opt_path = path

        interpolated_path = path.get_interpolated_path(metric=False)

        # redraw new path
        axes.cla()
        axes.imshow(track_image_arr)
        axes.plot(all_pts[:, 0], all_pts[:, 1], 'r+')
        axes.plot(interpolated_path[:, 0], interpolated_path[:, 1])

        plt.draw()
        plt.pause(0.3)
        
        print("New optimal path found! Duration of driving: {:.3f} seconds".format(t_min))

gda = GokartDriveAnimation(track_image_arr, opt_path, gokart)
gda.show()