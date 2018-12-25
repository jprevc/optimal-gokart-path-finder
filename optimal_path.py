import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import os
from tqdm import tqdm
import pickle
from OptimalPathClasses import Path, Gokart, GokartDriveAnimation, Track

track_image_arr = misc.imread('brnik_track_snip_3.png')

pts_filename = 'points_2.npy'
if not os.path.isfile(pts_filename):
    points = np.array(plt.ginput(500, timeout=150))
    np.save(pts_filename, points)
else:
    points = np.load(pts_filename)

unit_pts = points[:2,:]
pix_per_m = np.linalg.norm(unit_pts[0,:] - unit_pts[1,:]) / 20

gokart = Gokart(mass=200, f_grip=200, f_motor=2000, k_drag=0.6125)

# define track points, if they have not yet been defined
filename_suffix = "fifth_curve"
pts_filename = 'track_pts_' + filename_suffix + '.npy'
if not os.path.isfile(pts_filename):
    points = np.array(plt.ginput(500, timeout=600))
    np.save(pts_filename, points)
else:
    points = np.load(pts_filename)

# create track from defined points
track = Track(points, points_on_line=7, pix_to_m_ratio=pix_per_m)
all_pts = np.array(track.interpolated_track_points_mat)
all_pts = all_pts.reshape([all_pts.shape[0]*track.points_on_line,2])

# load path if it already exists in a file
if os.path.isfile('path_' + filename_suffix + '.txt'):
    with open('path_' + filename_suffix + '.txt', 'rb') as f:
        opt_path = pickle.load(f)
        t_min = opt_path.get_time_track(gokart, 0.1)[0][-1]
        interpolated_path = opt_path.get_interpolated_path(metric=False)
else:
    opt_path = None
    t_min = np.inf

# find optimal path
fig = plt.figure()
axes = fig.add_subplot(111)
N = int(1e6)
for _ in tqdm(range(N)):
    path = track.get_random_path()
    tvec,_ = path.get_time_track(gokart, 0.1)
    if tvec[-1] < t_min:
        with open('path_' + filename_suffix+ '.txt', 'wb') as f:
            pickle.dump(path, f)

        t_min = tvec[-1]
        opt_path = path

        axes.cla()
        axes.imshow(track_image_arr)
        axes.plot(all_pts[:, 0], all_pts[:, 1], 'r+')
        axes.plot(interpolated_path[:, 0], interpolated_path[:, 1])

        plt.draw()
        plt.pause(0.3)
        
        print(t_min)

plt.show()
