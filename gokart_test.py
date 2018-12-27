from optimal_path_classes import Gokart
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np

A = 1
drag_coef = 1
air_density = 1.225
k_drag = 0.5 * air_density * drag_coef * A

gokart = Gokart(mass=200,
                f_grip=100,
                f_motor=2000,
                k_drag=k_drag)

N = 10000
v = np.zeros(N)
dt = 0.01
for i in range(N-1):
    v[i+1] = v[i] + gokart.get_acceleration(v[i])*dt

plt.figure()
plt.plot(np.arange(N)*dt, v)

plt.show()



