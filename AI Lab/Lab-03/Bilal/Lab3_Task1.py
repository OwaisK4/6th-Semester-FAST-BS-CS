import numpy as np
from numpy.linalg import norm as euclidean_distance

N = 4
reference_point = np.array([4, 0])
robot_point = np.empty((N, 2))

for i in range(N):
    x, y = map(float, input(f"Enter coordinates for point {i+1} (x, y): ").split())
    robot_point[i] = np.array([x, y])

distances = euclidean_distance(robot_point - reference_point, axis=1)

print("Distances between the camera and robot locations:")
for i, distance in enumerate(distances):
    print(f"Point {i+1}: {distance:.2f}")
