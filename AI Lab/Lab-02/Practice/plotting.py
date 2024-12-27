import matplotlib.pyplot as plt
import numpy as np

# Draw two points in the diagram, one at position (1, 3) and one in position (8, 10):
xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
plt.plot(xpoints, ypoints, "or")
# plt.show()
# Draw a line in a diagram from position (1, 3) to (2, 8) then to (6, 1) and finally to position (8, 10):
xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])
plt.plot(xpoints, ypoints)
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.show()
