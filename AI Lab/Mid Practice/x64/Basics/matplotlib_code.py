# Code1:
# Plotting x and y points
# The plot() function is used to draw points (markers) in a diagram.
# By default, the plot() function draws a line from point to point.
# The function takes parameters for specifying points in the diagram.
# Parameter 1 is an array containing the points on the x-axis.
# Parameter 2 is an array containing the points on the y-axis.
# If we need to plot a line from (1, 3) to (8, 10), we have to pass two arrays [1, 8] and [3, 10] to the plot
# function.
# Draw a line in a diagram from position (1, 3) to position (8, 10):
import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
plt.plot(xpoints, ypoints)
plt.show()

# Code2:
# Draw two points in the diagram, one at position (1, 3) and one in position (8, 10):
xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
plt.plot(xpoints, ypoints, "o")
plt.show()
# Draw a line in a diagram from position (1, 3) to (2, 8) then to (6, 1) and finally to position (8, 10):
xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])
plt.plot(xpoints, ypoints)
plt.show()

# Code3:
# You can use also use the shortcut string notation parameter to specify the marker.
# This parameter is also called fmt, and is written with this syntax:
# marker|line|color
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, "o:r")
plt.show()
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, linestyle="dotted")
plt.show()

# Code4:
# Labels:
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)
plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.grid()
plt.show()

# Code5:
# Subplots
# With the subplot() function you can draw multiple plots in one figure:
# plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(1, 2, 1)
plt.plot(x, y)
# plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])
plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.show()

# Code6:
# Bar plots
# With Pyplot, you can use the bar() function to draw bar graphs:
x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])
plt.bar(x, y)
plt.show()
# for horizontal bar use 'barh'
plt.barh(x, y)
plt.show()
x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])
plt.bar(x, y, color="#4CAF50")
plt.show()
# histogram
x = np.random.normal(170, 10, 250)
plt.hist(x)
plt.show()
# Pie Chart
y = np.array([35, 25, 25, 15])
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
plt.pie(y, labels=mylabels, startangle=90)
plt.show()

#
import matplotlib.pyplot as graph
import matplotlib.pyplot as piechart

x = [1, 2, 3, 4, 5]
L1 = [6, 10, 2, 1, 4]
L2 = [0, 4, 7, 5, 4]

graph.figure(figsize=(6, 4))
graph.plot(x, L1, label="L1")
graph.plot(x, L2, "r--", label="L2", linewidth=2)
graph.xlabel("Time (minutes)")
graph.ylabel("Water Level (litres)")
graph.title("[GRAPH] Water level with respect to time")
graph.legend()
graph.grid()
graph.show()

piechart.title("Pie Chart")
data = ["Mortgage", "Repair", "Food", "Utilities"]
percentage = [51.72, 10.34, 17.24, 20.69]
color_codes = ["Purple", "red", "blue", "green"]

piechart.figure(figsize=(5, 5))
piechart.pie(
    percentage, labels=data, colors=color_codes, autopct="%1.1f%%", startangle=90
)
piechart.show()
