import matplotlib.pyplot as plt
import numpy as np

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
