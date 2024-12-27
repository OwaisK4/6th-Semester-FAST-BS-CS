class SimpleReflexAgent:
    def __init__(self):
        self.location = "A"
        self.status = "Dirty"
        self.actions = []

    def perceive(self, location, status):
        self.location = location
        self.status = status

    def think(self):
        if self.status == "Dirty":
            self.actions.append("Suck")
        else:
            if self.location == "A":
                self.actions.append("Right")
            else:
                self.actions.append("Left")

    def act(self):
        if self.actions:
            action = self.actions.pop(0)
            return action
        return "NoOp"

# Example usage
agent = SimpleReflexAgent()

while True:
    location = input("Enter the current location of the agent (A or B): ")
    status = input("Enter the status of the current location (Dirty or Clean): ")
    agent.perceive(location, status)
    agent.think()
    action = agent.act()
    print("Agent performs action:", action)
    if action == "NoOp":
        break


####################################
    
#Task01
# Consider an interactive and cognitive environment (ICE) in which a smart camera is
# monitoring robot movement from one location to another. Let a robot be at location A for
# some time instant and then moves to point B and eventually reaches at point C and so on
# and so forth shown in the Fig. Develop a Python code to calculate a distance between
# reference point R (4, 0) of a camera and A, B, and C and N number of locations.

import math

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

point_ = (4, 0)

points = {
    'A': (2, 3),
    'B': (5, 7),
    'C': (8, 4)
}

for point_name, point in points.items():
    distance = calculate_distance(point_, point)
    print(f"Distance between R and {point_name}: {distance}")


#Task02
# Consider a scenario, cameras placed on every side of the car — front, rear, left and right —
# to stitch together a 360-degree view of the environment. For a three-lane road a car is
# moving on a middle lane, consider the below scenario
# • If the front camera detects the object within range of 8 meters breaks are applied
# automatically.
# • If the left camera detects the object within range of 2 meters car moves to the right lane.
# • If the right camera detects the object within range of 2 meters car moves to the left lane.
# • For parking the car if the rear camera detects the object within 5 cm breaks are applied.

class Car:
    def __init__(self):
        self.front_camera_range = 8
        self.side_camera_range = 2
        self.rear_camera_range = 0.05

        self.current_lane = "middle"

    def front_camera_detect(self, distance):
        if distance <= self.front_camera_range:
            print("Object detected in front!")

    def left_camera_detect(self, distance):
        if distance <= self.side_camera_range:
            print("Object detected on the left!")
            self.current_lane = "right"

    def right_camera_detect(self, distance):
        if distance <= self.side_camera_range:
            print("Object detected on the right!")
            self.current_lane = "left"

    def rear_camera_detect(self, distance):
        if distance <= self.rear_camera_range:
            print("Object detected behind!")

my_car = Car()

front_distance = 6
left_distance = 1
right_distance = 0.5
rear_distance = 0.03

my_car.front_camera_detect(front_distance)
my_car.left_camera_detect(left_distance)
my_car.right_camera_detect(right_distance)
my_car.rear_camera_detect(rear_distance)

print("Current lane:", my_car.current_lane)

# Task#03 Simple Reflex Agents
# Consider the following scenario where the UAV receives temperature data from the installed
# sensors in a residential area. Assume that there are nine sensors installed that are measuring
# temperature in centigrade. Develop a Python code to calculate the average temperature
# in F.

temperatures = [20, 22, 21, 23, 25, 19, 18, 24, 26]

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

total_temperature_fahrenheit = 0
for temperature_celsius in temperatures:
    temperature_fahrenheit = celsius_to_fahrenheit(temperatures)
    total_temperature_fahrenheit += temperature_fahrenheit

average_temperature_fahrenheit = total_temperature_fahrenheit / len(temperatures)

print("Average temperature in Fahrenheit:", average_temperature_fahrenheit)