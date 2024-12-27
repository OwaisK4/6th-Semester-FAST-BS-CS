import random


def agent(camera, distance):
    front_range = 8
    side_range = 2
    rear_range = 0.05  # in metres

    if camera == "front" and distance <= front_range:
        print(f"Braking")
    elif camera in ("left", "right") and distance <= side_range:
        lane_change = "right" if camera == "left" else "left"
        print(f"Moving to {lane_change} lane")
    elif camera == "rear" and distance <= rear_range:
        print("Braking")
    else:
        print("No change in lanes")


front_distance = random.randint(0, 10)
agent("front", front_distance)
left_distance = random.randint(0, 10)
agent("left", left_distance)
right_distance = random.randint(0, 10)
agent("right", right_distance)
rear_distance = random.randint(0, 10)
agent("rear", rear_distance)
