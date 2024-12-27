from collections import deque

def bfs(graph, start, goal):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        current_room, path = queue.popleft()
        if current_room == goal:
            return path
        if current_room not in visited:
            visited.add(current_room)
            for neighbor in graph[current_room]:
                queue.append((neighbor, path + [neighbor]))
    return None

house = {
    'bedroom': ['Study room', 'living room'],
    'Study room': ['bedroom', 'TV lounge'],
    'living room': ['bedroom', 'kitchen'],
    'kitchen': ['room', 'living room']
}

start_room = 'bedroom'
goal_room = 'TV lounge'

path = bfs(house, start_room, goal_room)

if path:
    print("Steps to get from", start_room, "to", goal_room, ":")
    for i, room in enumerate(path):
        print(f"Step {i+1}: Go to {room}")
else:
    print("You got lost! Couldn't find the kitchen.")