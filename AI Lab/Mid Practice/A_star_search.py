from queue import PriorityQueue

# Define the grid for the maze
maze = [
    ["S", ".", ".", ".", "X"],
    [".", "X", ".", "X", "."],
    [".", ".", ".", ".", "."],
    ["X", ".", "X", ".", "E"],
]


def heuristic(start: tuple[int, int], end: tuple[int, int]):
    answer = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
    return answer


def astar_search(graph: list[list[int]], start: tuple[int, int], end: tuple[int, int]):
    n = len(graph)
    m = len(graph[0])
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    open_list = PriorityQueue()
    open_list.put((0, start))
    # closed_list = set(start)
    parent = {start: None}
    cost = {start: 0}
    while open_list.qsize() > 0:
        _, node = open_list.get()
        if node == end:
            path = ""
            current = node
            while current:
                path += str(current)
                current = parent[current]
            return "Found", path
        for dx, dy in moves:
            x = node[0] + dx
            y = node[1] + dy
            if x >= 0 and y >= 0 and x < n and y < m and graph[x][y] != "X":
                if (x, y) not in cost or cost[(x, y)] > cost[node] + 1:
                    cost[(x, y)] = cost[node] + 1
                    f_value = cost[node] + heuristic((x, y), end)
                    parent[(x, y)] = node
                    open_list.put((f_value, (x, y)))


# Test the algorithm
start = (0, 0)
end = (3, 4)
path = astar_search(maze, start, end)
if path:
    print("Shortest path found:", path)
else:
    print("No path found")
