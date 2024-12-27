from collections import deque
from queue import PriorityQueue


class Search:
    def __init__(
        self,
        distances: dict[str, list[tuple[str, int]]] = None,
        heuristics: dict[str, int] = None,
    ) -> None:
        self.distances = distances
        # In the format: distances[src] = [(dest1, distance), (dest2, distance)]
        self.heuristics = heuristics
        # In the format: heuristics[src] = heuristic_weight

    # Returns the path as a string and the path cost as an integer.
    # In case the destination is not reached, it returns (None, None)
    def BreadthFirstSearch(
        self, src: str, target: str
    ) -> tuple[str, int] | tuple[None, None]:
        visited = {src}
        distance = {src: 0}
        q = deque()
        q.append(src)
        path = src
        path_cost = 0
        found: bool = False
        while len(q) > 0:
            src = q.popleft()
            for node in self.distances[src]:
                dest, dist = node[0], node[1]
                if dest in visited:
                    continue
                visited.add(dest)
                path += f" -> {dest}"
                if dest not in distance:
                    distance[dest] = distance[src] + dist
                else:
                    distance[dest] = min(distance[dest], distance[src] + dist)
                if dest == target:
                    q.clear()
                    path_cost = distance[dest]
                    found = True
                    break
                q.append(dest)
        if not found:
            return None, None
        else:
            return path, path_cost

    # Returns the path as a string and the path cost as an integer.
    # In case the destination is not reached, it returns (None, None)
    def UniformCostSearch(
        self, src: str, target: str
    ) -> tuple[str, int] | tuple[None, None]:
        visited = {src}
        # parent = {src: ""}
        distance = {src: 0}
        q = PriorityQueue()
        q.put((0, src))
        path = ""
        path_cost = 0
        found: bool = False
        while not q.empty():
            dist, src = q.get()
            if path == "":
                path = src
            else:
                path += f" -> {src}"
            for node in self.distances[src]:
                dest, dist = node[0], node[1]
                if dest in visited:
                    continue
                visited.add(dest)
                if dest not in distance:
                    distance[dest] = distance[src] + dist
                elif distance[dest] > distance[src] + dist:
                    distance[dest] = distance[src] + dist
                if dest == target:
                    path += f" -> {dest}"
                    path_cost = distance[dest]
                    q.queue.clear()
                    found = True
                    break
                q.put((distance[dest], dest))
        if not found:
            return None, None
        else:
            return path, path_cost

    # Returns the path as a string and the path cost as an integer.
    # In case the destination is not reached, it returns (None, None)
    def GreedyBestFirstSearch(
        self, src: str, target: str
    ) -> tuple[str, int] | tuple[None, None]:
        closed_list = {src}
        open_list = PriorityQueue()
        open_list.put((self.heuristics[src], src))
        path = ""
        path_cost = 0
        found: bool = False
        while not open_list.empty():
            _, src = open_list.get()
            if path == "":
                path = src
            else:
                path += f" -> {src}"
            path_cost += self.heuristics[src]

            if src == target:
                open_list.queue.clear()
                found = True
                break

            open_list.queue.clear()
            for node in self.distances[src]:
                dest, dist = node[0], node[1]
                if dest not in closed_list:
                    open_list.put((self.heuristics[dest], dest))
                    closed_list.add(dest)

        if not found:
            return None, None
        else:
            return path, path_cost

    # Returns the path as a string and the path cost as an integer.
    # In case the destination is not reached, it returns (None, None)
    def IterativeDeepeningDepthFirstSearch(
        self, src: str, target: str
    ) -> tuple[str, int] | tuple[None, None]:
        visited = set(src)
        depths = {}

        def maxDepth_of_graph(src: str, depth: int):
            maxDepth = depth
            for dest, _ in self.distances[src]:
                if dest not in visited:
                    visited.add(dest)
                    maxDepth = max(maxDepth, maxDepth_of_graph(dest, depth + 1))
                    visited.remove(dest)
            return maxDepth

        self.cost = -1
        self.path = ""

        def DFS(
            src: str, cost: int, path: int, depth_limit: int, current_depth: int
        ) -> bool:
            if current_depth > depth_limit:
                return False
            if src == target:
                self.cost = cost
                self.path = path
                return True
            for dest, dist in self.distances[src]:
                if DFS(
                    dest,
                    cost + dist,
                    path + f" -> {dest}",
                    depth_limit,
                    current_depth + 1,
                ):
                    return True
            return False

        maxDepth = maxDepth_of_graph(src, 0)
        # print(f"Max Depth is: {maxDepth}")

        found: bool = False
        for i in range(1, maxDepth + 1):
            if DFS(src, 0, src, i, 0):
                found = True
                break
        if found:
            return self.path, self.cost
        else:
            return None, None
