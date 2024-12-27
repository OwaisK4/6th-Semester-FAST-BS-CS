from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def depth_first_search_graph(self, start, target, max_depth):
        for depth_limit in range(max_depth + 1):
            visited = set()
            if self.depth_limited_search(start, target, depth_limit, visited):
                return True
        return False

    def depth_limited_search(self, node, target, depth_limit, visited):
        if node == target:
            return True
        if depth_limit <= 0:
            return False
        if node in visited:
            return False

        visited.add(node)
        for neighbor in self.graph[node]:
            if self.depth_limited_search(neighbor, target, depth_limit - 1, visited):
                return True
        return False

# Example graph
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(1, 4)
g.add_edge(2, 5)
g.add_edge(2, 6)

start = 0
target = 6
max_depth = 3
if g.depth_first_search_graph(start, target, max_depth):
    print("Node found within depth limit")
else:
    print("Node not found within depth limit")
