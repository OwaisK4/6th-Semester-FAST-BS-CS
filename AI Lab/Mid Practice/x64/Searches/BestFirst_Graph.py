import heapq

def best_first_search_graph(graph, start, goal, heuristic):
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, (heuristic[start], start))
    
    while open_list:
        _, current_node = heapq.heappop(open_list)
        
        if current_node == goal:
            return current_node
        
        closed_set.add(current_node)
        
        for neighbor, weight in graph[current_node].items():
            if neighbor not in closed_set:
                heapq.heappush(open_list, (heuristic[neighbor], neighbor))
    
    return None

graph = {
    'A': {'B': 5, 'C': 3},
    'B': {'A': 5, 'D': 8, 'E': 2},
    'C': {'A': 3, 'F': 4},
    'D': {'B': 8},
    'E': {'B': 2, 'F': 3},
    'F': {'C': 4, 'E': 3}
}

heuristic = {
    'A': 7,
    'B': 6,
    'C': 5,
    'D': 2,
    'E': 3,
    'F': 0
}

start_node = 'A'
goal_node = 'F'
result_graph = best_first_search_graph(graph, start_node, goal_node, heuristic)

print("Path found in graph:", result_graph)