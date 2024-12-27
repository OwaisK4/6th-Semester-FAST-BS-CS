def limited_dfs(graph, start, depth, visited=None):
    if visited is None:
        visited = set()
    
    if depth == 0:
        return
    
    visited.add(start)
    print(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            limited_dfs(graph, neighbor, depth - 1, visited)
    
    return visited

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start_node = 'A'
depth_limit = 2
limited_dfs(graph, start_node, depth_limit)