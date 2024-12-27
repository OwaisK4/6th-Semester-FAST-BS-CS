import heapq

class Node:
    def __init__(self, value, heuristic):
        self.value = value
        self.heuristic = heuristic
        self.children = []

def astar_search_tree(root, goal_value):
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, (root.heuristic, root))
    
    while open_list:
        _, current_node = heapq.heappop(open_list)
        
        if current_node.value == goal_value:
            return current_node
        
        closed_set.add(current_node)
        
        for child in current_node.children:
            if child not in closed_set:
                total_cost = child.heuristic
                heapq.heappush(open_list, (total_cost, child))
    
    return None

root = Node('A', 10)
node_b = Node('B', 5)
node_c = Node('C', 8)
node_d = Node('D', 3)
node_e = Node('E', 2)
node_f = Node('F', 4)

root.children = [node_b, node_c]
node_b.children = [node_d, node_e]
node_c.children = [node_f]

goal_value = 'F'
result_node = astar_search_tree(root, goal_value)

if result_node:
    print("Goal node found:", result_node.value)
else:
    print("Goal node not found.")