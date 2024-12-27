class Node:
    def __init__(self, value, parent=None, heuristic=None):
        self.value = value
        self.parent = parent
        self.heuristic = heuristic
        self.children = []

    def add_child(self, child):
        self.children.append(child)

root = Node("S")
root.add_child(Node("A", root, 3))
root.add_child(Node("B", root, 2))
root.children[0].add_child(Node("C", root.children[0], 4))
root.children[0].add_child(Node("D", root.children[0], 1))
root.children[1].add_child(Node("E", root.children[1], 3))
root.children[1].add_child(Node("F", root.children[1], 1))
root.children[1].children[1].add_child(Node("I", root.children[1].children[1], 2))
root.children[1].children[1].add_child(Node("G", root.children[1].children[1], 3))
root.children[1].children[0].add_child(Node("H", root.children[1].children[0], 5))

def greedy_search(root, destination):
    current_node = root
    visited_nodes = []
    while current_node.value != destination:
        visited_nodes.append(current_node.value)
        if not current_node.children:
            while current_node.parent is not None:
                parent = current_node.parent
                parent.children.remove(current_node)
                if parent.children:
                    current_node = min(parent.children, key=lambda node: node.heuristic)
                    break
                else:
                    current_node = parent
            else:
                return None, visited_nodes
        else:
            current_node = min(current_node.children, key=lambda node: node.heuristic)
    visited_nodes.append(destination)
    return current_node, visited_nodes

destination_node, visited_nodes = greedy_search(root, 'H')

if destination_node:
    print("Destination node found:", destination_node.value)
else:
    print("Destination node H not reachable!")

print("Visited nodes:", visited_nodes)
