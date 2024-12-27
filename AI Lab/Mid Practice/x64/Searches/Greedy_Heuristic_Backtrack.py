class Node:
    def __init__(self, value, heuristic=None, parent=None):
        self.value = value
        self.heuristic = heuristic
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

def greedy_search(root, destination):
    current_node = root
    visited_nodes = [current_node.value]
    while current_node.value != destination:
        if not current_node.children:
            while current_node.parent is not None:
                parent = current_node.parent
                parent.children.remove(current_node)
                if parent.children:
                    current_node = min(parent.children, key=lambda node: node.heuristic)
                    visited_nodes.append(current_node.value)
                    break
                else:
                    current_node = parent
            else:
                return None, visited_nodes
        else:
            current_node = min(current_node.children, key=lambda node: node.heuristic)
            visited_nodes.append(current_node.value)
    return current_node, visited_nodes

def main():
    root = Node('S', heuristic=8)
    a = Node('A', heuristic=6)
    b = Node('B', heuristic=7)
    c = Node('C', heuristic=4)
    d = Node('D', heuristic=5)
    e = Node('E', heuristic=4)
    f = Node('F', heuristic=6)
    g = Node('G', heuristic=4)
    h = Node('H', heuristic=0)
    root.add_child(a)
    root.add_child(b)
    a.add_child(c)
    a.add_child(d)
    b.add_child(e)
    b.add_child(f)
    c.add_child(g)
    c.add_child(h)

    destination_node, visited_nodes = greedy_search(root, 'H')

    if destination_node:
        print("Optimal move:", visited_nodes)
        print("Destination node found:", destination_node.value)
    else:
        print("Destination node H not reachable!")

if __name__ == "__main__":
    main()
