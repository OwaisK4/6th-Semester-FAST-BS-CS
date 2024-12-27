class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

def depth_first_search_tree(root, target):
    if root is None:
        return None

    stack = [(root, 0)]

    while stack:
        node, depth = stack.pop()
        if node.value == target:
            return node, depth
        for child in reversed(node.children):
            stack.append((child, depth + 1))

    return None

# Example tree
root = TreeNode(1)
child1 = TreeNode(2)
child2 = TreeNode(3)
child3 = TreeNode(4)
child4 = TreeNode(5)

root.add_child(child1)
root.add_child(child2)
child1.add_child(child3)
child1.add_child(child4)

target = 5
result = depth_first_search_tree(root, target)
if result:
    node, depth = result
    print(f"Node found at depth {depth}")
else:
    print("Node not found")
