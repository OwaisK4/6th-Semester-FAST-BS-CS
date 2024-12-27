class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def dfs_tree_depth_limited(root, depth_limit):
    if not root:
        return []

    result = []

    def dfs(node, depth):
        if not node or depth == 0:
            return
        result.append(node.value)
        dfs(node.left, depth - 1)
        dfs(node.right, depth - 1)

    dfs(root, depth_limit)
    return result

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.left = TreeNode(6)
root.right.right = TreeNode(7)

depth_limit = 2

print(f"DFS traversal of the tree with depth limit {depth_limit}")
print(dfs_tree_depth_limited(root, depth_limit))
