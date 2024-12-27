class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def dfs_tree(root):
    if not root:
        return []

    result = []

    def dfs(node):
        if not node:
            return
        result.append(node.value)
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return result

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.left = TreeNode(6)
root.right.right = TreeNode(7)

print("DFS traversal of the tree:")
print(dfs_tree(root))
