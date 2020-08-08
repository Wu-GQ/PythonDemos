class TreeNode:

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:

    def intermediate_traversal(self, root: TreeNode) -> list:
        """
        二叉树的中序遍历
        :see
        """
        if root is None:
            return []

        result = self.intermediate_traversal(root.left)
        result.append(root.val)
        result += self.intermediate_traversal(root.right)

        return result

    def recoverTree(self, root: TreeNode) -> None:
        """
        99. 恢复二叉搜索树
        :see https://leetcode-cn.com/problems/recover-binary-search-tree/
        """

        def dfs(node: TreeNode):
            if node:
                dfs(node.left)
                values.append((node.val, node))
                dfs(node.right)

        # 中序遍历所有节点值
        values = [(-float('inf'), TreeNode(0))]
        dfs(root)

        # 找到被错误排列的两个节点
        errors = []
        for i in range(1, len(values)):
            if values[i][0] < values[i - 1][0]:
                errors.append((i - 1, i))

        # 重新排列
        # 当有两个组合排列错误时，交换前一个组合中的第一个值和后一个组合中的第二个值
        # 当只有一个组合排列错误时，交换这个组合的值
        a, b = errors[0][0], errors[0 if len(errors) == 1 else 1][1]

        values[a][1].val, values[b][1].val = values[b][0], values[a][0]

        print(self.intermediate_traversal(root))


if __name__ == '__main__':
    s = Solution()

    root = TreeNode(1)
    root.left = TreeNode(3)
    # root.right = TreeNode(3)
    # # #
    # root.left.left = TreeNode(4)
    root.left.right = TreeNode(2)

    # root.left.left.left = TreeNode(8)
    # root.left.left.right = TreeNode(9)

    # #
    # root.left.right.left = TreeNode(10)
    # root.left.right.right = TreeNode(11)
    #
    # root.right.left = TreeNode(6)
    # root.right.right = TreeNode(7)

    # root.right.left.right = TreeNode(7)

    print(s.recoverTree(root))
    print(s.intermediate_traversal(root))
