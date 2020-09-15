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

        ''' # 递归
        result = self.intermediate_traversal(root.left)
        result.append(root.val)
        result += self.intermediate_traversal(root.right)
        '''

        ''' # 迭代
        result = []
        stack = [root]
        node = root
        while stack:
            # 左节点走到头
            while node:
                stack.append(node)
                node = node.left

            # 输出中间节点
            node = stack.pop()
            result.append(node.val)

            # 开始遍历右节点
            node = node.right
        '''

        # Morris 中序遍历
        result = []
        node = root
        while node:
            if not node.left:
                # node 无左孩子，则输出 node 的值，node = node.right；
                result.append(node.val)
                node = node.right
            else:
                # node 有左孩子，则找到左子树的最右的节点作为 next
                # 当遍历完一颗左子树后，next.right用来指向左子树遍历完成后的第一个节点，以继续遍历中间节点和右子树
                next = node.left
                while next.right and next.right != node:
                    next = next.right

                if next.right:
                    # 若 next.right 为空，将 next.right 指向 node 自身
                    next.right = None
                    result.append(node.val)
                    node = node.right
                else:
                    # 若 next.right 不为空,说明左子树已遍历完成
                    next.right = node
                    node = node.left

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
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    # # #
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)

    # root.left.left.left = TreeNode(8)
    # root.left.left.right = TreeNode(9)

    # #
    # root.left.right.left = TreeNode(10)
    # root.left.right.right = TreeNode(11)
    #
    root.right.left = TreeNode(6)
    # root.right.right = TreeNode(7)

    # root.right.left.right = TreeNode(7)

    # print(s.recoverTree(root))
    print(s.intermediate_traversal(root))
