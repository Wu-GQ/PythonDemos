from LeetCode.LeetCode_Tree import TreeNode


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

    def minCameraCover(self, root: TreeNode) -> int:
        """
        968. 监控二叉树
        :see https://leetcode-cn.com/problems/binary-tree-cameras/
        """

        def min_camera_count_of_subtree(node: TreeNode) -> (int, int, int):
            """
            判断以当前节点作为根节点的子树，所需的最小摄像头数量
            :param node: 子树的根节点
            :return: 子树根节点没有监控时、有监控没摄像头时、有监控有摄像头时的最小摄像头数量
            """
            if not node:
                return 0, 0, 1

            left_non_monitor, left_non_camera, left_camera = min_camera_count_of_subtree(node.left)
            right_non_monitor, right_non_camera, right_camera = min_camera_count_of_subtree(node.right)

            # 根节点没有监控时，最小摄像头数量为左右节点有监控没摄像头时的数量之和
            non_monitor = left_non_camera + right_non_camera

            # 根节点有监控没摄像头时
            non_camera = min(left_camera + min(right_camera, right_non_camera), min(left_camera, left_non_camera) + right_camera)

            # 如果根节点有监控
            camera = min(left_non_monitor, left_non_camera, left_camera) + min(right_non_monitor, right_non_camera, right_camera) + 1

            # print(node.val, non_monitor, non_camera, camera)
            return non_monitor, non_camera, camera

        return min(min_camera_count_of_subtree(root)[1:])

    def treeToDoublyList(self, root: TreeNode) -> TreeNode:
        """
        剑指 Offer 36. 二叉搜索树与双向链表
        :see https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/
        """

        def inorder(node: TreeNode):
            nonlocal head_before, last
            if not node:
                return

            inorder(node.left)

            print(last.val, node.val)

            # 确定头结点
            if not head_before.right:
                head_before.right = node

            # 连接上一节点
            node.left = last
            last.right = node

            # 更新上一节点
            last = node

            inorder(node.right)

        if not root:
            return root

        head_before = TreeNode(0)
        last = head_before
        inorder(root)

        # 连接首尾节点
        last.right = head_before.right
        head_before.right.left = last

        return last.right

    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """
        235. 二叉搜索树的最近公共祖先
        :see https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
        """
        from functools import lru_cache

        @lru_cache(None)
        def exist(tree_root: TreeNode, node: TreeNode) -> bool:
            """ 判断node节点是否存在于tree_root作为根节点的子树中 """
            if not tree_root or not node:
                return False
            if tree_root and node and tree_root.val == node.val:
                return True
            if tree_root.val > node.val:
                return exist(tree_root.left, node)
            else:
                return exist(tree_root.right, node)

        # if not root or root.val == p.val and (exist(root.left, q) or exist(root.right, q)) or root.val == q.val and (exist(root.left, p) or exist(root.right, p)) or exist(root.left, p) and exist(root.right, q) or exist(root.left, q) and exist(root.right, p):
        #     return root
        # return self.lowestCommonAncestor(root.left, p, q) or self.lowestCommonAncestor(root.right, p, q)
        if root.val == p.val and exist(root, q) or root.val == q.val and exist(root, p):
            return root
        if p.val < root.val < q.val or q.val < root.val < p.val:
            return root
        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        if root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root


if __name__ == '__main__':
    s = Solution()

    root = TreeNode(6)
    root.left = TreeNode(2)
    root.right = TreeNode(8)
    # # #
    root.left.left = TreeNode(0)
    root.left.right = TreeNode(4)

    # root.left.left.left = TreeNode(8)
    # root.left.left.right = TreeNode(9)

    # #
    root.left.right.left = TreeNode(3)
    root.left.right.right = TreeNode(5)
    #
    root.right.left = TreeNode(7)
    root.right.right = TreeNode(9)

    # root.right.left.right = TreeNode(7)

    # print(s.recoverTree(root))
    print(s.lowestCommonAncestor(root, TreeNode(0), TreeNode(4)).val)
