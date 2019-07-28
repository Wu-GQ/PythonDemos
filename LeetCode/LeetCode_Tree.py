from queue import Queue


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def maxDepth(self, root: TreeNode) -> int:
        """
        二叉树的最大深度
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/7/trees/47/
        """
        if root is None:
            return 0

        one_queue = Queue()
        two_queue = Queue()
        two_queue.put(root)

        deep = 0
        while not one_queue.empty() or not two_queue.empty():
            if not one_queue.empty():
                index = one_queue.get()
                if index.left is not None:
                    two_queue.put(index.left)
                if index.right is not None:
                    two_queue.put(index.right)
            else:
                one_queue = two_queue
                two_queue = Queue()
                deep += 1

        return deep

    def isValidBST(self, root: TreeNode) -> bool:
        """
        验证二叉搜索树
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/7/trees/48/
        """
        # 错误解法
        # if root is None:
        #     return True
        #
        # a = True if root.left is None else root.left.val < root.val
        # b = True if root.right is None else root.right.val > root.val
        #
        # print("val: %d, left: %d, right: %d, result: %d" % (root.val, root.left.val if root.left is not None else -1, root.right.val if root.right is not None else -1, a and b))
        #
        # return self.isValidBST(root.left) and self.isValidBST(root.right) if a and b else False

        result_list = self.intermediate_traversal(root)

        for i in range(len(result_list) - 1):
            if result_list[i] >= result_list[i + 1]:
                return False
        return True

    def isSymmetric(self, root: TreeNode) -> bool:
        """
        对称二叉树
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/7/trees/49/
        """
        return True if root is None else self.compare_tree_node(root.left, root.right)

    def compare_tree_node(self, left_tree_node: TreeNode, right_tree_node: TreeNode) -> bool:
        """使用递归判断是否为对称二叉树"""
        if left_tree_node is None and right_tree_node is None:
            return True
        elif left_tree_node is None or right_tree_node is None:
            return False
        elif left_tree_node.val == right_tree_node.val:
            return self.compare_tree_node(left_tree_node.left, right_tree_node.right) and self.compare_tree_node(left_tree_node.right,
                                                                                                                 right_tree_node.left)
        else:
            return False

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

    def levelOrder(self, root: TreeNode) -> list:
        """
        二叉树的层次遍历
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/7/trees/50/
        """
        if root is None:
            return []

        result_list = []
        this_list = [root]
        next_list = []
        while len(this_list) > 0:
            new_list = []
            for i in this_list:
                if i.left is not None:
                    next_list.append(i.left)
                if i.right is not None:
                    next_list.append(i.right)
                new_list.append(i.val)

            result_list.append(new_list)
            this_list = next_list
            next_list = []

        return result_list


if __name__ == '__main__':
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(2)

    root.left.left = TreeNode(2)
    # root.left.right = TreeNode(4)

    root.right.left = TreeNode(2)
    # root.right.right = TreeNode(3)

    # root.right.left.left = TreeNode(9)
    # root.right.left.right = TreeNode(11)

    print(Solution().levelOrder(root))
