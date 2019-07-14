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

        deep = 1
        while not one_queue.empty() and not two_queue.empty():
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


if __name__ == '__main__':
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)

    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)

    print(Solution().maxDepth(root))
