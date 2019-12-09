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

    def kthSmallest(self, root: TreeNode, k: int) -> int:
        """
        二叉搜索树中第K小的元素
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/269/tree/1165/
        """

        def inorder_traversal(root: TreeNode, k: int, result_list: list) -> list:
            """ 二叉树的中序遍历 """
            if root is None or len(result_list) >= k:
                return result_list

            left_list = inorder_traversal(root.left, k, result_list)
            left_list.append(root.val)
            return inorder_traversal(root.right, k, left_list)

        # 将二叉搜索树的层次遍历序列，转为中序遍历序列，中序遍历序列是递增的序列
        search_list = inorder_traversal(root, k, result_list=[])

        return search_list[k - 1]

    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """
        二叉树的最近公共祖先
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/269/tree/1166/
        """
        # 只要是公共祖先，就可以同时找到p节点和q节点。若要找到最近公共祖先，则是查看其公共祖先的左节点或者右节点是否可以同时找到p节点和q节点
        if root is None or root.val == p.val or root.val == q.val:
            return root

        # 当可以从左子树或者右子树找到p节点或者q节点时，节点返回就不是空
        left_node = self.lowestCommonAncestor(root.left, p, q)
        right_node = self.lowestCommonAncestor(root.right, p, q)

        if left_node is not None and right_node is not None:
            return root
        elif left_node is not None:
            return left_node
        elif right_node is not None:
            return right_node
        else:
            return None

    def serialize(self, root: TreeNode) -> str:
        """
        将二叉树转换为以字符串表示的数组
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/269/tree/1167/
        """
        # 层次遍历，类似广度优先遍历
        if root is None:
            return '[]'

        # 类似广度优先遍历的队列
        queue = [root]
        # 队列中非空节点的数量
        queue_node_count = 1
        # 存储结果的列表
        # result_list = []
        result_string = "["

        while queue_node_count > 0:
            top_node = queue.pop(0)
            if top_node == '#' or top_node is None:
                # result_list.append(None)
                result_string = f'{result_string}#,'

                queue.append('#')
                queue.append('#')
            else:
                queue_node_count -= 1
                # result_list.append(top_node.val)
                result_string = f'{result_string}{top_node.val},'

                if top_node.left is None:
                    queue.append('#')
                else:
                    queue_node_count += 1
                    queue.append(top_node.left)

                if top_node.right is None:
                    queue.append('#')
                else:
                    queue_node_count += 1
                    queue.append(top_node.right)

        return f'{result_string[:-1]}]'

    def deserialize(self, data: str) -> TreeNode:
        """
        将以字符串表示的数组转换为二叉树
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/269/tree/1167/
        """

        def create_tree(father_node: TreeNode, father_index: int, tree_list: list):
            """ 根据数组创建二叉树 """
            if father_node is None:
                return

            left_index = 2 * father_index + 1
            if left_index >= len(tree_list):
                return

            left_value = tree_list[left_index]
            if left_value.lstrip('-').isdigit():
                father_node.left = TreeNode(int(left_value))
                create_tree(father_node.left, left_index, tree_list)

            if left_index + 1 >= len(tree_list):
                return
            right_value = tree_list[left_index + 1]
            if right_value.lstrip('-').isdigit():
                father_node.right = TreeNode(int(right_value))
                create_tree(father_node.right, left_index + 1, tree_list)

        # 预处理输入的字符串
        tree_list = data[1:-1].split(',')

        if len(tree_list) == 0 or not tree_list[0].lstrip('-').isdigit():
            return None

        # 创建根节点
        root_node = TreeNode(int(tree_list[0]))

        # 创建二叉树
        create_tree(root_node, 0, tree_list)

        return root_node

    def getSkyline(self, buildings: list) -> list:
        """
        天际线问题
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/269/tree/1168/
        """
        # 解法参考（扫描线解法）: https://leetcode-cn.com/problems/the-skyline-problem/solution/218tian-ji-xian-wen-ti-sao-miao-xian-fa-by-ivan_al/
        in_queue_list = []
        out_queue_list = []

        for i in buildings:
            in_queue_list.append([i[0], i[2]])
            out_queue_list.append([i[1], i[2]])

        in_queue_list.sort(key=lambda x: x[0])
        out_queue_list.sort(key=lambda x: x[0])

        height_list = []
        result_list = []
        last_max_height = 0

        while len(in_queue_list) > 0 or len(out_queue_list) > 0:
            if len(out_queue_list) > 0 and len(in_queue_list) > 0 and in_queue_list[0][0] == out_queue_list[0][0]:
                last_building = out_queue_list.pop(0)
                height_list.remove(last_building[1])

                building = in_queue_list.pop(0)
                height_list.append(building[1])
            elif len(out_queue_list) == 0 or (len(in_queue_list) > 0 and in_queue_list[0][0] < out_queue_list[0][0]):
                building = in_queue_list.pop(0)
                height_list.append(building[1])
            else:
                building = out_queue_list.pop(0)
                height_list.remove(building[1])

            max_height = max(height_list) if len(height_list) > 0 else 0

            if len(result_list) == 0 or (result_list[-1][0] != building[0] and max_height != last_max_height):
                last_max_height = max_height
                result_list.append([building[0], max_height])
            elif result_list[-1][0] == building[0]:
                last_max_height = max_height
                result_list[-1] = [building[0], max_height]

        return result_list

    max_path_sum = float('-inf')

    def maxPathSum(self, root: TreeNode) -> int:
        """
        二叉树中的最大路径和
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/272/dynamic-programming/1175/
        """
        if root is None:
            return 0
        self.max_path_sum = root.val

        return max(self.recursive_child_tree_max_path_sum(root), self.max_path_sum)

    def recursive_child_tree_max_path_sum(self, root: TreeNode) -> int:
        # 使用递归，判断每个节点的子树的最大路径和
        if root is None:
            return 0

        # 计算左子树和右子树的最大路径和
        left_max_path_sum = self.recursive_child_tree_max_path_sum(root.left)
        right_max_path_sum = self.recursive_child_tree_max_path_sum(root.right)

        # 路径只包含一半子树和该子树的父节点
        part_child_tree_max_path_sum = root.val + max(0, max(left_max_path_sum, right_max_path_sum))
        # 路径只在这个子树中
        child_tree_max_path_sum = root.val + max(0, left_max_path_sum) + max(0, right_max_path_sum)

        # 更新最大路径和
        self.max_path_sum = max(self.max_path_sum, child_tree_max_path_sum, part_child_tree_max_path_sum)

        return part_child_tree_max_path_sum

    def buildTree(self, preorder: list, inorder: list) -> TreeNode:
        """
        从前序与中序遍历序列构造二叉树
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-medium/32/trees-and-graphs/87/
        """
        if len(preorder) == 0 or len(inorder) == 0:
            return None

        # 前序序列的第一个节点即为根节点
        root_node_value = preorder.pop(0)

        # 找到根节点在中序序列中的位置
        root_node_index_in_inorder = inorder.index(root_node_value)

        # 中序序列根节点左侧的序列，即为左子树的中序序列
        # 左子树的前序序列的节点数量，与左子树的中序序列的节点数量相同
        left_tree_in_order = inorder[:root_node_index_in_inorder]
        left_tree_pre_order = preorder[:len(left_tree_in_order)]

        right_tree_in_order = inorder[root_node_index_in_inorder + 1:]
        right_tree_pre_order = preorder[len(left_tree_in_order):]

        # 通过递归，组建二叉树
        root_node = TreeNode(root_node_value)
        root_node.left = self.buildTree(left_tree_pre_order, left_tree_in_order)
        root_node.right = self.buildTree(right_tree_pre_order, right_tree_in_order)

        return root_node


if __name__ == '__main__':
    # root = TreeNode(1)
    # # root.left = TreeNode(2)
    # root.right = TreeNode(3)
    #
    # # root.left.left = TreeNode(6)
    # # root.left.right = TreeNode(2)
    # #
    # # root.left.right.left = TreeNode(7)
    # # root.left.right.right = TreeNode(4)
    #
    # # root.right.left = TreeNode(4)
    # root.right.right = TreeNode(5)

    # print(Solution().lowestCommonAncestor(root, TreeNode(5), TreeNode(4)).val)
    # string = Solution().serialize(root)
    # print(string)
    # print(Solution().getSkyline([[1, 2, 1], [1, 2, 2], [1, 2, 3]]))

    s = Solution()
    tree = s.buildTree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])
    print(tree.val)
