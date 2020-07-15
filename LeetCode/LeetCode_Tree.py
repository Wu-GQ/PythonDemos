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

        def compare_tree_node(left: TreeNode, right: TreeNode) -> bool:
            """使用递归判断是否为对称二叉树"""
            if left is None and right is None:
                return True
            elif left is None or right is None:
                return False
            elif left.val == right.val:
                return compare_tree_node(left.left, right.right) and compare_tree_node(left.right, right.left)
            else:
                return False

        if not root:
            return True
        return compare_tree_node(root.left, root.right)

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
        """
        if not root:
            return []

        stack = [root]
        length = 1
        result = []
        while stack:
            value_list = []
            while length > 0:
                length -= 1

                node = stack.pop(0)
                value_list.append(node.val)

                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)

            result.append(value_list)
            length = len(stack)

        return result

    def averageOfLevels(self, root: TreeNode) -> list:
        """
        637. 二叉树的层平均值
        :see https://leetcode-cn.com/problems/average-of-levels-in-binary-tree/
        """
        if not root:
            return []

        stack = [root]
        length = 1
        result = []
        while stack:
            value_sum = 0

            for i in range(length):
                node = stack.pop(0)
                value_sum += node.val

                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)

            result.append(value_sum / length)
            length = len(stack)

        return result

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
        124.二叉树中的最大路径和
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/272/dynamic-programming/1175/
        """

        def max_path_of_subtree(root: TreeNode) -> (int, int):
            if not root:
                return 0, -float('inf')

            left = max_path_of_subtree(root.left)
            right = max_path_of_subtree(root.right)

            return max(max(left[0], right[0]), 0) + root.val, max(left[1], right[1], left[0] + right[0] + root.val,
                                                                  left[0] + root.val,
                                                                  right[0] + root.val, root.val)

        return max_path_of_subtree(root)[1] if root else 0

    def buildTree(self, preorder: list, inorder: list) -> TreeNode:
        """
        105. 从前序与中序遍历序列构造二叉树
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

    def minDepth(self, root: TreeNode) -> int:
        """
        二叉树的最小深度
        :see https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/
        """
        if root is None:
            return 0
        elif root.left is None and root.right is None:
            return 1
        elif root.left is not None and root.right is not None:
            return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
        elif root.left is not None:
            return self.minDepth(root.left) + 1
        elif root.right is not None:
            return self.minDepth(root.right) + 1

    def invertTree(self, root: TreeNode) -> TreeNode:
        """
        翻转二叉树
        :see https://leetcode-cn.com/problems/invert-binary-tree/
        """
        if root is None:
            return None

        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)

        return root

    def preorderTraversal(self, root: TreeNode) -> list:
        """
        前序遍历
        :see https://leetcode-cn.com/problems/binary-tree-preorder-traversal/
        """
        # result_list = []
        #
        # def preorder_traversal(root: TreeNode):
        #     if root is None:
        #         return
        #
        #     result_list.append(root.val)
        #     preorder_traversal(root.left)
        #     preorder_traversal(root.right)
        #
        # preorder_traversal(root)
        #
        # return result_list

        if root is None:
            return []

        result_list = []
        stack_list = [root]
        checked_node_set = set()

        while len(stack_list) > 0:
            tree_node = stack_list[-1]

            if tree_node not in checked_node_set:
                result_list.append(tree_node.val)
                checked_node_set.add(tree_node)

            if tree_node.left is not None and tree_node.left not in checked_node_set:
                stack_list.append(tree_node.left)
            elif tree_node.right is not None and tree_node.right not in checked_node_set:
                stack_list.append(tree_node.right)
            else:
                stack_list.pop()

        return result_list

    def flatten(self, root: TreeNode) -> None:
        """
        二叉树展开为链表
        :see https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/
        """
        if root is None:
            return

        self.flatten(root.left)
        self.flatten(root.right)

        if root.left is not None:
            temp = root.right
            root.right = root.left
            root.left = None

            tree_node = root.right
            while tree_node.right is not None:
                tree_node = tree_node.right
            tree_node.right = temp

    def maxProduct(self, root: TreeNode) -> int:
        """
        1339. 分裂二叉树的最大乘积
        :see https://leetcode-cn.com/problems/maximum-product-of-splitted-binary-tree/
        """
        sum_of_sub_tree_dict = {}

        def sum_of_sub_tree(node: TreeNode) -> int:
            """ 求子树的节点和 """
            if node is None:
                return 0

            if node in sum_of_sub_tree_dict:
                return sum_of_sub_tree_dict[node]

            node_sum = sum_of_sub_tree(node.left) + sum_of_sub_tree(node.right) + node.val
            sum_of_sub_tree_dict[node] = node_sum

            return node_sum

        total_sum = sum_of_sub_tree(root)

        max_value = 10 ** 9 + 7

        product = 0
        subtract = float('inf')
        for node in sum_of_sub_tree_dict:
            result = abs(total_sum - sum_of_sub_tree_dict[node] * 2)
            if result < subtract:
                subtract = result

                product = (total_sum - sum_of_sub_tree_dict[node]) * sum_of_sub_tree_dict[node]
                if product > max_value:
                    product %= max_value

        return product

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        """
        二叉树的直径
        :see https://leetcode-cn.com/problems/diameter-of-binary-tree/
        """

        def max_deep_of_tree(root: TreeNode) -> int:
            """ 以root为根节点的树的最大深度 """
            if root is None:
                return 0
            if root in max_deep_dict:
                return max_deep_dict[root]
            max_deep = max(max_deep_of_tree(root.left), max_deep_of_tree(root.right)) + 1
            max_deep_dict[root] = max_deep
            return max_deep

        def max_path_of_tree(root: TreeNode) -> int:
            """ 以root为根节点的树的最长路径 """
            if root is None:
                return 0
            return max(max_path_of_tree(root.left), max_path_of_tree(root.right),
                       max_deep_of_tree(root.left) + max_deep_of_tree(root.right) - 1)

        max_deep_dict = {}

        return max_deep_of_tree(root) - 1 if root is not None else 0

    def rightSideView(self, root: TreeNode) -> list:
        """
        199. 二叉树的右视图
        :see https://leetcode-cn.com/problems/binary-tree-right-side-view/
        """
        # 层次遍历即可，等同于广度优先遍历
        if not root:
            return []

        result = []
        current_queue = [root]
        while current_queue:
            result.append(current_queue[-1].val)

            next_queue = []
            while current_queue:
                next_node = current_queue.pop(0)
                if next_node.left:
                    next_queue.append(next_node.left)
                if next_node.right:
                    next_queue.append(next_node.right)
            current_queue = next_queue

        return result

    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        """
        572. 另一个树的子树
        :see https://leetcode-cn.com/problems/subtree-of-another-tree/
        """

        def is_same_tree(a: TreeNode, b: TreeNode) -> bool:
            """ 判断是否是完全相同的树 """
            if not a and not b:
                return True
            elif not a or not b:
                return False
            else:
                return a.val == b.val and is_same_tree(a.left, b.left) and is_same_tree(a.right, b.right)

        # 递归每一个节点，当节点值相同时，判断该节点为根节点的子树是否与目标树相同
        if not s and not t:
            return True
        elif not s or not t:
            return False

        return s.val == t.val and is_same_tree(s, t) or self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

    def isValidBST(self, root: TreeNode) -> bool:
        """
        98. 验证二叉搜索树
        :see https://leetcode-cn.com/problems/validate-binary-search-tree/
        """

        # 验证中序遍历的顺序是否为从小到大的顺序

        def inorder_traversal(root_node: TreeNode):
            if root_node is None:
                return
            inorder_traversal(root_node.left)
            node_list.append(root_node.val)
            inorder_traversal(root_node.right)

        node_list = []
        inorder_traversal(root)

        for i in range(1, len(node_list)):
            if node_list[i] <= node_list[i - 1]:
                return False
        return True

    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """
        236. 二叉树的最近公共祖先
        :see https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/
        """
        if not root or p == root or q == root:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root
        return left if left else right

    def goodNodes(self, root: TreeNode) -> int:
        """
        5398. 统计二叉树中好节点的数目
        :param root:
        :return:
        """
        if not root:
            return 0

        # 最大栈
        val_stack = [root.val]
        node_stack = [root]

        result = 0
        check_node_set = set()

        while node_stack:
            # print(node_stack[-1].val)
            if node_stack[-1].left and node_stack[-1].left not in check_node_set:
                node = node_stack[-1].left
                node_stack.append(node)
                check_node_set.add(node)

                if val_stack[-1] <= node.val:
                    # print(node.val)
                    result += 1
                val_stack.append(max(node.val, val_stack[-1]))
                continue

            if node_stack[-1].right and node_stack[-1].right not in check_node_set:
                node = node_stack[-1].right
                node_stack.append(node)
                check_node_set.add(node)

                if val_stack[-1] <= node.val:
                    # print(node.val)
                    result += 1
                val_stack.append(max(node.val, val_stack[-1]))
                continue

            node_stack.pop()
            val_stack.pop()

        return result + 1

    def pseudoPalindromicPaths(self, root: TreeNode) -> int:
        """
        5418. 二叉树中的伪回文路径
        :param root:
        :return:
        """

        def next_child_node(node: TreeNode, value_set: set) -> int:
            if not node:
                return 0
            a_set = value_set.copy()
            if node.val in a_set:
                a_set.remove(node.val)
            else:
                a_set.add(node.val)

            if not node.left and not node.right:
                return 1 if len(a_set) < 2 else 0

            c = 0
            if node.left:
                c += next_child_node(node.left, a_set)
            if node.right:
                c += next_child_node(node.right, a_set)

            return c

        if not root:
            return 0

        return next_child_node(root, set())

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        """
        543. 二叉树的直径
        :param root:
        :return:
        """

        def depth_of_tree(root: TreeNode) -> (int, int):
            # 求树的深度
            if not root:
                return 0, 0

            left_depth = depth_of_tree(root.left)
            right_depth = depth_of_tree(root.right)

            return max(left_depth[0], right_depth[0]) + 1, max(left_depth[1], right_depth[1],
                                                               left_depth[0] + right_depth[0])

        return depth_of_tree(root)[1] if root else 0

    def robIII(self, root: TreeNode) -> int:
        """
        337. 打家劫舍 III
        :see https://leetcode-cn.com/problems/house-robber-iii/
        """

        def rob_of_subtree(root: TreeNode) -> (int, int):
            if not root:
                return -float('inf'), -float('inf')

            left = rob_of_subtree(root.left)
            right = rob_of_subtree(root.right)

            # print(root.val, left, right)

            # 第一个返回值代表不取根节点时的最大值，第二个返回值代表取根节点时的最大值
            return max(left[0], left[1], 0) + max(right[0], right[1], 0), max(left[0], 0) + max(right[0], 0) + root.val

        return max(rob_of_subtree(root)) if root else 0

    def recoverFromPreorder(self, S: str) -> TreeNode:
        """
        1028. 从先序遍历还原二叉树
        :see https://leetcode-cn.com/problems/recover-a-tree-from-preorder-traversal/
        """
        if not S:
            return None

        # 以 (所处层级, 节点) 的形式存储在栈中，以便确定父节点
        node_stack = []

        # 连续短横线的数量
        dash_line_count = 0
        # 连续的数字
        current_number = 0

        for i in range(len(S)):
            # 过滤字符为短横线的情况
            if S[i] == '-':
                dash_line_count += 1
                continue

            # 拼接连续数字
            current_number = current_number * 10 + int(S[i])

            # 当字符串结束，或者下个字符为短横线时，需要把当前数字作为一个新的节点
            if i == len(S) - 1 or S[i + 1] == '-':
                current_node = TreeNode(current_number)

                # 找到父节点，建立与父节点的关系
                if dash_line_count > 0:
                    # 移除层级比自己大的节点
                    while node_stack[-1][0] > dash_line_count:
                        node_stack.pop()

                    # 若最后一个节点的层级与自己相同，则说明自己是父节点的右子节点
                    if node_stack[-1][0] == dash_line_count:
                        node_stack.pop()
                        node_stack[-1][1].right = current_node
                    else:
                        node_stack[-1][1].left = current_node

                # 将该节点入栈
                node_stack.append((dash_line_count, current_node))

                # 重置变量
                dash_line_count = 0
                current_number = 0

        return node_stack[0][1]

    def postorderTraversal(self, root: TreeNode) -> list:
        """
        145. 二叉树的后序遍历
        :see https://leetcode-cn.com/problems/binary-tree-postorder-traversal/
        """
        if not root:
            return []

        nodes_stack = [root]
        result = []

        while nodes_stack:
            node = nodes_stack[-1]
            if node.left:
                nodes_stack.append(node.left)
                node.left = None
            elif node.right:
                nodes_stack.append(node.right)
                node.right = None
            else:
                result.append(node.val)
                nodes_stack.pop()

        return result

    def zigzagLevelOrder(self, root: TreeNode) -> list:
        """
        103. 二叉树的锯齿形层次遍历
        :see https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/
        """
        if not root:
            return []

        left2right = True
        nodes_stack = [root]
        result = []

        while nodes_stack:
            stack = []
            out = []
            while nodes_stack:
                node = nodes_stack.pop()
                out.append(node.val)

                if left2right:
                    if node.left:
                        stack.append(node.left)
                    if node.right:
                        stack.append(node.right)
                else:
                    if node.right:
                        stack.append(node.right)
                    if node.left:
                        stack.append(node.left)

            left2right = not left2right
            nodes_stack = stack
            result.append(out)

        return result

    def buildTreeII(self, inorder: list, postorder: list) -> TreeNode:
        """
        106. 从中序与后序遍历序列构造二叉树
        :see https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
        """
        if len(inorder) != len(postorder) or not inorder:
            return None

        # 先找到根节点，根节点是后序遍历的最后一位
        root = TreeNode(postorder[-1])

        # 在中序遍历中找到根节点的位置
        root_index = inorder.index(root.val)

        # 将中序遍历按照根节点所在位置分成左子树和右子树
        left_in_order = inorder[:root_index]
        right_in_order = inorder[root_index + 1:]

        # 将后序遍历按照左子树节点数量分为左子树和右子树
        left_post_order = postorder[:root_index]
        right_post_order = postorder[root_index:-1]

        # 递归左子树和右子树
        root.left = self.buildTreeII(left_in_order, left_post_order)
        root.right = self.buildTreeII(right_in_order, right_post_order)

        return root

    def sortedArrayToBST(self, nums: list) -> TreeNode:
        """
        108. 将有序数组转换为二叉搜索树
        :see https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/
        """
        if not nums:
            return None
        mid_index = len(nums) // 2
        root = TreeNode(nums[mid_index])
        root.left = self.sortedArrayToBST(nums[:mid_index])
        root.right = self.sortedArrayToBST(nums[mid_index + 1:])
        return root

    def longestZigZag(self, root: TreeNode) -> int:
        """
        1372. 二叉树中的最长交错路径
        :see https://leetcode-cn.com/problems/longest-zigzag-path-in-a-binary-tree/
        """

        def sub_tree_longest_zig_zag(root_node: TreeNode) -> (int, int):
            # 返回当前节点的左节点和右节点的最长的交错路径
            if not root_node:
                return 0, 0

            # 左节点的最长交错路径 = 左节点的右子节点的最长交错路径 + 1
            left = sub_tree_longest_zig_zag(root_node.left)[1] + 1
            # 右节点的最长交错路径 = 右节点的左子节点的最长交错路径 + 1
            right = sub_tree_longest_zig_zag(root_node.right)[0] + 1

            # 更新最长的交错路径
            nonlocal result
            result = max(result, left, right)

            return left, right

        result = 0
        sub_tree_longest_zig_zag(root)
        # 因为 result 统计的时候路径上节点的数量，边的数量 = 节点数量 - 1
        return result - 1

    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        """
        112. 路径总和
        :see https://leetcode-cn.com/problems/path-sum/
        """

        def backtrace(node: TreeNode, total: int) -> bool:
            if not node:
                return False
            if not node.left and not node.right:
                return total + node.val == sum
            return backtrace(node.left, total + node.val) or backtrace(node.right, total + node.val)

        return backtrace(root, 0)

    def isBalanced(self, root: TreeNode) -> bool:
        """
        110. 平衡二叉树
        :see https://leetcode-cn.com/problems/balanced-binary-tree/
        """

        def height_of_tree(root: TreeNode) -> int:
            if not root:
                return 0
            elif root in height_dict:
                return height_dict[root]
            else:
                height = max(height_of_tree(root.left) + 1, height_of_tree(root.right) + 1)
                height_dict[root] = height
                return height

        def check(root: TreeNode) -> bool:
            if not root:
                return True

            left = height_of_tree(root.left)
            right = height_of_tree(root.right)

            return (check(root.left) and check(root.right)) if abs(left - right) < 2 else False

        height_dict = {}
        return check(root)


if __name__ == '__main__':
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    # # #
    root.left.left = TreeNode(4)
    # root.left.right = TreeNode(5)

    root.left.left.left = TreeNode(8)
    # root.left.left.right = TreeNode(9)

    # #
    # root.left.right.left = TreeNode(10)
    # root.left.right.right = TreeNode(11)
    #
    root.right.left = TreeNode(5)
    root.right.right = TreeNode(6)

    # root.right.left.right = TreeNode(7)

    print(Solution().isBalanced(root))
    # string = Solution().serialize(root)
    # print(string)
    # print(Solution().getSkyline([[1, 2, 1], [1, 2, 2], [1, 2, 3]]))

    # s = Solution()
    # print(s.robIII(root))
    # print(s.zigzagLevelOrder(root))
    # root = s.sortedArrayToBST([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(s.preorderTraversal(root))
