from typing import List


class Solution:

    def solveSudoku(self, board: list) -> None:
        """
        37. 解数独
        :see https://leetcode-cn.com/problems/sudoku-solver/
        """

        def check(x: int, y: int) -> bool:
            # 校验行
            checked = set()
            for i in range(9):
                if board[x][i] != '.' and board[x][i] in checked:
                    return False
                checked.add(board[x][i])

            # 校验列
            checked = set()
            for i in range(9):
                if board[i][y] != '.' and board[i][y] in checked:
                    return False
                checked.add(board[i][y])

            # 校验块
            start_x = x // 3 * 3
            start_y = y // 3 * 3
            checked = set()
            for i, j in [(start_x, start_y), (start_x, start_y + 1), (start_x, start_y + 2), (start_x + 1, start_y), (start_x + 1, start_y + 1),
                         (start_x + 1, start_y + 2), (start_x + 2, start_y), (start_x + 2, start_y + 1), (start_x + 2, start_y + 2)]:
                if board[i][j] != '.' and board[i][j] in checked:
                    return False
                checked.add(board[i][j])

            return True

        def backtrace(x: int, y: int) -> bool:
            if x == 9:
                return True

            # 下一个坐标
            next_x = x
            next_y = y + 1
            if next_y == 9:
                next_x += 1
                next_y = 0

            # 如果已经有数字，则不需要尝试
            if board[x][y] != '.':
                return backtrace(next_x, next_y)

            # 从1~9逐个尝试
            result = False
            for i in range(1, 10):
                # 先赋值为i
                board[x][y] = str(i)

                # 检查i是否符合条件，如果符合，则继续尝试下一个坐标
                if check(x, y) and backtrace(next_x, next_y):
                    result = True
                    break

                # 尝试失败时，将坐标的值还原
                board[x][y] = '.'

            return result

        backtrace(0, 0)

    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        """
        787. K 站中转内最便宜的航班
        :see https://leetcode-cn.com/problems/cheapest-flights-within-k-stops/
        """
        # 建立邻接表
        neigh_dict = {}
        for item in flights:
            if item[0] not in neigh_dict:
                neigh_dict[item[0]] = []
            neigh_dict[item[0]].append(item)

        # dp[i][j] 表示最多经过j次中转，最便宜的价格
        dp = [[float('inf')] * (K + 2) for _ in range(n)]

        # 初始化
        for i in range(K + 2):
            dp[src][i] = 0

        queue = [src]
        step = 1
        length = 1
        while step < K + 2 and queue:
            node = queue.pop(0)
            length -= 1
            # 从node开始遍历下一个节点
            if node not in neigh_dict:
                continue

            for item in neigh_dict[node]:
                # 如果node->item[1]的价格更大，就不需要重新遍历这个item[1]
                if dp[item[1]][step] < dp[node][step - 1] + item[2]:
                    continue

                # 将新的节点添加到队列中
                queue.append(item[1])

                # 更新在step次中转内node->item[1]的最便宜价格
                for i in range(step, K + 2):
                    dp[item[1]][i] = min(dp[item[1]][i], dp[node][i - 1] + item[2])

            if length == 0:
                step += 1
                length = len(queue)

            # print(node)
            # for i in dp:
            #     print(i)

        return dp[dst][-1] if dp[dst][-1] != float('inf') else -1

    def sumOfDistancesInTree(self, N: int, edges: List[List[int]]) -> List[int]:
        """
        834. 树中距离之和
        :see https://leetcode-cn.com/problems/sum-of-distances-in-tree/
        """

        def dfs_for_distance(node: int, parent: int, step: int) -> int:
            """ 通过DFS统计距离之和 """
            if node not in neigh_dict:
                return 0

            # i到所有子节点的距离
            sub_dist = 0
            for i in neigh_dict[node]:
                if i != parent:
                    sub_dist += dfs_for_distance(i, node, step + 1)

            return sub_dist + step

        def dfs_for_sub_node_count(node: int, parent: int) -> int:
            """ 通过DFS，统计以parent为父节点，node为根节点所在子树的节点数量之和 """
            if node not in neigh_dict:
                return 0

            sub_node_count = 0
            for i in neigh_dict[node]:
                if i != parent:
                    sub_node_count += dfs_for_sub_node_count(i, node)

            # 以parent为父节点，node为根节点所在子树的节点数量之和
            son_count_dict[node, parent] = sub_node_count + 1
            # 以node为父节点，parent为根节点所在子树的节点数量之和
            son_count_dict[parent, node] = N - sub_node_count - 1

            return sub_node_count + 1

        def dfs_for_all_node_distance(node: int, parent: int):
            """ 通过DFS，统计从parent点的节点距离之和转换为node的节点距离之和 """
            if node not in neigh_dict:
                return

            # 比如说统计节点2，0和2的距离之和转换方式为result[2] = result[0] + son_count_dict[0, 2] - son_count_dict[2, 0]
            if node != parent:
                result[node] = result[parent] + son_count_dict[parent, node] - son_count_dict[node, parent]

            # 递归计算子节点的距离之和
            for i in neigh_dict[node]:
                if i != parent:
                    dfs_for_all_node_distance(i, node)

        # 建立邻接表
        neigh_dict = {}
        for item in edges:
            a, b = item[0], item[1]
            if a in neigh_dict:
                neigh_dict[a].append(b)
            else:
                neigh_dict[a] = [b]

            if b in neigh_dict:
                neigh_dict[b].append(a)
            else:
                neigh_dict[b] = [a]

        result = [0] * N

        # 先把0当成树的根节点，统计节点0与其他所有节点的距离之和
        result[0] = dfs_for_distance(0, 0, 0)

        # 统计当0为根节点时，各个节点的子节点数量
        son_count_dict = {}
        dfs_for_sub_node_count(0, 0)

        # 从0的相邻节点开始，逐个统计每个节点到所有节点的距离之和
        dfs_for_all_node_distance(0, 0)

        return result

    def totalNQueens(self, n: int) -> int:
        """
        52. N皇后 II
        :see https://leetcode-cn.com/problems/n-queens-ii/
        """

        def check(index: int, value: int) -> bool:
            for i in range(index):
                # 不能处于同列：y值不能相同
                if arr[i] == value:
                    return False

                # 不能处于同一左上右下对角线：y-x不能相同
                if arr[i] - i == value - index:
                    return False

                # 不能处于同一右上左下对角线：x + y不能相同
                if arr[i] + i == value + index:
                    return False

            return True

        def backtrace(index: int):
            nonlocal result
            if index == n:
                result += 1
                return

            for i in range(n):
                arr[index] = i
                if check(index, i):
                    backtrace(index + 1)

        if n < 2:
            return n

        result = 0
        arr = [-1] * n
        backtrace(0)
        return result

    def findCriticalAndPseudoCriticalEdges(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        1489. 找到最小生成树里的关键边和伪关键边
        :see https://leetcode-cn.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/
        """
        from LeetCode.Class.UnionFindClass import UnionFindClass

        def generateMST(ignore: int) -> (int, set):
            """
            删除某条边后，生成的最小生成树的权重和
            :param ignore: 删除的边的下标
            :return: 权重和，若为-1，则表示不能生成最小生成树;最小生成树的未使用边的下标
            """
            union = UnionFindClass(n)
            unused = {ignore}
            value = 0

            for i in range(len(graph)):
                item = graph[i]
                if item[3] == ignore:
                    continue

                if union.merge(item[1], item[2]):
                    unused.add(item[3])
                else:
                    value += item[0]

            return (value, unused) if union.get_root_count() == 1 else (-1, None)

        def generateMST2(pre: int) -> int:
            """ 使用某一边，构成最小生成树的权重和 """
            union = UnionFindClass(n)
            union.merge(edges[pre][0], edges[pre][1])
            value = edges[pre][2]

            for i in range(len(graph)):
                item = graph[i]
                if not union.merge(item[1], item[2]):
                    value += item[0]

            return value

        graph = sorted([(edges[i][2], edges[i][0], edges[i][1], i) for i in range(len(edges))])
        # 一直没有被使用的边
        min_val, unused_edges = generateMST(-1)
        unused_edges.remove(-1)
        # 关键边
        key_edges = set()

        for i in range(len(graph)):
            # 删除第i条边后，生成的最小生成树
            v, un = generateMST(i)
            if v == -1 or v > min_val:
                key_edges.add(i)
            else:
                unused_edges &= un

        # 对未出现过的边进行检查，如果这条边加入后，构成的最小生成树的权重和未改变边，则说明这条边是可能出现的边
        delete = set()
        for i in unused_edges:
            v = generateMST2(i)
            if v == min_val:
                delete.add(i)
        unused_edges -= delete

        # 可能出现过的边 = 所有边 - 关键边 - 未出现过的边
        other = {i for i in range(len(edges))} - key_edges - unused_edges

        return [list(key_edges), list(other)]

    def swimInWater(self, grid: List[List[int]]) -> int:
        """
        778. 水位上升的泳池中游泳
        :See: https://leetcode-cn.com/problems/swim-in-rising-water/
        """
        N = len(grid)
        costs = [[float('inf')] * N for i in range(N)]
        costs[0][0] = grid[0][0]

        queue = [(0, 0)]
        while queue:
            x, y = queue.pop(0)
            for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= i < N and 0 <= j < N:
                    cost = max(costs[x][y], grid[i][j])
                    if cost < costs[i][j]:
                        costs[i][j] = cost
                        queue.append((i, j))

        return int(costs[-1][-1])

    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        212. 单词搜索 II
        :see https://leetcode-cn.com/problems/word-search-ii/
        """
        from Class.TrieTree import TrieTree, TrieNode
        tree = TrieTree()
        for i in words:
            tree.insert(i)

        def dfs(x: int, y: int, node: TrieNode, s: List[str], path: set):
            if board[x][y] not in node.children or (x, y) in path:
                return

            path.add((x, y))
            s.append(board[x][y])

            next_node: TrieNode = node.children[board[x][y]]
            if next_node.is_word:
                result.add(''.join(s))

            for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= i < len(board) and 0 <= j < len(board[i]):
                    dfs(i, j, next_node, s, path)

            path.remove((x, y))
            s.pop()

        result = set()
        for i in range(len(board)):
            for j in range(len(board[i])):
                dfs(i, j, tree.root, [], set())

        return list(result)


if __name__ == '__main__':
    s = Solution()
    print(s.findWords([["o", "a", "a", "n"], ["e", "t", "a", "e"], ["i", "h", "k", "r"], ["i", "f", "l", "v"]], ["oath", "pea", "eat", "rain"]))
