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


if __name__ == '__main__':
    s = Solution()
    arr = [[0, 1], [0, 2], [2, 3], [2, 4], [2, 5]]
    print(s.sumOfDistancesInTree(6, arr))
