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


if __name__ == '__main__':
    s = Solution()
    arr = [[3, 4, 7], [6, 2, 2], [0, 2, 7], [0, 1, 2], [1, 7, 8], [4, 5, 2], [0, 3, 2], [7, 0, 6], [3, 2, 7], [1, 3, 10], [1, 5, 1], [4, 1, 6],
           [4, 7, 5], [5, 7, 10]]
    print(s.findCheapestPrice(8, arr, 4, 3, 7))
