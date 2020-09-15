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


if __name__ == '__main__':
    s = Solution()
    arr = [["5", "3", ".", ".", "7", ".", ".", ".", "."],
           ["6", ".", ".", "1", "9", "5", ".", ".", "."],
           [".", "9", "8", ".", ".", ".", ".", "6", "."],
           ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
           ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
           ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
           [".", "6", ".", ".", ".", ".", "2", "8", "."],
           [".", ".", ".", "4", "1", "9", ".", ".", "5"],
           [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
    s.solveSudoku(arr)
    for i in arr:
        print(i)
