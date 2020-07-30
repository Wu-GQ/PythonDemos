class Solution:

    def findDiagonalOrder(self, matrix: list) -> list:
        """
        498. 对角线遍历
        :see https://leetcode-cn.com/problems/diagonal-traverse/
        """
        if not matrix or not matrix[0]:
            return []
        m, n = len(matrix), len(matrix[0])
        result = []
        for i in range(m + n - 1):
            if i & 1 == 0:
                for j in range(min(i, m - 1), max(i - n, -1), -1):
                    result.append(matrix[j][i - j])
            else:
                for j in range(max(i - n + 1, 0), min(i + 1, m)):
                    result.append(matrix[j][i - j])
        return result

    def findMin(self, nums: list) -> int:
        """
        154. 寻找旋转排序数组中的最小值 II
        :see https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/
        """
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            print(l, mid, r, nums[l], nums[mid], nums[r])
            if nums[l] <= nums[mid] < nums[r] or nums[l] < nums[mid] <= nums[r]:
                break
            elif nums[mid] == nums[r] or nums[l] == nums[r]:
                r -= 1
            elif nums[l] == nums[mid]:
                l += 1
            elif nums[r] < nums[l] < nums[mid] or nums[r] < nums[l] < nums[mid]:
                l = mid
            elif nums[mid] < nums[r] < nums[l] or nums[mid] < nums[r] < nums[l]:
                r = mid
        return nums[l]

    def numOfSubarrays(self, arr: list) -> int:
        """
        5457. 和为奇数的子数组数目
        :param arr:
        :return:
        """
        # 记录从第一个数字开始的奇数和出现的次数
        count = 0
        # 记录从第一个数字开始的和
        total = 0

        result = 0
        for i in range(len(arr)):
            total += arr[i]
            # 当sum[:i + 1]为奇数时，需要加上arr[:i]的奇数和出现的次数，即i - count + 1
            # 当sum[:i + 1]为偶数时，需要加上arr[:i]的偶数和出现的次数，即count
            if total % 2 == 1:
                result += i - count + 1
                count += 1
            else:
                result += count
            # print(i, total, result, count)

        return result % 1000000007

    def minNumberOperations(self, target: list) -> int:
        """
        5459. 形成目标数组的子数组最少增加次数
        :param target:
        :return:
        """
        target.append(0)
        stack = [0]
        result = 0
        for i in range(len(target)):
            top = stack[-1]
            while stack and target[i] <= stack[-1]:
                stack.pop()
            result += max(top - target[i], 0)
            stack.append(target[i])
            # print(i, result, stack)
        return result

    def searchRange(self, nums: list, target: int) -> list:
        """
        34. 在排序数组中查找元素的第一个和最后一个位置
        :see https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/
        """
        # 仿bisect_left找左边界
        left1, right1 = 0, len(nums)
        while left1 < right1:
            mid = left1 + (right1 - left1) // 2
            if nums[mid] < target:
                left1 = mid + 1
            else:
                right1 = mid

        # 仿bisect_right找右边界
        left2, right2 = 0, len(nums)
        while left2 < right2:
            mid = left2 + (right2 - left2) // 2
            if nums[mid] <= target:
                left2 = mid + 1
            else:
                right2 = mid

        return [left1, left2 - 1] if left1 < left2 else [-1, -1]

    def isValidSudoku(self, board: list) -> bool:
        """
        36. 有效的数独
        :see https://leetcode-cn.com/problems/valid-sudoku/
        """

        def check(nums: list) -> bool:
            nums.sort()
            for i in range(1, len(nums)):
                if nums[i] == nums[i - 1] and nums[i] != '.':
                    # print(nums)
                    return False
            return True

        # 校验每行
        for row in board:
            if not check(row[:]):
                return False

        # 校验每列
        for j in range(9):
            if not check([board[i][j] for i in range(9)]):
                return False

        # 校验每小格
        for i in range(9):
            x = i // 3 * 3 + 1
            y = (3 * i + 1) % 9
            # print(x, y)
            if not check([board[x - 1][y - 1], board[x - 1][y], board[x - 1][y + 1],
                          board[x][y - 1], board[x][y], board[x][y + 1],
                          board[x + 1][y - 1], board[x + 1][y], board[x + 1][y + 1]]):
                return False

        return True

    def combinationSum(self, candidates: list, target: int) -> list:
        """
        39. 组合总和
        :see https://leetcode-cn.com/problems/combination-sum/
        """

        def backtrace(index: int, total: int, nums: list):
            if total == target:
                result.append(nums[:])
                return
            elif total > target:
                return

            for i in range(index, len(candidates)):
                nums.append(candidates[i])
                backtrace(i, total + candidates[i], nums)
                nums.pop()

        result = []
        backtrace(0, 0, [])
        return result

    def combinationSum2(self, candidates: list, target: int) -> list:
        """
        40. 组合总和 II
        :see https://leetcode-cn.com/problems/combination-sum-ii/
        """

        def backtrace(index, total: int, nums: list):
            # 终止条件
            if total == target:
                result.append(nums[:])
                return
            elif total > target:
                return

            for i in range(index, len(candidates)):
                # 去重
                if i > index and candidates[i] == candidates[i - 1]:
                    continue
                # 先加入这个值
                nums.append(candidates[i])
                # 回溯入口。注意，此处 i + 1 表示不能重复利用同一个数字，这是跟第39题的核心差别
                backtrace(i + 1, total + candidates[i], nums)
                # 这个值的回溯结束后，把这个值移除
                nums.pop()

        # 排序，以便回溯中的去重操作
        candidates.sort()
        result = []
        backtrace(0, 0, [])
        return result


if __name__ == '__main__':
    s = Solution()
    print(s.combinationSum2([2, 5, 2, 1, 2], 5))
