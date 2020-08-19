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

    def smallestRange(self, nums: list) -> list:
        """
        632. 最小区间
        :see https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/
        """
        # 解题：思路将所有数组的数字升序放到一个大数组中，然后找到这个大数组中包含所有数组数字的最小区间即可
        # 1. 实现所有数组的合并和排序
        all_nums = []
        for i in range(len(nums)):
            for j in range(len(nums[i])):
                all_nums.append((nums[i][j], i, j))
        all_nums.sort()

        # 2. 使用滑动窗口，找到包含所有数组数组的子数组
        # 包左包右
        left, right = 0, 0
        # 用{数字所在数组: 该数组数字出现的次数}表示是否所有的数组都在这个all_nums[left:right+1]的子数组内
        index_dict = {all_nums[0][1]: 1}

        result = [all_nums[0][0], all_nums[-1][0]]
        while left <= right:
            # 右指针移动末尾时，结束即可
            if right == len(all_nums) - 1:
                break

            # 右指针右移，对应的数组数字出现次数+1
            right += 1
            index_dict[all_nums[right][1]] = index_dict.get(all_nums[right][1], 0) + 1

            # 左指针所在数组的数字若已经出现多次，则左指针右移
            while left < right and index_dict[all_nums[left][1]] > 1:
                index_dict[all_nums[left][1]] -= 1
                left += 1

            # 若此时窗口内的数字已包含所有数组，则更新result
            if len(index_dict) == len(nums) and all_nums[right][0] - all_nums[left][0] < result[1] - result[0]:
                result = [all_nums[left][0], all_nums[right][0]]

        return result

    def countGoodTriplets(self, arr: list, a: int, b: int, c: int) -> int:
        """
        统计好三元组
        :see
        """
        result = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                for k in range(j + 1, len(arr)):
                    if abs(arr[i] - arr[j]) <= a and abs(arr[j] - arr[k]) <= b and abs(arr[i] - arr[k]) <= c:
                        result += 1
        return result

    def getWinner(self, arr: list, k: int) -> int:
        """
        找出数组游戏的赢家
        :see
        """
        times = 0
        index = 0
        for i in range(1, len(arr)):
            if arr[i] < arr[index]:
                times += 1
            else:
                times = 1
                index = i

            if times == k:
                return arr[index]

        return arr[index]

    def maxSum(self, nums1: list, nums2: list) -> int:
        """
        最大得分
        :see
        """
        num1_dict = {}
        for i in range(len(nums1)):
            num1_dict[nums1[i]] = i

        nums2_dict = {}
        for i in range(len(nums2)):
            nums2_dict[nums2[i]] = i

        index1 = 0
        num1_s = 0
        index2 = 0
        num2_s = 0
        while index1 < len(nums1) or index2 < len(nums2):
            if index1 == len(nums1):
                num2_s += nums2[index2]
                index2 += 1
            elif index2 == len(nums2):
                num1_s += nums1[index1]
                index1 += 1
            elif nums1[index1] == nums2[index2]:
                r = nums1[index1] + max(num1_s, num2_s)
                num1_s = r
                num2_s = r
                index1 += 1
                index2 += 1
            elif nums1[index1] < nums2[index2]:
                num1_s += nums1[index1]
                index1 += 1
            else:
                num2_s += nums2[index2]
                index2 += 1

        return max(num1_s, num2_s) % (pow(10, 9) + 7)

    def minSwaps(self, grid: list) -> int:
        """
        排布二进制网格的最少交换次数
        :see
        """
        one_list = [0] * len(grid)
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    one_list[i] = j

            if one_list[i] == len(grid[i]):
                return -1

        # print(one_list)

        count = 0
        index = 0
        while index < len(one_list):
            if one_list[index] > index:
                tmp = index
                for i in range(index + 1, len(one_list)):
                    if one_list[i] <= index:
                        tmp = i
                        break
                # print(tmp)
                if tmp == index:
                    return -1
                count += tmp - index
                a = one_list.pop(tmp)
                one_list.insert(index, a)
            # print(index, one_list)

            index += 1

        return count

    def maxNonOverlapping(self, nums: list, target: int) -> int:
        """
        5471. 和为目标值的最大数目不重叠非空子数组数目
        :see https://leetcode-cn.com/problems/maximum-number-of-non-overlapping-subarrays-with-sum-equals-target/
        """
        prefix_sum = 0
        prefix_index_dict = {0: 0}
        dp = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_sum += nums[i]
            dp[i + 1] = dp[i]
            if prefix_sum - target in prefix_index_dict:
                index = prefix_index_dict[prefix_sum - target]
                dp[i + 1] = max(dp[i + 1], dp[index] + 1)
            prefix_index_dict[prefix_sum] = i + 1
        # print(prefix_index_dict)
        # print(dp)

        return dp[-1]

    def findKthPositive(self, arr: list, k: int) -> int:
        """
        5468. 第 k 个缺失的正整数
        :see https://leetcode-cn.com/problems/kth-missing-positive-number/
        """
        index = 1
        for i in range(len(arr)):
            while arr[i] != index:
                k -= 1
                if k == 0:
                    return index
                index += 1
            index += 1

        return index + k - 1 if index > 1 else index + k

    def maxDistance(self, position: list, m: int) -> int:
        """
        1552. 两球之间的磁力
        :see https://leetcode-cn.com/problems/magnetic-force-between-two-balls/
        """
        # 对position进行排序
        position.sort()

        low = 0
        high = position[-1]
        print('->', low, high)
        while low < high:
            # 使用二分法开始测试最多可以放几个球
            mid = (low + high) // 2

            # 以mid作为间距，来推测可以放多少个球
            count = 1
            old = position[0]
            for i in range(1, len(position)):
                if position[i] - old >= mid:
                    count += 1
                    old = position[i]

                if count > m:
                    break

            # print(mid, count)

            # 根据可放的count数量，来改变间距。可放的球数量count大于等于m，说明间距过小，应该增大间距，反之则减少
            if count >= m:
                low = mid + 1
            else:
                high = mid

            # print('->', low, high)

        return low - 1


if __name__ == '__main__':
    s = Solution()
    print(s.maxDistance(
        [269826447, 974181916, 225871443, 189215924, 664652743, 592895362, 754562271, 335067223, 996014894, 510353008, 48640772, 228945137], 3))
