import bisect
import re


class Solution:
    def longest_substring(self, s: str, k: int) -> int:
        """
        至少有K个重复字符的最长子串
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/272/dynamic-programming/1174/
        """
        # 统计字符出现的次数
        count_dict = {}
        for i in s:
            if i in count_dict:
                count_dict[i] += 1
            else:
                count_dict[i] = 1

        # 统计不符合数量要求的字符
        mismatched_char_list = []
        for i in count_dict:
            if count_dict[i] < k:
                mismatched_char_list.append(i)

        # 满足条件时返回
        if len(mismatched_char_list) == 0:
            return len(s)

        # 按照不符合数量要求的字符，对源字符串进行分割
        divided_string_list = re.split('|'.join(mismatched_char_list), s)

        # 求各个子串的最大满足条件的最长子串长度
        # max_length = 0
        # for i in divided_string_list:
        #     length = longest_substring(i, k)
        #     if length > max_length:
        #         max_length = length
        # return max_length
        return max(self.longest_substring(string, k) for string in divided_string_list)

    # 存储完全平方数的计算结果
    _count_list = [0]

    def num_squares(self, n: int) -> int:
        """
        完全平方数
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/272/dynamic-programming/1178/
        """
        length = len(self._count_list)

        if length > n:
            return self._count_list[n]

        for i in range(length, n + 1):
            j = 1
            self._count_list.append(i)
            while i - j * j >= 0:
                self._count_list[i] = min(self._count_list[i], self._count_list[i - j * j] + 1)
                j += 1

        return self._count_list[n]

    def lengthOfLIS(self, nums: list) -> int:
        """
        300.最长上升子序列
        :see https://leetcode-cn.com/problems/longest-increasing-subsequence/
        """
        # count_list = [1] * len(nums)
        #
        # max_length = 0
        #
        # for i in range(len(nums)):
        #     for j in range(i):
        #         if nums[j] < nums[i]:
        #             count_list[i] = max(count_list[i], count_list[j] + 1)
        #
        #     if count_list[i] > max_length:
        #         max_length = count_list[i]
        #
        # return max_length
        if len(nums) < 2:
            return len(nums)

        tail = []
        for num in nums:
            # 找到小于 num 的第 1 个数
            index = bisect.bisect_left(tail, num)

            if index == len(tail):
                tail.append(num)
            else:
                tail[index] = num

            print(tail)
        return len(tail)

    def coinChange(self, coins: list, amount: int) -> int:
        """
        322.零钱兑换
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/272/dynamic-programming/1180/
        """
        amount_list = [0]

        for i in range(1, amount + 1):
            amount = float('inf')
            for coin in coins:
                if i >= coin:
                    amount = min(amount, amount_list[i - coin] + 1)
            amount_list.append(amount)
        print(amount_list)
        return amount_list[-1] if amount_list[-1] != float('inf') else -1

    def rob(self, nums: list) -> int:
        """
        打家劫舍
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/272/dynamic-programming/1177/
        """
        # dp[i] = nums[i] + max(dp[i-2], dp[i-3])
        # i < 2, dp[i] = nums[i]
        # i == 2, dp[i] = dp[0] + nums[2]

        money_list = [0] * len(nums)
        max_money = 0

        for i in range(len(nums)):
            if i < 2:
                money_list[i] = nums[i]
            elif i == 2:
                money_list[i] = nums[0] + nums[2]
            else:
                money_list[i] = max(money_list[i - 3], money_list[i - 2]) + nums[i]

            max_money = max(max_money, money_list[i])

        return max_money

    def climbStairs(self, n):
        """
        爬楼梯
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/23/dynamic-programming/54/
        """
        _stairs_array = [0, 1, 2]
        for i in range(3, n + 1):
            _stairs_array.append(_stairs_array[i - 2] + _stairs_array[i - 1])
        return _stairs_array[n]

    def maxProfit2(self, prices: list) -> int:
        """
        买卖股票的最佳时机 II
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/1/array/22/
        """
        profit = 0

        i = 0
        while i < len(prices) - 1:
            low = prices[i]
            high = prices[i + 1]

            if low >= high:
                i += 1
            else:
                # 找到极大值
                j = i + 1
                while j < len(prices):
                    if prices[j] >= high:
                        high = prices[j]
                        j += 1
                    else:
                        break

                # 计算利润
                profit += high - low

                # 更新下标
                i = j

        return profit

    def maxProfit(self, prices: list) -> int:
        """
        买卖股票的最佳时机
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/23/dynamic-programming/55/
        """
        low = float('inf')
        max_profit = 0
        for i in prices:
            if i < low:
                low = i
            else:
                max_profit = max(max_profit, i - low)
        return max_profit

    def maxArea(self, height: list) -> int:
        """
        11. 盛水最多的容器，S = max((j-i) * min(a[i], a[j]))
        :see https://leetcode-cn.com/problems/container-with-most-water/
        """
        if len(height) < 2:
            return 0

        # 使用双指针法，从两端向内缩进，因为缩进时(j-i)的值必然减小，min(a[i], a[j])的值如果减小或不变，则必然不符合“最多”的要求，
        # 因此，每次向内缩进的原则是，选择较小的边，向内缩进
        i = 0
        j = len(height) - 1
        max_area = 0
        while i < j:
            max_area = max(max_area, (j - i) * min(height[i], height[j]))
            if height[i] > height[j]:
                j -= 1
            else:
                i += 1

        return max_area

    def maxSubArray(self, nums: list) -> int:
        """
        最大子序和
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/23/dynamic-programming/56/
        """
        # 状态转移方程 S[n] = max(S[n-1] + a[n], a[n])
        if len(nums) < 1:
            return 0

        # 整个数组的最大子序和
        max_sum = nums[0]
        # 某个子数组的最大子序和
        sums = nums[0]
        for i in range(1, len(nums)):
            sums = max(sums + nums[i], nums[i])
            max_sum = max(sums, max_sum)

            # print(nums[i], max_sum, sums)

        return max_sum

    def maxProduct(self, nums: list) -> int:
        """
        乘积最大子序列
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/264/array/1126/
        """
        # 状态转移方程 S[n] = max(Smax[n-1] * a[n], Smin[n-1] * a[n], S[n-1])
        if len(nums) < 1:
            return 0

        max_value = nums[0]
        min_value = nums[0]
        max_result = nums[0]

        for i in range(1, len(nums)):
            if nums[i] >= 0:
                max_value, min_value = max(max_value * nums[i], nums[i]), min(min_value * nums[i], nums[i])
            else:
                max_value, min_value = max(min_value * nums[i], nums[i]), min(max_value * nums[i], nums[i])
            max_result = max(max_result, max_value, min_value)
            # print(f'{nums[i]}: {max_value}, {min_value}, {max_result}')

        return max_result

    def uniquePaths(self, m: int, n: int) -> int:
        """
        不同路径
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-medium/51/dynamic-programming/105/
        """
        # 实际上的求组合问题 C(m - 1, m + n - 2)
        min_num, max_num = (m - 1, n - 1) if m < n else (n - 1, m - 1)
        print(min_num, max_num)
        return 1 if min_num == 0 else int(
            self.consequentMultiple(max_num + 1, min_num + max_num) / self.consequentMultiple(1, min_num))

    def consequentMultiple(self, start: int, end: int) -> int:
        """ 连乘 """
        result = start
        for i in range(start + 1, end + 1):
            result *= i
        return result

    def minDistance(self, word1: str, word2: str) -> int:
        """
        72.编辑距离
        :see https://leetcode-cn.com/problems/edit-distance/
        """
        """
        distance_list = []

        m = len(word1) + 1
        n = len(word2) + 1
        for i in range(m * n):
            x = i % m
            y = i // m

            if x == 0:
                distance_list.append(y)
                continue
            if y == 0:
                distance_list.append(x)
                continue

            distance_list.append(min(distance_list[i - m] + 1,
                                     distance_list[i - 1] + 1,
                                     distance_list[i - m - 1] + (0 if word1[x - 1] == word2[y - 1] else 1)))

        return distance_list[m * n - 1]
        """

        word1 = ' ' + word1
        word2 = ' ' + word2

        distance_list = [[0] * len(word2) for _ in word1]

        for i in range(len(word1)):
            for j in range(len(word2)):
                if i == 0:
                    distance_list[i][j] = j
                elif j == 0:
                    distance_list[i][j] = i
                else:
                    distance_list[i][j] = min(distance_list[i - 1][j] + 1,
                                              distance_list[i][j - 1] + 1,
                                              distance_list[i - 1][j - 1] + (1 if word1[i] != word2[j] else 0))

        for i in distance_list:
            print(i)

        return distance_list[-1][-1]

    def longestPalindrome(self, s: str) -> str:
        """
        最长回文子串
        :see https://leetcode-cn.com/problems/longest-palindromic-substring/
        """
        length = len(s)
        if length == 0:
            return ""

        def palindrome_string(start: int, end: int) -> (int, int):
            """ 返回将回文串两边扩展后最长的回文串的左下标和右下标 """
            if start < 0 or end >= length or s[start] != s[end]:
                return start, start

            for i in range(length):
                left_index = start - i
                right_index = end + i

                if left_index < 0 or right_index >= length or s[left_index] != s[right_index]:
                    return left_index + 1, right_index - 1

            return start, start

        start_index: int = 0
        end_index: int = 0

        for i in range(0, length):
            single = palindrome_string(i, i)
            double = palindrome_string(i, i + 1)

            if single[1] - single[0] > end_index - start_index:
                start_index = single[0]
                end_index = single[1]
            if double[1] - double[0] > end_index - start_index:
                start_index = double[0]
                end_index = double[1]

        return s[start_index:end_index + 1]

    def longestValidParentheses(self, s: str) -> int:
        """
        最长有效括号
        :see https://leetcode-cn.com/problems/longest-valid-parentheses/
        """
        # 有效长度数组
        valid_length_list = []
        # 当前的有效（数量
        parentheses_count = 0
        # 最大长度
        max_length = 0

        for i in range(0, len(s)):
            if s[i] == '(':
                parentheses_count += 1
                valid_length_list.append(0)
            elif parentheses_count > 0:
                parentheses_count -= 1
                if i - valid_length_list[i - 1] - 2 >= 0:
                    valid_length_list.append(
                        valid_length_list[i - 1] + valid_length_list[i - valid_length_list[i - 1] - 2] + 2)
                else:
                    valid_length_list.append(valid_length_list[i - 1] + 2)
            else:
                valid_length_list.append(0)

            if valid_length_list[i] > max_length:
                max_length = valid_length_list[i]
        # print(valid_length_list)
        return max_length

    def largestRectangleArea(self, heights: list) -> int:
        """
        柱状图中最大的矩形
        :see https://leetcode-cn.com/problems/largest-rectangle-in-histogram/
        """
        length = len(heights)
        if length < 1:
            return 0

        max_area = 0
        height_stack = [(-1, -1)]

        for i in range(0, length):
            while heights[i] < height_stack[-1][1]:
                index_height = height_stack.pop()
                max_area = max(max_area, index_height[1] * (i - height_stack[-1][0] - 1))

            height_stack.append((i, heights[i]))

        while len(height_stack) > 1:
            index_height = height_stack.pop()
            max_area = max(max_area, index_height[1] * (length - height_stack[-1][0] - 1))

        return max_area

    def maximalRectangle(self, matrix: list) -> int:
        """
        最大矩形
        :see https://leetcode-cn.com/problems/maximal-rectangle/
        """
        # 参考柱状图中最大的矩形
        matrix_height = len(matrix)
        if matrix_height < 1:
            return 0
        matrix_width = len(matrix[0])
        if matrix_width < 1:
            return 0

        max_area = 0

        for row in range(0, matrix_height):
            # 将每一行看成柱状图中最大的矩形求解
            index_stack_list = [-1]
            height_stack_list = [-1]

            for column in range(0, matrix_width):
                height = int(matrix[row][column])
                if height > 0 and row > 0:
                    height += matrix[row - 1][column]
                matrix[row][column] = height

                while height < height_stack_list[-1]:
                    index_stack_list.pop()
                    last_height = height_stack_list.pop()
                    max_area = max(max_area, last_height * (column - index_stack_list[-1] - 1))

                index_stack_list.append(column)
                height_stack_list.append(height)

            while len(height_stack_list) > 1:
                index_stack_list.pop()
                last_height = height_stack_list.pop()
                max_area = max(max_area, last_height * (matrix_width - index_stack_list[-1] - 1))

        return max_area

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        """
        交错字符串
        :see https://leetcode-cn.com/problems/interleaving-string/
        """
        if len(s3) != len(s1) + len(s2):
            return False

        result_list = [[False] * (len(s1) + 1) for i in s2 + " "]

        for row in range(0, len(s2) + 1):
            for column in range(0, len(s1) + 1):
                if row == 0 and column == 0:
                    result_list[0][0] = True
                elif row == 0:
                    result_list[0][column] = result_list[0][column - 1] and s1[column - 1] == s3[column - 1]
                elif column == 0:
                    result_list[row][0] = result_list[row - 1][0] and s2[row - 1] == s3[row - 1]
                else:
                    result_list[row][column] = (result_list[row - 1][column] and s2[row - 1] == s3[
                        row - 1 + column]) or (
                                                       result_list[row][column - 1] and s1[column - 1] == s3[
                                                   column - 1 + row])

        return result_list[len(s2)][len(s1)]

    def numDistinct(self, s: str, t: str) -> int:
        """
        不同的子序列
        :see https://leetcode-cn.com/problems/distinct-subsequences/
        """
        if len(s) < len(t):
            return 0

        result_list = [[0] * len(s) for i in t]

        for row in range(0, len(t)):
            for column in range(row, len(s)):
                if s[column] == t[row]:
                    result_list[row][column] = result_list[row][column - 1] + (
                        result_list[row - 1][column - 1] if row > 0 else 1)
                else:
                    result_list[row][column] = result_list[row][column - 1]

        return result_list[-1][-1]

    def maxProfit(self, prices: list) -> int:
        """
        买卖股票的最佳时机
        :see https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/
        """
        # 状态转移方程：i代表第几天；j代表买卖的次数；0代表无持仓，1代表有持仓
        # 卖出: f[i][j][0] = max(f[i - 1][j][0], f[i - 1][j][1] + prices[i])
        # 买入: f[i][j][1] = max(f[i - 1][j][1], f[i - 1][j - 1][0] - prices[i])
        """
        if len(prices) < 2:
            return 0

        result_list = [[], []]

        length = len(prices)
        for i in range(0, length):
            if i == 0:
                result_list[0].append((0, -float('INF')))
                result_list[1].append((-float('INF'), -prices[i]))
            else:
                result_list[0].append((max(result_list[0][i - 1][0], result_list[0][i - 1][1] + prices[i]), result_list[0][i - 1][1]))
                result_list[1].append((max(result_list[1][i - 1][0], result_list[1][i - 1][1] + prices[i]),
                                       max(result_list[1][i - 1][1], result_list[0][i - 1][0] - prices[i])))
        
        return max(max(result_list[0][length - 1]), max(result_list[1][length - 1]))
        """
        if len(prices) < 2:
            return 0
        max_profit, min_price = 0, float('inf')
        for i in prices:
            max_profit = max(max_profit, i - min_price)
            min_price = min(min_price, i)
            # print(i, min_price, max_profit)
        return max_profit

    def maxProfit3(self, prices: list) -> int:
        """
        买卖股票的最佳时机 III
        :see https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/
        """
        """
        状态转移方程:
        f(N天，K交易次数，持有) = max(f(N-1天，K交易次数，持有), f(N-1天，K-1交易次数，无)-当日价格)
        f(N天，K交易次数，无) = max(f(N-1天，K交易次数，无)，f(N-1，K交易次数，持有)+当日价格)
        """
        length = len(prices)
        if length < 2:
            return 0

        # 持有，交易1次
        possess_one_profit = -float('inf')
        # 持有，交易2次
        possess_two_profit = -float('inf')
        # 无，交易1次
        none_one_profit = -float('inf')
        # 无，交易2次
        none_two_profit = -float('inf')

        # 动态规划
        for i in range(0, length):
            if i == 0:
                possess_one_profit, possess_two_profit, none_one_profit, none_two_profit_list = -prices[i], -float(
                    'inf'), 0, -float('inf')
            else:
                possess_one_profit, possess_two_profit, none_one_profit, none_two_profit = \
                    max(possess_one_profit, -prices[i]), \
                    max(possess_two_profit, none_one_profit - prices[i]) if i > 1 else -float('inf'), \
                    max(none_one_profit, possess_one_profit + prices[i]), \
                    max(none_two_profit, possess_two_profit + prices[i]) if i > 2 else -float('inf')

        return max(none_one_profit, none_two_profit)

    def uniquePathsWithObstacles(self, obstacleGrid: list) -> int:
        """
        63. 不同路径 II
        :see https://leetcode-cn.com/problems/unique-paths-ii/
        """
        # f(x,y) = f(x-1,y)+f(x,y-1) if a(x,y)==0 else 0
        if len(obstacleGrid) == 0 or len(obstacleGrid[0]) == 0 or obstacleGrid[0][0] == 1:
            return 0

        path_list = [0] * len(obstacleGrid[0])
        path_list[0] = 1

        for single_line in obstacleGrid:
            for i in range(0, len(single_line)):
                if single_line[i] == 1:
                    path_list[i] = 0
                elif i > 0:
                    path_list[i] = path_list[i - 1] + path_list[i]

        return path_list[-1]

    def minPathSum(self, grid: list) -> int:
        """
        64. 最小路径和
        :see https://leetcode-cn.com/problems/minimum-path-sum/
        """
        if len(grid) == 0 or len(grid[0]) == 0:
            return 0

        for i in range(0, len(grid)):
            for j in range(0, len(grid[i])):
                if i == 0 and j == 0:
                    continue
                elif i == 0 and j > 0:
                    grid[i][j] += grid[i][j - 1]
                elif i > 0 and j == 0:
                    grid[i][j] += grid[i - 1][j]
                else:
                    grid[i][j] += min(grid[i][j - 1], grid[i - 1][j])

        return grid[-1][-1]

    def largestMultipleOfThree(self, digits: list) -> str:
        """
        1363. 形成三的最大倍数
        :see https://leetcode-cn.com/problems/largest-multiple-of-three/
        """

        # 状态: f(余0)，f(余1)，f(余2)
        def compare_nums(a: list, b: list) -> bool:
            for i in range(9, -1, -1):
                if a[i] > b[i]:
                    return True
                elif a[i] < b[i]:
                    return False
            return True

        zero_nums, one_nums, two_nums = [0] * 10, [0] * 10, [0] * 10
        zero_count, one_count, two_count = 0, 0, 0
        zero_sum, one_sum, two_sum = 0, 0, 0

        for digit in digits:
            remainder = digit % 3
            if remainder == 0:
                zero_nums[digit] += 1
                zero_count += 1
                zero_sum += digit

                one_nums[digit] += 1
                one_count += 1
                one_sum += digit

                two_nums[digit] += 1
                two_count += 1
                two_sum += digit
            elif remainder == 1:
                if two_sum % 3 == 2:
                    temp_zero_nums = two_nums.copy()
                    temp_zero_nums[digit] += 1
                    temp_zero_count = two_count + 1
                    temp_zero_sum = two_sum + digit

                    if temp_zero_count < zero_count or (
                            temp_zero_count == zero_count and compare_nums(zero_nums, temp_zero_nums)):
                        temp_zero_nums = zero_nums
                        temp_zero_count = zero_count
                        temp_zero_sum = zero_sum
                else:
                    temp_zero_nums = zero_nums
                    temp_zero_count = zero_count
                    temp_zero_sum = zero_sum

                if zero_sum % 3 == 0:
                    temp_one_nums = zero_nums.copy()
                    temp_one_nums[digit] += 1
                    temp_one_count = zero_count + 1
                    temp_one_sum = zero_sum + digit

                    if temp_one_count < one_count or (
                            temp_one_count == one_count and compare_nums(one_nums, temp_one_nums)):
                        temp_one_nums = one_nums
                        temp_one_count = one_count
                        temp_one_sum = one_sum
                else:
                    temp_one_nums = one_nums
                    temp_one_count = one_count
                    temp_one_sum = one_sum

                if one_sum % 3 == 1:
                    temp_two_nums = one_nums.copy()
                    temp_two_nums[digit] += 1
                    temp_two_count = one_count + 1
                    temp_two_sum = one_sum + digit

                    if temp_two_count < two_count or (
                            temp_two_count == two_count and compare_nums(two_nums, temp_two_nums)):
                        temp_two_nums = two_nums
                        temp_two_count = two_count
                        temp_two_sum = two_sum
                else:
                    temp_two_nums = two_nums
                    temp_two_count = two_count
                    temp_two_sum = two_sum

                zero_nums, one_nums, two_nums = temp_zero_nums, temp_one_nums, temp_two_nums
                zero_count, one_count, two_count = temp_zero_count, temp_one_count, temp_two_count
                zero_sum, one_sum, two_sum = temp_zero_sum, temp_one_sum, temp_two_sum
            else:
                if one_sum % 3 == 1:
                    temp_zero_nums = one_nums.copy()
                    temp_zero_nums[digit] += 1
                    temp_zero_count = one_count + 1
                    temp_zero_sum = one_sum + digit

                    if temp_zero_count < zero_count or (
                            temp_zero_count == zero_count and compare_nums(zero_nums, temp_zero_nums)):
                        temp_zero_nums = zero_nums
                        temp_zero_count = zero_count
                        temp_zero_sum = zero_sum
                else:
                    temp_zero_nums = zero_nums
                    temp_zero_count = zero_count
                    temp_zero_sum = zero_sum

                if two_sum % 3 == 2:
                    temp_one_nums = two_nums.copy()
                    temp_one_nums[digit] += 1
                    temp_one_count = two_count + 1
                    temp_one_sum = two_sum + digit

                    if temp_one_count < one_count or (
                            temp_one_count == one_count and compare_nums(one_nums, temp_one_nums)):
                        temp_one_nums = one_nums
                        temp_one_count = one_count
                        temp_one_sum = one_sum
                else:
                    temp_one_nums = one_nums
                    temp_one_count = one_count
                    temp_one_sum = one_sum

                if zero_sum % 3 == 0:
                    temp_two_nums = zero_nums.copy()
                    temp_two_nums[digit] += 1
                    temp_two_count = zero_count + 1
                    temp_two_sum = zero_sum + digit

                    if temp_two_count < two_count or (
                            temp_two_count == two_count and compare_nums(two_nums, temp_two_nums)):
                        temp_two_nums = two_nums
                        temp_two_count = two_count
                        temp_two_sum = two_sum
                else:
                    temp_two_nums = two_nums
                    temp_two_count = two_count
                    temp_two_sum = two_sum

                zero_nums, one_nums, two_nums = temp_zero_nums, temp_one_nums, temp_two_nums
                zero_count, one_count, two_count = temp_zero_count, temp_one_count, temp_two_count
                zero_sum, one_sum, two_sum = temp_zero_sum, temp_one_sum, temp_two_sum

            # print(zero_count, one_count, two_count)
            # print(digit, zero_nums, one_nums, two_nums)

        if zero_sum == 0 and zero_count == 0:
            return ""

        string = f"{''.join([str(9)] * zero_nums[9])}{''.join([str(8)] * zero_nums[8])}{''.join([str(7)] * zero_nums[7])}" \
                 f"{''.join([str(6)] * zero_nums[6])}{''.join([str(5)] * zero_nums[5])}{''.join([str(4)] * zero_nums[4])}" \
                 f"{''.join([str(3)] * zero_nums[3])}{''.join([str(2)] * zero_nums[2])}{''.join([str(1)] * zero_nums[1])}" \
                 f"{''.join([str(0)] * zero_nums[0])}"

        return "0" if string[0] == '0' else string

    def minimumTotal(self, triangle: list) -> int:
        """
        120. 三角形最小路径和
        :see https://leetcode-cn.com/problems/triangle/
        """
        if len(triangle) == 0 or len(triangle[0]) == 0:
            return 0

        for i in range(1, len(triangle)):
            for j in range(0, i + 1):
                if j == 0:
                    triangle[i][j] += triangle[i - 1][j]
                elif j == i:
                    triangle[i][j] += triangle[i - 1][j - 1]
                else:
                    triangle[i][j] += min(triangle[i - 1][j - 1], triangle[i - 1][j])

        return min(triangle[-1])

    def numDecodings(self, s: str) -> int:
        """
        91. 解码方法
        :see https://leetcode-cn.com/problems/decode-ways/
        """
        # f(n) = f(n - 1) + f(n - 2), 1-9, 10-16
        if len(s) < 1 or s[0] == '0':
            return 0

        result_list = [0] * len(s)

        for i in range(0, len(s)):
            if i == 0:
                result_list[i] = 1
            elif i == 1:
                if s[i] == '0':
                    if s[i - 1] == '1' or s[i - 1] == '2':
                        result_list[i] = 1
                    else:
                        return 0
                else:
                    result_list[i] = 2 if 9 < int(s[i - 1:i + 1]) < 27 else 1
            elif s[i] == '0':
                if s[i - 1] == '1' or s[i - 1] == '2':
                    result_list[i] = result_list[i - 2]
                else:
                    return 0
            else:
                result_list[i] = result_list[i - 1] + (result_list[i - 2] if 9 < int(s[i - 1:i + 1]) < 27 else 0)
        print(result_list)
        return result_list[-1]

    def isMatch(self, s: str, p: str) -> bool:
        """
        面试题19. 正则表达式匹配
        :see https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/
        """
        """
        # 回溯递归算法
        # 执行用时 : 1968 ms , 在所有 Python3 提交中击败了 8.91% 的用户
        # 内存消耗 : 13.5 MB , 在所有 Python3 提交中击败了 100.00% 的用户
        if len(s) == 0 and len(p) == 0:
            return True
        elif len(s) > 0 and len(p) == 0:
            return False
        elif len(p) > 1 and p[1] == '*':
            if len(s) > 0 and (p[0] == '.' or s[0] == p[0]):
                return self.isMatch(s[1:], p) or self.isMatch(s, p[2:])
            else:
                return self.isMatch(s, p[2:])
        elif len(s) > 0 and (s[0] == p[0] or p[0] == '.'):
            return self.isMatch(s[1:], p[1:])
        else:
            return False
        """
        # 动态规划
        # 执行用时 : 48 ms, 在所有 Python3 提交中击败了 95.40% 的用户
        # 内存消耗 : 13.4 MB, 在所有 Python3 提交中击败了 100.00% 的用户

        s = ' ' + s
        p = ' ' + p

        result_list = [[False] * len(p) for _ in s]

        for i in range(len(s)):
            for j in range(len(p)):
                if i == 0 and j == 0:
                    result_list[i][j] = True
                elif i > 0 and j == 0:
                    result_list[i][j] = False
                elif p[j] == '*':
                    if s[i] == p[j - 1] or (i > 0 and p[j - 1] == '.'):
                        result_list[i][j] = result_list[i - 1][j] or result_list[i][j - 2]
                    else:
                        result_list[i][j] = result_list[i][j - 2]
                elif s[i] == p[j] or (i > 0 and p[j] == '.'):
                    result_list[i][j] = result_list[i - 1][j - 1]
                else:
                    result_list[i][j] = False

        return result_list[-1][-1]

    def stoneGame(self, piles: list) -> bool:
        """
        877. 石子游戏
        :see https://leetcode-cn.com/problems/stone-game/
        """
        # A[i][j]表示先手最高分数，B[i][j]表示后手最高分数，最优得分之差 = A[0][N-1] - B[0][N-1]
        # 先手：
        # A[i][j] = max(B[i+1][j] + piles[i], B[i][j-1] + piles[j])
        # 后手：
        # B[i][j] = A[i+1][j], A[i][j-1]
        # 当i=j时，只有一个石子堆，先手为piles[i]，后手为0

        first_dp = [[0] * len(piles) for _ in piles]
        back_dp = [[0] * len(piles) for _ in piles]

        for i in range(0, len(piles)):
            first_dp[i][i] = piles[i]

        for i in range(1, len(piles)):
            for j in range(0, len(piles) - i):
                # print(j, i + j)
                left = back_dp[j + 1][i + j] + piles[j]
                right = back_dp[j][i + j - 1] + piles[i + j]

                if left >= right:
                    first_dp[j][i + j] = left
                    back_dp[j][i + j] = first_dp[j + 1][i + j]
                else:
                    first_dp[j][i + j] = right
                    back_dp[j][i + j] = first_dp[j][i + j - 1]

        return first_dp[0][-1] > back_dp[0][-1]

    def stoneGameII(self, piles: list) -> int:
        """
        1140. 石子游戏 II
        :see https://leetcode-cn.com/problems/stone-game-ii/
        """
        # 最开始，A可选 1 或 2，即 piles[0]，或者 piles[0] + piles[1]
        # 然后，后手变成先手，从 i = 1, M = 1 或者 i = 2, M = 2 开始
        #
        # 先手：
        # A[i][j] 表示从第i堆开始，最多取 j 堆的数量, 1 <= k <= 2j
        # A[i][j] = max(B[i + k][max(j, k)] + sum(piles[i:i + k]))
        #
        # 后手：
        # 根据先手选择的堆数 k
        # B[i][j] = A[i + k][max(j, k)]
        #
        # 基础条件：
        # i + 2 * j >= N, A[i][j] = sum(piles[i:]), B[i][j] = 0

        first_dp = [[0] * len(piles) for _ in piles]
        back_dp = [[0] * len(piles) for _ in piles]

        for i in range(len(piles)):
            for j in range(len(piles)):
                if i + 2 * j >= len(piles):
                    first_dp[i][j] = sum(piles[i:])

        # 从倒数第三行开始逆序遍历
        for i in range(len(piles) - 3, -1, -1):
            for j in range(1, len(piles)):
                if i + 2 * j >= len(piles):
                    continue

                # 从 1 到 2 * j 之间查找，可以获得的最多的石头数量
                max_k = 0
                max_stones = 0
                for k in range(1, 2 * j + 1):
                    if i + k >= len(piles):
                        break
                    stones = back_dp[i + k][max(j, k)] + sum(piles[i:i + k])
                    if stones > max_stones:
                        max_stones = stones
                        max_k = k

                # 先手可以获得的最多的石头数量就是max_stones
                first_dp[i][j] = max_stones
                # 后手可以获得最多的石头数量就是，去掉这次先手取的 k 个石子之后的先手的数量
                back_dp[i][j] = first_dp[i + max_k][max(j, max_k)]

        # for i in range(len(first_dp)):
        #     for j in range(1, len(back_dp)):
        #         print(f'({first_dp[i][j]}, {back_dp[i][j]}, {first_dp[i][j] + back_dp[i][j]})', end=' ')
        #     print()

        return first_dp[0][1]

    def stoneGameIII(self, stoneValue: list) -> str:
        """
        1406. 石子游戏 III
        :see https://leetcode-cn.com/problems/stone-game-iii/
        """
        # 先手first_dp[i]，表示先手从第 i 堆开始取，可以取的最大数量
        # first_dp[i] = max(back_dp[i + k] + sum(stoneValue[i:i + k])), 1 <= k <= 3
        #
        # 后手back_dp[i]，表示后手在先手取完前 k 堆后，从第 i + k 堆开始取，可以取的最大数量
        # back_dp[i] = first_dp[i + k]
        #
        # 初始条件，当只剩最后 1 堆时，先手取全部，后手取0
        # 当 i == len(stoneValue) - 1 时, first_dp[i] = stoneValue[-1], back_dp[i] = 0

        first_dp = [-1000] * len(stoneValue)
        back_dp = [-1000] * len(stoneValue)

        first_dp[-1] = stoneValue[-1]
        back_dp[-1] = 0

        for i in range(len(stoneValue) - 2, -1, -1):
            max_k = 0
            max_stone = -1000
            for k in [1, 2, 3]:
                if i + k > len(stoneValue):
                    break
                stones = (back_dp[i + k] if i + k < len(stoneValue) else 0) + sum(stoneValue[i:i + k])
                if stones > max_stone:
                    max_stone = stones
                    max_k = k

            first_dp[i] = max_stone
            back_dp[i] = first_dp[i + max_k] if i + max_k < len(stoneValue) else 0

        if first_dp[0] > back_dp[0]:
            return 'Alice'
        elif first_dp[0] == back_dp[0]:
            return 'Tie'
        else:
            return 'Bob'

    def smallestSufficientTeam(self, req_skills: list, people: list) -> list:
        """
        1125. 最小的必要团队
        :see https://leetcode-cn.com/problems/smallest-sufficient-team/
        """
        req_skills_dict = {req_skills[i]: len(req_skills) - 1 - i for i in range(len(req_skills))}

        # ---- 剪枝 ----
        # 将每个人的技能整理成列表
        people_skills_list = []
        for i in range(len(people)):
            # 计算该成员所掌握的技能
            skills = 0
            for skill in people[i]:
                index = req_skills_dict.get(skill, -1)
                if index >= 0:
                    skills |= 1 << index

            # 遍历判断该成员的技能是否已被他人掌握或者掌握他人技能
            for j in range(len(people_skills_list)):
                tmp_skill = people_skills_list[j][1] | skills
                if tmp_skill == people_skills_list[j][1]:
                    # 该成员的技能已被另一成员完全包含
                    skills = 0
                    break
                elif tmp_skill == skills:
                    # 该成员的技能完全包含另一成员
                    people_skills_list[j][1] = 0

            if skills > 0:
                people_skills_list.append([i, skills])

        # 去除无技能的成员
        people_skills_list = [i for i in people_skills_list if i[1] != 0]

        # for index, skill in people_skill_list:
        #     print(f'({index}, {bin(skill)})')

        # ---- 动态规划 ----
        # dp[i]表示，满足掌握技能i的最少人数及其成员
        dp = [[0, []] for _ in range(1 << len(req_skills))]
        for index, skills in people_skills_list:
            dp[skills] = [1, [index]]
            for i in range(len(dp)):
                if dp[i][0] == 0:
                    continue
                next_skills = skills | i
                if dp[next_skills][0] == 0 or dp[i][0] + 1 < dp[next_skills][0]:
                    dp[next_skills] = [dp[i][0] + 1, dp[i][1] + [index]]

        # for i in range(len(dp)):
        #     print(f'{bin(i)}: {dp[i][0]}, {dp[i][1]}')

        return dp[-1][1]

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """
        1143. 最长公共子序列
        :see https://leetcode-cn.com/problems/longest-common-subsequence/
        """
        text1 = ' ' + text1
        text2 = ' ' + text2

        result = [[0] * len(text1) for _ in text2]
        for i in range(len(text2)):
            for j in range(len(text1)):
                if text2[i] == text1[j]:
                    result[i][j] = result[i - 1][j - 1] + 1
                else:
                    result[i][j] = max(result[i - 1][j], result[i][j - 1])
        # for i in result:
        #     print(i)
        return result[-1][-1] - 1

    def maxEnvelopes(self, envelopes: list) -> int:
        """
        354. 俄罗斯套娃信封问题
        :see https://leetcode-cn.com/problems/russian-doll-envelopes/
        """
        # 最长上升子序列进阶版
        # 因为当宽度相同时，即使高度比较大，也不能放入。因此，宽度升序排列，高度降序排列
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        # print(envelopes)

        result = []
        for width, height in envelopes:
            index = bisect.bisect_left(result, height)
            if index == len(result):
                result.append(height)
            else:
                result[index] = height
        # print(result)
        return len(result)

    def robII(self, nums: list) -> int:
        """
        213. 打家劫舍 II
        :see https://leetcode-cn.com/problems/house-robber-ii/
        """

        def rob(nums: list) -> int:
            money_list = [0] * len(nums)
            max_money = 0

            for i in range(len(nums)):
                if i < 2:
                    money_list[i] = nums[i]
                elif i == 2:
                    money_list[i] = nums[0] + nums[2]
                else:
                    money_list[i] = max(money_list[i - 3], money_list[i - 2]) + nums[i]

                max_money = max(max_money, money_list[i])

            return max_money

        # 核心原则是第一个和最后一个不能同时抢，那么取max(nums[:-1], nums[1:])即可
        return max(rob(nums[:-1]), rob(nums[1:])) if len(nums) > 1 else sum(nums)

    def waysToChange(self, n: int) -> int:
        """
        面试题 08.11. 硬币
        :see https://leetcode-cn.com/problems/coin-lcci/
        """
        # 错误思路：f(n) = f(n - 25) + f(n - 10) + f(n - 5) + f(n - 1)
        # 错误思路中存在这种情况f(6) = 1 + 5, f(6) = 5 + 1的重复计算
        # 换一种思路，先计算只有1，再计算5，然后计算10，最后计算25的数量
        # 当只有1时，f(n) = f(n - 1)
        # 当有5时，f(n) += f(n - 5)，因为此时的f(n - 5)是只有1的情况，所以不存在重复运算
        # 10和25以此类推
        self.ways_to_change_result_list = [1]

        if len(self.ways_to_change_result_list) > n:
            return self.ways_to_change_result_list[n]

        last_length = len(self.ways_to_change_result_list)

        for coin in [1, 5, 10, 25]:
            for i in range(last_length, n + 1):
                if i >= len(self.ways_to_change_result_list):
                    self.ways_to_change_result_list.append(0)
                if i >= coin:
                    self.ways_to_change_result_list[i] = (self.ways_to_change_result_list[i] +
                                                          self.ways_to_change_result_list[i - coin]) % 1000000007
        # print(self.ways_to_change_result_list)
        return self.ways_to_change_result_list[n]

    def maxProfitV(self, prices: list) -> int:
        """
        309. 最佳买卖股票时机含冷冻期
        :see https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
        """
        # 没有，dp_zero[i] = max(dp_zero[i - 1], dp_one[i - 1] + price[i])
        # 持有，dp_one[i] = max(dp_one[i - 1], dp_zero[i - 2] - price[i])
        if len(prices) < 2:
            return 0
        dp = [(0, -prices[0]), (max(0, prices[1] - prices[0]), max(-prices[0], -prices[1]))]
        for i in range(2, len(prices)):
            dp.append((max(dp[i - 1][0], dp[i - 1][1] + prices[i]), max(dp[i - 1][1], dp[i - 2][0] - prices[i])))
        # print(dp)
        return dp[-1][0]

    def maxProfitVI(self, prices: list, fee: int) -> int:
        """
        714. 买卖股票的最佳时机含手续费
        :see https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/
        """
        # 没有，dp_zero[i] = max(dp_zero[i - 1], dp_one[i - 1] + price[i] - fee)
        # 持有，dp_one[i] = max(dp_one[i - 1], dp_zero[i - 1] - price[i])
        if len(prices) < 2:
            return 0
        dp = [(0, -prices[0] - fee), (max(0, prices[1] - prices[0] - fee), max(-prices[0], -prices[1]))]
        for i in range(2, len(prices)):
            dp.append((max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee), max(dp[i - 1][1], dp[i - 1][0] - prices[i])))
        print(dp)
        return dp[-1][0]

    def constrainedSubsetSum(self, nums: list, k: int) -> int:
        """
        5180. 带限制的子序列和
        :param nums:
        :param k:
        :return:
        """
        # 单调栈
        # max_dp: (i, sum, list)
        max_sum_dp = []
        for i in range(len(nums)):
            max_sum = 0
            for j in range(len(max_sum_dp) - 1, -1, -1):
                if max_sum_dp[j][0] < i - k:
                    break
                if max_sum_dp[j][1] > max_sum:
                    index, max_sum = max_sum_dp[j]

            while len(max_sum_dp) > 0 and max_sum_dp[-1][1] < max_sum + nums[i]:
                max_sum_dp.pop()

            max_sum_dp.append((i, max_sum + nums[i]))
            # print(max_sum_dp)

        return max(i[1] for i in max_sum_dp)

    def longestPalindromeSubseq(self, s: str) -> int:
        """
        516. 最长回文子序列
        :see https://leetcode-cn.com/problems/longest-palindromic-subsequence/
        """
        # dp[i][j]表示s[i...j]的最长回文子序列的长度
        # dp[i][j] = dp[i + 1][j - 1] + 2 if s[i] == s[j] else max(dp[i + 1][j], dp[i][j - 1])
        dp = [[0] * len(s) for _ in s]

        for i in range(len(s)):
            dp[i][i] = 1

        for i in range(1, len(s)):
            for j in range(len(s) - i):
                # print(j, i + j)
                x = j
                y = i + j
                dp[x][y] = (dp[x + 1][y - 1] + 2) if s[x] == s[y] else max(dp[x + 1][y], dp[x][y - 1])
        return dp[0][-1]

    def calculateMinimumHP(self, dungeon: list) -> int:
        """
        174. 地下城游戏
        :see https://leetcode-cn.com/problems/dungeon-game/
        """
        for i in range(len(dungeon) - 1, -1, -1):
            for j in range(len(dungeon[i]) - 1, -1, -1):
                if i == len(dungeon) - 1 and j == len(dungeon[i]) - 1:
                    continue
                elif i == len(dungeon) - 1:
                    dungeon[i][j] += dungeon[i][j + 1] if dungeon[i][j + 1] < 0 else 0
                elif j == len(dungeon[i]) - 1:
                    dungeon[i][j] += dungeon[i + 1][j] if dungeon[i + 1][j] < 0 else 0
                else:
                    dungeon[i][j] += max(dungeon[i][j + 1] if dungeon[i][j + 1] < 0 else 0,
                                         dungeon[i + 1][j] if dungeon[i + 1][j] < 0 else 0)

        # for i in dungeon:
        #     print(i)

        return 1 - dungeon[0][0] if dungeon[0][0] <= 0 else 1

    def mincostTickets(self, days: list, costs: list) -> int:
        """
        983. 最低票价
        :param days:
        :param costs:
        :return:
        """
        # f(n) = min(f(n - 1天) + costs[0], f(n - 7天) + cost[1], f(n - 30天) + cost[2])
        dp = [0] * len(days)
        for i, day in enumerate(days):
            one_day_cost = dp[i - 1] + costs[0]

            index = i - 1
            while index >= 0 and day - days[index] < 7:
                index -= 1
            seven_days_cost = dp[index] + costs[1]

            while index >= 0 and day - days[index] < 30:
                index -= 1
            thirty_days_cost = dp[index] + costs[2]

            dp[i] = min(one_day_cost, seven_days_cost, thirty_days_cost)

        return dp[-1]

    def maximalSquare(self, matrix: list) -> int:
        """
        221. 最大正方形
        :see https://leetcode-cn.com/problems/maximal-square/
        """
        # dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        dp = [[0] * len(matrix[0]) for _ in matrix]
        max_length = 0

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == '0':
                    continue
                elif i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

                max_length = max(max_length, dp[i][j])
        # print(dp)
        return max_length ** 2

    def maxCoins(self, nums: list) -> int:
        """
        312. 戳气球
        :see https://leetcode-cn.com/problems/burst-balloons/
        """
        # dp[i][j] = max(nums[k] * nums[i - 1] * nums[j + 1] + dp[i][k - 1] + dp[k + 1][j]), i <= k <= j
        dp = [[0] * len(nums) for _ in nums]
        nums.append(1)

        for i in range(len(nums) - 1):
            dp[i][i] = nums[i - 1] * nums[i] * nums[i + 1]

        for i in range(1, len(nums) - 1):
            for j in range(len(nums) - i - 1):
                # print(j, i + j)
                x = j
                y = i + j

                max_coin = 0
                for k in range(x, y + 1):
                    max_coin = max(max_coin,
                                   nums[k] * nums[x - 1] * nums[y + 1] + dp[x][k - 1] + dp[k + 1][y] if k + 1 < len(
                                       dp) else 0)

                dp[x][y] = max_coin

        for i in dp:
            print(i)

        return dp[0][-2]


if __name__ == "__main__":
    print(Solution().maxCoins([3, 1, 5, 8]))
