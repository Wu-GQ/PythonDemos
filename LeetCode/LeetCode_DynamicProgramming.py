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
        最长上升子序列
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/272/dynamic-programming/1179/
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
        size = len(nums)
        if size < 2:
            return size

        tail = []
        for num in nums:
            # 找到大于等于 num 的第 1 个数
            l = 0
            r = len(tail)
            while l < r:
                mid = l + (r - l) // 2
                if tail[mid] < num:
                    l = mid + 1
                else:
                    r = mid
            if l == len(tail):
                tail.append(num)
            else:
                tail[l] = num

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
        盛水最多的容器，S = max((j-i) * min(a[i], a[j]))
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
        编辑距离
        :see https://leetcode-cn.com/problems/edit-distance/
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
                    result_list[row][column] = (result_list[row - 1][column] and s2[row - 1] == s3[row - 1 + column]) or (
                            result_list[row][column - 1] and s1[column - 1] == s3[column - 1 + row])

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
                    result_list[row][column] = result_list[row][column - 1] + (result_list[row - 1][column - 1] if row > 0 else 1)
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
                possess_one_profit, possess_two_profit, none_one_profit, none_two_profit_list = -prices[i], -float('inf'), 0, -float('inf')
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

                    if temp_zero_count < zero_count or (temp_zero_count == zero_count and compare_nums(zero_nums, temp_zero_nums)):
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

                    if temp_one_count < one_count or (temp_one_count == one_count and compare_nums(one_nums, temp_one_nums)):
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

                    if temp_two_count < two_count or (temp_two_count == two_count and compare_nums(two_nums, temp_two_nums)):
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

                    if temp_zero_count < zero_count or (temp_zero_count == zero_count and compare_nums(zero_nums, temp_zero_nums)):
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

                    if temp_one_count < one_count or (temp_one_count == one_count and compare_nums(one_nums, temp_one_nums)):
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

                    if temp_two_count < two_count or (temp_two_count == two_count and compare_nums(two_nums, temp_two_nums)):
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


if __name__ == "__main__":
    print(Solution().isMatch("mississippi", "mis*is*ip*."))
