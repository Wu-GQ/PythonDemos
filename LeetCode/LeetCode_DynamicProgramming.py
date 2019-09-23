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
        零钱兑换
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/272/dynamic-programming/1180/
        """
        amount_list = [-1] * (amount + 1)
        amount_list[0] = 0

        coins.sort()

        for i in range(amount + 1):
            for coin in coins:
                if i > coin and amount_list[i - coin] >= 0:
                    amount_list[i] = amount_list[i - coin] + 1 if amount_list[i] == -1 else min(amount_list[i], amount_list[i - coin] + 1)
                elif i == coin:
                    amount_list[i] = 1
                elif i < coin:
                    break

        return amount_list[amount]

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

    def maxProfit(self, prices: list) -> int:
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

        return max_sum


if __name__ == "__main__":
    coins_list = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(Solution().maxSubArray(coins_list))
