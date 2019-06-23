import math


class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        """
        3的幂
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/274/math/1194/
        """
        return False if n <= 0 else 3 ** int(math.log(n, 3)) == n

    def missingNumber(self, nums: list) -> int:
        nums_sum = 0
        for i in nums:
            nums_sum += i

        return (len(nums) * (len(nums) + 1) >> 1) - nums_sum

    _array = []

    def fizzBuzz(self, n: int) -> list:
        """
        Fizz Buzz
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/25/math/60/
        """
        if n < len(self._array):
            return self._array[:n]

        self._array = []
        for i in range(0, n):
            a = (i + 1) % 3 == 0
            b = (i + 1) % 5 == 0
            if a and b:
                self._array.append("FizzBuzz")
            elif a:
                self._array.append("Fizz")
            elif b:
                self._array.append("Buzz")
            else:
                self._array.append(str(i + 1))
        return self._array

    def maxProfit(self, prices: list) -> int:
        """
        买卖股票的最佳时机 II
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/1/array/22/
        """
        low = float('inf')
        high = 0
        profit = 0

        for i in range(len(prices)):
            if prices[i] <= low:
                low = prices[i]

                j = i + 1
                while j < len(prices):
                    if prices[j] > high:
                        high = prices[j]
                    j += 1

                profit += high - low
                low = high

        return profit


if __name__ == '__main__':
    print(Solution().fizzBuzz(15))
    print(Solution().fizzBuzz(9))
    print(Solution().fizzBuzz(20))
