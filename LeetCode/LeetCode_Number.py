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

    def distributeCandies(self, candies: int, num_people: int) -> list:
        """
        1103. 分糖果 II
        :see https://leetcode-cn.com/problems/distribute-candies-to-people/
        """
        if num_people == 0:
            return []

        # 完整的分糖果次数
        n: int = math.floor(0.5 * (-1 + math.sqrt(1 + 8 * candies)))
        line: int = math.floor(n / num_people)
        column: int = n - line * num_people
        left = candies - (1 + n) * n * 0.5

        result_list = []

        for i in range(0, num_people):
            candy = 0
            if i < column:
                candy = (i + 1) * (line + 1) + num_people * line * (line + 1) / 2
            elif i == column:
                candy = (i + 1) * line + num_people * line * (line - 1) / 2 + left
            else:
                candy = (i + 1) * line + num_people * line * (line - 1) / 2
            result_list.append(int(candy))

        return result_list

    def isPalindrome(self, x: int) -> bool:
        """
        回文数
        :see https://leetcode-cn.com/problems/palindrome-number/
        """
        if x < 0:
            return False

        string = str(x)
        for i in range(len(string)):
            if string[i] != string[-1 - i]:
                return False

        return True

    def isRectangleOverlap(self, rec1: list, rec2: list) -> bool:
        """
        836. 矩形重叠
        :see https://leetcode-cn.com/problems/rectangle-overlap/
        """
        # 矩形1的中心: (x1 + x2) / 2, (y1 + y2) / 2
        # 矩形1的宽度: (x2 - x1) / 2, 高度: (y2 - y1) / 2
        # 矩形2的中心: (x3 + x4) / 2, (y3 + y4) / 2
        # 矩形2的宽度: (x4 - x3) / 2, 高度: (y4 - y3) / 2
        # 矩形重叠 = 矩形1和矩形2之间的距离
        return rec1[0] - rec2[2] < 0 < rec1[2] - rec2[0] and rec1[1] - rec2[3] < 0 < rec1[3] - rec2[1]


if __name__ == '__main__':
    print(Solution().isRectangleOverlap([0,0,1,1], [1,0,2,1]))
