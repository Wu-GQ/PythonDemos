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
        # 矩形重叠 = 矩形1和矩形2之间的距离是否在一定范围内
        return rec1[0] < rec2[2] and rec1[2] > rec2[0] and rec1[1] < rec2[3] and rec1[3] > rec2[1]

    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        """
        365. 水壶问题
        :see https://leetcode-cn.com/problems/water-and-jug-problem/
        """
        if z > x + y:
            return False

        water_stack = [(0, 0)]
        checked_water = set()

        while water_stack:
            remain_x, remain_y = water_stack.pop(0)
            if remain_x == z or remain_y == z or remain_x + remain_y == z:
                return True

            if (remain_x, remain_y) in checked_water:
                continue
            else:
                checked_water.add((remain_x, remain_y))

            # x清空
            water_stack.append((0, remain_y))
            # y清空
            water_stack.append((remain_x, 0))
            # x加满
            water_stack.append((x, remain_y))
            # y加满
            water_stack.append((remain_x, y))
            # x->y
            left_y = y - remain_y
            if left_y >= remain_x:
                water_stack.append((0, remain_x + remain_y))
            else:
                water_stack.append((left_y - remain_x, y))
            # x<-y
            left_x = x - remain_x
            if left_x >= remain_y:
                water_stack.append((remain_x + remain_y, 0))
            else:
                water_stack.append((x, left_x - remain_y))

        return False

    def minIncrementForUnique(self, A: list) -> int:
        """
        945. 使数组唯一的最小增量
        :see https://leetcode-cn.com/problems/minimum-increment-to-make-array-unique/
        """
        if not A:
            return 0
        A.sort()
        max_num = A[0] - 1
        add_times = 0
        # print(A)
        for i in A:
            # print(i, max_num, add_times)
            if i <= max_num:
                add_times += max_num - i + 1
                max_num += 1
            else:
                max_num = i
        return add_times

    def hasGroupsSizeX(self, deck: list) -> bool:
        """
        914. 卡牌分组
        :see https://leetcode-cn.com/problems/x-of-a-kind-in-a-deck-of-cards/
        """
        if len(deck) < 2:
            return False

        count_dict = {}
        for i in deck:
            count_dict[i] = count_dict.get(i, 0) + 1

        gcd_result = count_dict[deck[0]]
        for i in count_dict.values():
            gcd_result = math.gcd(gcd_result, i)
            if gcd_result < 2:
                return False
        return gcd_result >= 2

    def lastRemaining(self, n: int, m: int) -> int:
        """
        圆圈中最后剩下的数字
        https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/
        """
        # 下标的关系，n = 5, m = 3
        # --------------
        # 第一次删除：
        # 0, 1, 2, 3, 4 
        # 0, 1,  , 3, 4
        # 2, 3,  , 0, 1
        # --------------
        # 删除一个数字后
        # 3->0, 4->1, 0->2, 1->3
        # 建立转换关系函数：
        # f(n - 1) = (f(n) + m) % (n - 1)
        # --------------
        # 第二次删除：
        # 0, 1, 2, 3
        # 0, 1,  , 3
        # 1, 2,  , 0
        # --------------
        # 验证第二次删除的情况
        # 3->0, 0->1, 1->2
        # 转换函数关系为：
        # f(n - 2) = (f(n - 1) + m) % (n - 1)
        # --------------
        # 当n = 1时，只有一个数字，无论怎么转换，都是0
        # 因此，f(1) = 0，即最后一次转换后，最后剩下的这个数字的下标为0
        # 那么，通过上述的转换函数，可以一步步逆推回去，找到未转换时，该数的下标
        if (n == 1):
            return 0
        
        a = (self.lastRemaining(n - 1, m) + m) % (n - 1)
        print(n , a)
        return a

if __name__ == '__main__':
    print(Solution().lastRemaining(5, 3))
