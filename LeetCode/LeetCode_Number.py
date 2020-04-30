import math


class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        """
        3的幂
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/274/math/1194/
        """
        return n == 3 ** (round(math.log(n, 3) * 1000) / 1000) if n > 0 else False

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
        # f(n) = (f(n - 1) + m) % n
        # f(n - 1) = (x - m) % n if (x - m) % n >= 0 else (x - m) % n + n
        # --------------
        # 第二次删除：
        # 0, 1, 2, 3
        # 0, 1,  , 3
        # 1, 2,  , 0
        # --------------
        # 验证第二次删除的情况
        # 3->0, 0->1, 1->2
        # 转换函数关系成立
        # --------------
        # 当n = 1时，只有一个数字，无论怎么转换，都是0
        # 因此，f(1) = 0，即最后一次转换后，最后剩下的这个数字的下标为0
        # 那么，通过上述的转换函数，可以一步步逆推回去，找到未转换时，该数的下标n
        return (self.lastRemaining(n - 1, m) + m) % n if n > 1 else 0

    def checkOverlap(self, radius: int, x_center: int, y_center: int, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        圆和矩形是否有重叠
        :param radius:
        :param x_center:
        :param y_center:
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :return:
        """

        def distance(x, y) -> float:
            return (x_center - x) ** 2 + (y_center - y) ** 2

        return (x1 - radius <= x_center <= x2 + radius and y1 <= y_center <= y2) or (
                x1 <= x_center <= x2 and y1 - radius <= y_center <= y2 + x_center) or (
                       distance(x1, y2) <= radius ** 2) or (
                       distance(x2, y2) <= radius ** 2) or (distance(x1, y1) <= radius ** 2) or (
                       distance(x2, y1) <= radius ** 2)

    def numSteps(self, s: str) -> int:
        """
        将二进制表示减到 1 的步骤数
        :param s:
        :return:
        """

        def step_count(nums: list) -> int:
            if len(nums) == 1:
                return 0

            if nums[-1] == '0':
                nums = nums[:-1]
            else:
                nums[-1] = '0'
                changed = False
                for i in range(len(nums) - 2, -1, -1):
                    if nums[i] == '0':
                        nums[i] = '1'
                        changed = True
                        break
                    else:
                        nums[i] = '0'
                if not changed:
                    nums.insert(0, '1')
            # print(nums)
            return step_count(nums) + 1

        nums_list = [i for i in s]
        return step_count(nums_list)

    def intersection(self, start1: list, end1: list, start2: list, end2: list) -> list:
        """
        面试题 16.03. 交点
        :see https://leetcode-cn.com/problems/intersection-lcci/
        """
        if start1[0] > end1[0]:
            start1, end1 = end1, start1
        if start2[0] > end2[0]:
            start2, end2 = end2, start2

        a_y_distance = end1[1] - start1[1]
        a_x_distance = end1[0] - start1[0]
        b_y_distance = end2[1] - start2[1]
        b_x_distance = end2[0] - start2[0]

        if a_x_distance == 0 and b_x_distance == 0:
            # 水平的两根线
            if start1[0] == start2[0] and (start1[1] <= start2[1] <= end1[1] or start2[1] <= start1[1] <= end2[1]):
                return [start1[0], max(start1[1], start2[1])]
            else:
                return []
        elif a_x_distance * b_y_distance == a_y_distance * b_x_distance:
            # 平行线，判断是否重叠
            a_y0 = (end1[0] * start1[1] - start1[0] * end1[1]) / (end1[0] - start1[0])
            b_y0 = (end2[0] * start2[1] - start2[0] * end2[1]) / (end2[0] - start2[0])
            if a_y0 == b_y0 and (start1[0] <= start2[0] <= end1[0] or start2[0] <= start1[0] <= end2[0]):
                a = [start1, end1, start2, end2]
                a.sort(key=lambda x: x[0])
                return a[1]
            else:
                return []
        elif a_x_distance == 0 and (start2[0] <= start1[0] <= end2[0]):
            # a 线是水平的
            y = b_y_distance * start1[0] / b_x_distance + start2[1] - b_y_distance * start2[0] / b_x_distance
            if start1[1] <= y <= end1[1]:
                return [start1[0], y]
            else:
                return []
        elif b_x_distance == 0:
            # b 线是水平的
            y = a_y_distance * start2[0] / a_x_distance + start1[1] - b_y_distance * start1[0] / b_x_distance
            if start2[1] <= y <= end2[1]:
                return [start2[0], y]
            else:
                return []
        else:
            x = (start2[1] - start2[0] * b_y_distance / b_x_distance - start1[1] + start1[
                0] * a_y_distance / a_x_distance) / (
                        a_y_distance / a_x_distance - b_y_distance / b_x_distance)
            if start1[0] <= x <= end1[0] or start2[0] <= x <= end2[0]:
                y = (a_y_distance / a_x_distance) * x + (start1[1] - start1[0] * a_y_distance / a_x_distance)
                if (start1[1] <= y <= end1[1] or end1[1] <= y <= start1[1]) and (
                        start2[1] <= y <= end2[1] or end2[1] <= y <= start2[1]):
                    return [x, y]
        return []

    def numOfWays(self, n: int) -> int:
        """
        5383. 给 N x 3 网格图涂色的方案数
        :param n:
        :return:
        """
        # aba = (aba * 0.6 + 0.5 * abc) * 5
        # abc = (aba * 0.4 + 0.5 * abc) * 4
        # 当 N = 1 时，直接返回 12
        # 当 N = 2 时，初始值 30 和 24

        # if n == 1:
        #     return 12
        # aba, abc = 30, 24
        # for i in range(2, n):
        #     aba, abc = int(3 * aba + 2.5 * abc) % (10 ** 9 + 7), int(1.6 * aba + 2 * abc) % (10 ** 9 + 7)
        #     print(f'{i}: {aba}, {abc}, {aba + abc}')
        # return (aba + abc) % (10 ** 9 + 7)

        a, b = 6, 6
        while n > 1:
            n -= 1
            a, b = 2 * a + 2 * b, 2 * a + 3 * b
            # print(a, b)
        return (a + b) % (10 ** 9 + 7)

    def movingCount(self, m: int, n: int, k: int) -> int:
        """
        机器人的运动范围
        :see https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/
        """
        # 可达的格子
        num_set = set()

        for i in range(0, m):
            m_sum = i // 10 + i % 10
            # x 坐标超过 k 时，不用继续搜索
            if m_sum > k:
                break

            for j in range(0, n):
                n_sum = j // 10 + j % 10
                # y 坐标超过 k 时，不用继续搜索
                if n_sum > k:
                    break

                # 只要左边的格子或者上边的格子可达，该格子即可达
                if m_sum + n_sum <= k and (i == 0 or (i - 1, j) in num_set or (i, j - 1) in num_set):
                    num_set.add((i, j))
                    # print(i, j)

        return len(num_set)

    def isHappy(self, n: int) -> bool:
        """
        202. 快乐数
        :see https://leetcode-cn.com/problems/happy-number/
        """
        num_set = set()
        while n not in num_set:
            num_set.add(n)

            num_sum = 0
            while n > 0:
                num_sum += (n % 10) ** 2
                n //= 10

            if num_sum == 1:
                return True
            else:
                n = num_sum

            # print(n)

        return False


if __name__ == '__main__':
    print(Solution().isHappy(17))
