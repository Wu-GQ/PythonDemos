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

    def maxDiff(self, num: int) -> int:
        """
        5385. 改变一个整数能得到的最大差值
        :param num:
        :return:
        """
        max_num_list = list(str(num))
        min_num_list = list(str(num))

        ch = 'a'
        for i in range(len(max_num_list)):
            if ch == 'a' and max_num_list[i] != '9':
                ch = max_num_list[i]
                max_num_list[i] = '9'
            elif ch != 'a' and max_num_list[i] == ch:
                max_num_list[i] = '9'

        ch = 'a'
        next_ch = 'a'
        for i in range(len(min_num_list)):
            if ch == 'a' and i == 0 and min_num_list[i] != '1':
                ch = min_num_list[i]
                next_ch = '1'
                min_num_list[i] = '1'
            elif ch == 'a' and i > 0 and min_num_list[0] == '1' and min_num_list[i] != '0' and min_num_list[i] != '1':
                ch = min_num_list[i]
                next_ch = '0'
                min_num_list[i] = '0'
            elif ch == 'a' and i > 0 and min_num_list[0] != '1' and min_num_list[i] != '0':
                ch = min_num_list[i]
                next_ch = '0'
                min_num_list[i] = '0'
            elif ch != 'a' and min_num_list[i] == ch:
                min_num_list[i] = next_ch

        max_num = int(''.join(max_num_list))
        min_num = int(''.join(min_num_list))
        # print(max_num, min_num)
        return max_num - min_num

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

    def mySqrt(self, x: int) -> int:
        """
        69. x 的平方根
        :see https://leetcode-cn.com/problems/sqrtx/
        """
        current, next = x, x / 2
        while abs(current - next) >= 0.1:
            current, next = next, (next + x / next) / 2
            # print(current, next)
        return int(next)

    def myPow(self, x: float, n: int) -> float:
        """
        50. Pow(x, n)
        :see https://leetcode-cn.com/problems/powx-n/
        """
        # 快速幂方法
        # f(x, n) = f(x, n // 2) ** 2
        if n == 0:
            return 1

        if n < 0:
            x = 1 / x
            n = -n

        if n % 2 == 1:
            return self.myPow(x, n - 1) * x
        else:
            return self.myPow(x, n // 2) ** 2

    def simplifiedFractions(self, n: int) -> list:
        """
        5397. 最简分数
        :param n:
        :return:
        """
        result = []
        for i in range(1, n + 1):
            for j in range(1, i):
                if math.gcd(i, j) == 1:
                    result.append(f'{j}/{i}')
        return result

    def countDigitOne(self, n: int) -> int:
        """
        233. 数字 1 的个数
        :see https://leetcode-cn.com/problems/number-of-digit-one/
        """

        def string2num(s: str) -> int:
            # 字符串转数字
            return int(s) if s else 0

        # 以计算数字n的千位上的1个数为例，n = int('abcdef')
        # 若 c = 0 时，千位出现1的个数为，int('ab') * 1000
        # 若 c = 1 时，千位出现1的个数为，int('ab') * 1000 + int('def') + 1，1是代表'000'
        # 若 c > 1 时，千位出现1的个数为，int('ab') * 1000 + 1000
        if n < 0:
            return 0
        result = 0
        string = str(n)
        length = len(string)
        for i in range(length):
            index = length - 1 - i
            if string[index] == '0':
                result += string2num(string[:index]) * pow(10, i)
            elif string[index] == '1':
                result += string2num(string[:index]) * pow(10, i) + string2num(string[index + 1:]) + 1
            else:
                result += (string2num(string[:index]) + 1) * pow(10, i)

        return result

    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        """
        201. 数字范围按位与
        :see https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/
        """
        # 按位与，与运算中若只要有一个为零，则该位为0
        # 比如说 9 和 12 的最长公共前缀
        # 9 : 00001 001
        # 10: 00001 010
        # 11: 00001 011
        # 12: 00001 100
        # r : 00001 000
        # 则这题变成了"查找 m 和 n 的最长公共前缀"
        result = 0

        while m != n:
            m >>= 1
            n >>= 1
            result += 1

        # 将 m 还原，不相同的位已经被右移去掉，只需要左移相同的位数即可
        return m << result

    def hammingDistance(self, x: int, y: int) -> int:
        """
        461. 汉明距离
        :see https://leetcode-cn.com/problems/hamming-distance/
        """
        num = x ^ y
        result = 0
        '''# 逐步位移法
        while num > 0:
            if num & 1 == 1:
                result += 1
            num >>= 1
        '''
        # 布赖恩·克尼根算法，通过 n & (n - 1) 可以快速舍去末尾开始的最后一个 1
        # 例如 n = 8
        # 8: 0000 1000
        # 7: 0000 0111
        # 8 & 7 = 0，快速舍去 8 的最后一位1，而不需要只要 1 在哪个位置
        # 这题需要知道 num 中有几个 1，即计算 num 可被该算法舍去多少次
        while num != 0:
            num &= num - 1
            result += 1

        return result

    def isUgly(self, num: int) -> bool:
        """
        263. 丑数
        :see https://leetcode-cn.com/problems/ugly-number/
        """
        if num < 1:
            return False

        while num % 2 == 0:
            num //= 2
        while num % 3 == 0:
            num //= 3
        while num % 5 == 0:
            num //= 5

        return num == 1

    def kthFactor(self, n: int, k: int) -> int:
        """
        5433. n 的第 k 个因子
        :param n:
        :param k:
        :return:
        """
        index = 0
        while k > 0:
            index += 1
            if n % index == 0:
                k -= 1

            if index > n:
                return -1

        return index

    def toHex(self, num: int) -> str:
        """
        405. 数字转换为十六进制数
        :see https://leetcode-cn.com/problems/convert-a-number-to-hexadecimal/
        """
        # 补码 = 符号位不变，原码取反后加 1
        # 二进制原码，i = 0 为最低位
        bin_list = [0] * 32
        # 判断是否是负数
        is_negative = num < 0

        # 转换为二进制原码
        num = abs(num)
        index = 0
        while num > 0 and index < 32:
            bin_list[index] = num % 2
            index += 1
            num >>= 1

        # 反转原码
        if is_negative:
            bin_list = [1 - i for i in bin_list]
            c = 1
            for i in range(32):
                if c > 0:
                    tmp = bin_list[i] + c
                    bin_list[i] = tmp % 2
                    c = tmp // 2
                else:
                    break

        # 符号位赋值
        bin_list[-1] = 1 if is_negative else 0

        # 二进制转16进制
        hex_string = '0123456789abcdef'
        result = []
        for i in range(0, 32, 4):
            r = bin_list[i] + bin_list[i + 1] * 2 + bin_list[i + 2] * 4 + bin_list[i + 3] * 8
            result.insert(0, hex_string[r])

        # 去掉前置0
        while result and result[0] == '0':
            result.pop(0)

        return ''.join(result) if result else '0'

    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        """
        5464. 换酒问题
        :param numBottles:
        :param numExchange:
        :return:
        """
        result = 0
        a, b = numBottles, 0
        while a > 0:
            result += a
            b += a
            tmp = b // numExchange
            a = tmp
            b -= tmp * numExchange
        return result

    def subarrayBitwiseORs(self, A: list) -> int:
        """
        898. 子数组按位或操作
        :see https://leetcode-cn.com/problems/bitwise-ors-of-subarrays/
        """
        '''
        对于子数组 A[:i]，包含 A[0:i], A[1:i], A[2:i], ..., A[i-1:i]
        但是考虑到位运算的性质，A[j] 必定小于等于 A[j]|A[j+1]，且 A[j]|A[j+1] 包含了 A[j] 二进制表示下的所有位置的 1
        由于 0 <= A[i] <= 10^9，因此对应每一个以 A[i] 结尾的子数组，最多存在 30 种情况(2^30 > 10^9)
        最差的情况就是[1, 2, 4, 8, 16, ..., 2^29]，这就恰好存在 30 种情况
        所以，时间复杂度为 O(30 * N)
        '''
        result = set()
        # temp 存储以某一位结尾的所有结果
        temp = set()
        for i in A:
            temp = {i | j for j in temp} | {i}
            result |= temp
        # print(result)
        return len(result)

    def closestToTarget(self, arr: list, target: int) -> int:
        """
        1521. 找到最接近目标值的函数值
        :see https://leetcode-cn.com/problems/find-a-value-of-a-mysterious-function-closest-to-target/
        """
        result = abs(arr[0] - target)
        # temp 存储以某一位结尾的所有结果
        temp = set()
        for i in arr:
            temp = {i & j for j in temp} | {i}
            # print(i, temp)
            for j in temp:
                result = min(result, abs(j - target))
            if result == 0:
                return 0
        return result

    def closestDivisors(self, num: int) -> list:
        """
        1362. 最接近的因数
        :see https://leetcode-cn.com/problems/closest-divisors/
        """
        floor = math.ceil(math.sqrt(num + 2)) + 1
        result, diff = [1, num + 1], num
        for i in range(2, floor):
            if (num + 1) % i == 0:
                a = (num + 1) // i
                if a < diff + i:
                    result = [a, i]
                    diff = a - i
            if (num + 2) % i == 0:
                a = (num + 2) // i
                if a < diff + i:
                    result = [a, i]
                    diff = a - i
        return result

    def countOdds(self, low: int, high: int) -> int:
        """
        5456. 在区间范围内统计奇数数目
        :param low:
        :param high:
        :return:
        """
        result = (high - low) // 2
        if low % 2 == 1 or high % 2 == 1:
            result += 1
        return result


if __name__ == '__main__':
    print(Solution().countOdds(2, 4))
