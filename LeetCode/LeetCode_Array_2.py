import bisect
import heapq
from queue import PriorityQueue


class Solution:

    def subarraySum(self, nums: list, k: int) -> int:
        """
        560. 和为K的子数组
        :see https://leetcode-cn.com/problems/subarray-sum-equals-k/
        """
        sum_dict = {0: 1}
        last_sum = 0
        result = 0

        for num in nums:
            last_sum += num
            if last_sum - k in sum_dict:
                result += sum_dict[last_sum - k]
            sum_dict[last_sum] = sum_dict.get(last_sum, 0) + 1

        return result

    def busyStudent(self, startTime: list, endTime: list, queryTime: int) -> int:
        """
        5412. 在既定时间做作业的学生人数
        :param startTime:
        :param endTime:
        :param queryTime:
        :return:
        """
        result = 0
        for i in range(len(startTime)):
            if startTime[i] <= queryTime <= endTime[i]:
                result += 1
        return result

    def peopleIndexes(self, favoriteCompanies: list) -> list:
        """
        5414. 收藏清单
        :param favoriteCompanies:
        :return:
        """
        favoriteCompanies = [set(i) for i in favoriteCompanies]
        result = []

        for i in range(len(favoriteCompanies)):
            is_sub_set = False
            for j in range(len(favoriteCompanies)):
                if i != j and favoriteCompanies[i].issubset(favoriteCompanies[j]):
                    is_sub_set = True
                    break

            if not is_sub_set:
                result.append(i)

        return result

    def findMedianSortedArrays(self, nums1: list, nums2: list) -> float:
        """
        4. 寻找两个正序数组的中位数
        :see https://leetcode-cn.com/problems/median-of-two-sorted-arrays/
        """

        def add_num(num: int):
            # 向两个最大堆和最小堆中加入新的数据
            if not small_heapq or num < -small_heapq[0]:
                heapq.heappush(small_heapq, -num)
            else:
                heapq.heappush(big_heapq, num)

            # 平衡两个堆，最小堆的最大数量可以比最大堆多1个
            while len(small_heapq) > len(big_heapq) + 1:
                heapq.heappush(big_heapq, -heapq.heappop(small_heapq))

            while len(big_heapq) > len(small_heapq):
                heapq.heappush(small_heapq, -heapq.heappop(big_heapq))

        if not nums1 and not nums2:
            return 0

        small_heapq = []
        big_heapq = []

        i, j = 0, 0
        while i < len(nums1) or j < len(nums2):
            if i == len(nums1):
                add_num(nums2[j])
                j += 1
                continue
            if j == len(nums2):
                add_num(nums1[i])
                i += 1
                continue

            if nums1[i] <= nums2[j]:
                add_num(nums1[i])
                i += 1
            else:
                add_num(nums2[j])
                j += 1

        print(small_heapq, big_heapq)

        if len(small_heapq) == len(big_heapq):
            return (-small_heapq[0] + big_heapq[0]) / 2
        else:
            return -small_heapq[0]

    def subarraysDivByK(self, A: list, K: int) -> int:
        """
        974. 和可被 K 整除的子数组
        :see https://leetcode-cn.com/problems/subarray-sums-divisible-by-k/
        """
        # 类似“560. 和为K的子数组”
        sum_dict = {0: 1}
        result = 0

        num_sum = 0
        for i in A:
            num_sum += i
            # 此处只需要存余数即可，因为当余数相同的两个数字相减，即可被K整除
            key = num_sum % K
            count = sum_dict.get(key, 0)
            result += count
            sum_dict[key] = count + 1

        return result

    def countArrangement(self, N: int) -> int:
        """
        526. 优美的排列
        :see https://leetcode-cn.com/problems/beautiful-arrangement/
        """

        def check(index: int, value: int) -> bool:
            # 判断是否满足条件
            return index % value == 0 if index >= value else value % index == 0

        def backtrace(index: int, nums: list):
            nonlocal result
            if index == N + 1:
                result += 1
                return

            for i in range(N):
                if nums[i] == 0 and check(index, i + 1):
                    nums[i] = index
                    backtrace(index + 1, nums)
                    nums[i] = 0

        result = 0
        backtrace(1, [0] * N)
        return result

    def canBeEqual(self, target: list, arr: list) -> bool:
        """
        5408. 通过翻转子数组使两个数组相等
        :param target:
        :param arr:
        :return:
        """
        return sorted(target) == sorted(arr)

    def maxProduct(self, nums: list) -> int:
        """
        5424. 数组中两元素的最大乘积
        :param nums:
        :return:
        """
        result = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                result = max(result, (nums[i] - 1) * (nums[j] - 1))
        return result

    def maxArea(self, h: int, w: int, horizontalCuts: list, verticalCuts: list) -> int:
        """
        5425. 切割后面积最大的蛋糕
        :param h:
        :param w:
        :param horizontalCuts:
        :param verticalCuts:
        :return:
        """
        horizontalCuts.sort()
        verticalCuts.sort()

        horizontal = horizontalCuts[0]
        for i in range(len(horizontalCuts)):
            if horizontalCuts[i] - horizontalCuts[i - 1] > horizontal:
                horizontal = horizontalCuts[i] - horizontalCuts[i - 1]
        if h - horizontalCuts[-1] > horizontal:
            horizontal = h - horizontalCuts[-1]

        vertical = verticalCuts[0]
        for i in range(len(verticalCuts)):
            if verticalCuts[i] - verticalCuts[i - 1] > vertical:
                vertical = verticalCuts[i] - verticalCuts[i - 1]
        if w - verticalCuts[-1] > vertical:
            vertical = w - verticalCuts[-1]

        return horizontal * vertical % (10 ** 9 + 7)

    def spiralOrder(self, matrix: list) -> list:
        """
        面试题29. 顺时针打印矩阵/54. 螺旋矩阵
        :param matrix:
        :see https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/
        :see https://leetcode-cn.com/problems/spiral-matrix/
        """
        if not matrix:
            return []

        result = []
        length = len(matrix)
        width = len(matrix[0])
        for i in range((length + 1) // 2):
            # 1. 从左到右打印上方的一行，起点(i, i)，终点(i, width - i - 1)
            for j in range(i, len(matrix[i]) - i):
                # print(1, i, j)
                result.append(matrix[i][j])

            if i < width - i:
                # 2. 从上到下打印右侧的一列，起点(i + 1, width - i - 1)，终点(length - i - 2, width - i - 1)
                for j in range(i + 1, length - i - 1):
                    # print(2, j, width - i - 1)
                    result.append(matrix[j][width - i - 1])

            if i < length - i - 1:
                # 3. 从右到左打印下方的一行，起点(length - i - 1, width - i - 1)，终点(length - i - 1, i)
                for j in range(width - i - 1, i - 1, -1):
                    # print(3, length - i - 1, j)
                    result.append(matrix[length - i - 1][j])

            if i < width - i - 1:
                # 4. 从下到上打印左侧的一列，起点(length - i - 2, i)，终点(i + 1, i)
                for j in range(length - i - 2, i, -1):
                    # print(4, j, i)
                    result.append(matrix[j][i])

        return result

    def longestConsecutive(self, nums: list) -> int:
        """
        128. 最长连续序列
        :see https://leetcode-cn.com/problems/longest-consecutive-sequence/
        """
        nums_dict = {}
        for i in nums:
            nums_dict[i] = nums_dict.get(i, 1)

        max_length = 0
        for i in nums_dict:
            x = i + 1
            while x in nums_dict:
                x += nums_dict[x]
            nums_dict[i] = x - i
            max_length = max(max_length, nums_dict[i])

        return max_length

    def shuffle(self, nums: list, n: int) -> list:
        """
        5428. 重新排列数组
        :param nums:
        :param n:
        :return:
        """
        return [nums[i // 2] if i % 2 == 0 else nums[i // 2 + n] for i in range(2 * n)]

    def getStrongest(self, arr: list, k: int) -> list:
        """
        5429. 数组中的 k 个最强值
        :param arr:
        :param k:
        :return:
        """
        arr.sort()
        return sorted(arr, key=lambda x: (-abs(x - arr[(len(arr) - 1) // 2]), -x))[:k]

    def equationsPossible(self, equations: list) -> bool:
        """
        990. 等式方程的可满足性
        :see https://leetcode-cn.com/problems/satisfiability-of-equality-equations/
        """
        # 考察并查集
        equal_dict = {}
        non_equal_list = []

        for s in equations:
            if s[1] == '!':
                non_equal_list.append((s[0], s[-1]))
                continue

            if s[0] in equal_dict and s[-1] in equal_dict and equal_dict[s[0]] != equal_dict[s[-1]]:
                # 将s[-1]所在集合，合并到s[0]
                value = equal_dict[s[-1]]
                for i in equal_dict:
                    if equal_dict[i] == value:
                        equal_dict[i] = equal_dict[s[0]]
            elif s[0] in equal_dict and s[-1] not in equal_dict:
                equal_dict[s[-1]] = equal_dict[s[0]]
            elif s[0] not in equal_dict and s[-1] in equal_dict:
                equal_dict[s[0]] = equal_dict[s[-1]]
            elif s[0] not in equal_dict and s[-1] not in equal_dict:
                equal_dict[s[0]] = s[0]
                equal_dict[s[-1]] = s[0]

        for start, end in non_equal_list:
            if start == end or (start in equal_dict and end in equal_dict and equal_dict[start] == equal_dict[end]):
                return False
        return True

    def removeElement(self, nums: list, val: int) -> int:
        """
        27. 移除元素
        :see https://leetcode-cn.com/problems/remove-element/
        """
        i, j = 0, len(nums) - 1
        while i <= j:
            if nums[i] == val:
                while j >= 0 and nums[j] == val:
                    j -= 1
                if i > j:
                    break
                nums[i] = nums[j]
                j -= 1
            i += 1
        return i

    def dailyTemperatures(self, T: list) -> list:
        """
        739. 每日温度
        :see https://leetcode-cn.com/problems/daily-temperatures/
        """
        result = [0] * len(T)

        stack = []
        for i, v in enumerate(T):
            while stack and stack[-1][1] < v:
                index, value = stack.pop()
                result[index] = i - index
            stack.append((i, v))

        while stack:
            index, value = stack.pop()
            result[index] = 0

        return result

    def threeSumClosest(self, nums: list, target: int) -> int:
        """
        16. 最接近的三数之和
        :see https://leetcode-cn.com/problems/3sum-closest/
        """
        result = float('inf')
        nums.sort()
        length = len(nums)

        for i in range(length - 2):
            left, right = i + 1, length - 1
            while left < right:
                arr_sum = sum([nums[i], nums[left], nums[right]])
                if abs(arr_sum - target) < abs(result - target):
                    result = arr_sum

                if arr_sum < target:
                    left += 1
                elif arr_sum > target:
                    right -= 1
                else:
                    return arr_sum

        return result

    def fourSum(self, nums: list, target: int) -> list:
        """
        18. 四数之和
        :see https://leetcode-cn.com/problems/4sum/
        """
        nums.sort()
        result = []
        length = len(nums)

        for i in range(length - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            for j in range(i + 1, length - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                left = j + 1
                right = length - 1
                while left < right:
                    arr_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    if arr_sum < target:
                        left += 1
                    elif arr_sum > target:
                        right -= 1
                    else:
                        result.append([nums[i], nums[j], nums[left], nums[right]])

                        left += 1
                        while left < right and nums[left] == nums[left - 1]:
                            left += 1

                        right -= 1
                        while left < right and nums[right] == nums[right + 1]:
                            right -= 1

        return result

    def fourSumCount(self, A: list, B: list, C: list, D: list) -> int:
        """
        454. 四数相加 II
        :see https://leetcode-cn.com/problems/4sum-ii/
        """
        AB_dict = {}
        for i in A:
            for j in B:
                s = i + j
                AB_dict[s] = AB_dict.get(s, 0) + 1

        result = 0
        for i in C:
            for j in D:
                if -i - j in AB_dict:
                    result += AB_dict[-i - j]

        return result

    def letterCombinations(self, digits: str) -> list:
        """
        17. 电话号码的字母组合
        :see https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/
        """

        def backtrace(index: int, str_list: list):
            if index == len(digits):
                result.append(''.join(str_list))
                return

            for ch in num_str_dict[digits[index]]:
                str_list.append(ch)
                backtrace(index + 1, str_list)
                str_list.pop()

        if not digits:
            return []
        num_str_dict = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv',
                        '9': 'wxyz'}
        result = []
        backtrace(0, [])
        return result

    def generateMatrix(self, n: int) -> list:
        """
        59. 螺旋矩阵 II
        :see https://leetcode-cn.com/problems/spiral-matrix-ii/
        """
        grip = [[0] * n for _ in range(n)]

        num = 1
        for i in range(n):
            step = n - 2 * i - 1
            # 上方一行，从(i, i)到(i, n - i - 2)
            for j in range(step):
                grip[i][i + j] = num
                num += 1

            # 右方一列，从(i, n - i - 1)到(n - i - 2, n - i - 1)
            for j in range(step):
                grip[i + j][-i - 1] = num
                num += 1

            # 下方一行，从(n - i - 1, n - i - 1)到(n - i - 1, i + 1)
            for j in range(step):
                grip[- i - 1][-i - 1 - j] = num
                num += 1

            # 左方一列，从(n - i - 1, i)到(i + 1, i)
            for j in range(step):
                grip[-i - 1 - j][i] = num
                num += 1

        if n & 1:
            index = n // 2
            grip[index][index] = num

        return grip

    def searchMatrix(self, matrix: list, target: int) -> bool:
        """
        74. 搜索二维矩阵
        :see https://leetcode-cn.com/problems/search-a-2d-matrix/
        """
        if not matrix or not matrix[0]:
            return False

        row = 0
        while row < len(matrix) - 1:
            if matrix[row][0] == target:
                return True
            elif matrix[row + 1][0] <= target:
                row += 1
            else:
                break

        arr = matrix[row]
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return True
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return False

    def searchMatrixII(self, matrix, target) -> bool:
        """
        240. 搜索二维矩阵 II
        :see https://leetcode-cn.com/problems/search-a-2d-matrix-ii/
        """
        if not matrix or not matrix[0]:
            return False

        i, j = 0, len(matrix[0]) - 1
        while 0 <= i < len(matrix) and 0 <= j < len(matrix[0]):
            if target > matrix[i][j]:
                i += 1
            elif target < matrix[i][j]:
                j -= 1
            else:
                return True

        return False

    def sortColors(self, nums: list) -> None:
        """
        75. 颜色分类
        :see https://leetcode-cn.com/problems/sort-colors/
        """
        """
        zero, one, two = 0, 0, 0
        for i in nums:
            if i == 0:
                zero += 1
            elif i == 1:
                one += 1
            elif i == 2:
                two += 1

        for i in range(len(nums)):
            if zero > 0:
                nums[i] = 0
                zero -= 1
            elif one > 0:
                nums[i] = 1
                one -= 1
            elif two > 0:
                nums[i] = 2
                two -= 1

        print(nums)
        """
        # 参考 partition 过程
        # [0, left): < 1
        # [left, i): = 1
        # [i, right): 待检测
        # [right, len): > 1

        left, right = 0, len(nums)
        i = 0

        while i < right:
            if nums[i] < 1:
                nums[left], nums[i] = nums[i], nums[left]
                left += 1
                i += 1
            elif nums[i] == 1:
                i += 1
            else:
                right -= 1
                nums[i], nums[right] = nums[right], nums[i]
            print(left, i, right, nums)

    def subsets(self, nums: list) -> list:
        """
        78. 子集
        :see https://leetcode-cn.com/problems/subsets/
        """

        def backtrace(index: int, arr: list):
            result.append(arr)
            for i in range(index, len(nums)):
                backtrace(i + 1, arr + [nums[i]])

        result = []
        backtrace(0, [])
        return result

    def finalPrices(self, prices: list) -> list:
        """
        5420. 商品折扣后的最终价格
        :see https://leetcode-cn.com/problems/final-prices-with-a-special-discount-in-a-shop/
        """
        if len(prices) == 1:
            return prices

        result = []
        for i in range(len(prices)):
            cost = 0
            for j in range(i + 1, len(prices)):
                if prices[j] <= prices[i]:
                    cost = prices[j]
                    break
            result.append(prices[i] - cost)

        return result

    def minSumOfLengths(self, arr: list, target: int) -> int:
        """
        找两个和为目标值且不重叠的子数组
        :see https://leetcode-cn.com/problems/find-two-non-overlapping-sub-arrays-each-with-target-sum/
        """
        sum_dict = {0: -1}
        s = 0
        for i, v in enumerate(arr):
            s += v
            sum_dict[s] = i

        dis_arr = []

        for i in sum_dict:
            if i + target in sum_dict:
                dis_arr.append((sum_dict[i + target] - sum_dict[i], sum_dict[i], sum_dict[i + target]))

        if len(dis_arr) < 2:
            return -1

        dis_arr.sort()
        length = 0
        flag = False
        a, b = 0, 0
        for i in range(len(dis_arr)):
            if i == 0:
                length += dis_arr[i][0]
                a, b = dis_arr[i][1], dis_arr[i][2]
            elif dis_arr[i][1] >= b or dis_arr[i][2] <= a:
                length += dis_arr[i][0]
                flag = True
                break

        return length if flag else -1

    def findBestValue(self, arr: list, target: int) -> int:
        """
        1300. 转变数组后最接近目标值的数组和
        :see https://leetcode-cn.com/problems/sum-of-mutated-array-closest-to-target/
        """
        arr.sort()

        index = 0
        s = target
        while index < len(arr):
            value = round(s / (len(arr) - index))
            if value <= arr[index]:
                return value
            s -= arr[index]
            index += 1

        return arr[-1]

    def runningSum(self, nums: list) -> list:
        """
        5436. 一维数组的动态和
        :see https://leetcode-cn.com/problems/running-sum-of-1d-array/
        """
        result = [0]
        for i in nums:
            result.append(result[-1] + i)
        return result[1:]

    def findLeastNumOfUniqueInts(self, arr: list, k: int) -> int:
        """
        5437. 不同整数的最少数目
        :see https://leetcode-cn.com/problems/least-number-of-unique-integers-after-k-removals/
        """
        nums_dict = {}
        for i in arr:
            nums_dict[i] = nums_dict.get(i, 0) + 1

        nums_count_list = [nums_dict[i] for i in nums_dict]
        nums_count_list.sort()

        index = -1
        while k >= 0:
            index += 1
            if index >= len(nums_count_list):
                return 0

            k -= nums_count_list[index]

        return len(nums_count_list) - index

    def maxScoreSightseeingPair(self, A: list) -> int:
        """
        1014. 最佳观光组合
        :see https://leetcode-cn.com/problems/best-sightseeing-pair/
        """
        # A[i] + A[j] + i - j = (A[i] + i) + (A[j] - j)
        result = 0
        max_add = A[0] + 0
        for i in range(1, len(A)):
            result = max(result, max_add + (A[i] - i))
            max_add = max(max_add, A[i] + i)
        return result

    def xorOperation(self, n: int, start: int) -> int:
        """
        1486. 数组异或操作
        :see https://leetcode-cn.com/problems/xor-operation-in-an-array/
        """
        result = start
        for i in range(start + 2, start + 2 * n, 2):
            result ^= i
        return result

    def getFolderNames(self, names: list) -> list:
        """
        1487. 保证文件名唯一
        :see https://leetcode-cn.com/problems/making-file-names-unique/
        """
        checked_set = set()
        result = []
        for i in names:
            if i in checked_set:
                index = 1
                file = f'{i}({index})'
                while file in checked_set:
                    index += 1
                    file = f'{i}({index})'
                result.append(file)
                checked_set.add(file)
            else:
                result.append(i)
                checked_set.add(i)
        return result

    def avoidFlood(self, rains: list) -> list:
        """
        1488. 避免洪水泛滥
        :see https://leetcode-cn.com/problems/avoid-flood-in-the-city/
        """
        # 存储池子上一次被填满时的天数
        lakes_dict = {}
        # (开始时间，截止时间，湖泊编号)
        lakes_queue = []
        for i, v in enumerate(rains):
            if v > 0 and v in lakes_dict:
                lakes_queue.append((lakes_dict[v], i, v))
            lakes_dict[v] = i

        # 按照可被抽干的开始时间排序，以免还没填满时就被抽水
        lakes_queue.sort(key=lambda x: x[0])

        result = []
        # 用来存储连续晴天的天数
        zero_count = 0

        for i in range(len(rains)):
            if rains[i] == 0:
                zero_count += 1
            else:
                # 晴天后的第一个雨天结算前，先计算前几个晴天需要抽哪几个池子的水
                if zero_count > 0:
                    # 取出所有满足开始时间的池子
                    queue = PriorityQueue()
                    while lakes_queue:
                        if i <= lakes_queue[0][0]:
                            break
                        elif i > lakes_queue[0][1]:
                            # 如果发现任意一个池子的过期时间已过，则返回空数组
                            return []

                        lake = lakes_queue.pop(0)
                        # 用截止时间作为优先级
                        queue.put((lake[1], lake))

                    # 抽干优先级最高的池子
                    while zero_count > 0:
                        if queue.empty():
                            # 没池子可抽水时则填入1
                            result.append(1)
                        else:
                            result.append(queue.get()[1][2])
                        zero_count -= 1

                    # 将剩下的池子放回队列中，待下次抽干
                    while not queue.empty():
                        lakes_queue.insert(0, queue.get()[1])

                # 晴天不能抽水
                result.append(-1)

        # 如果还有池子没被抽干，则说明不满足条件；如果 zero_count > 0，则说明最后还有几天晴天且没池子可以抽水
        return result + [1] * zero_count if not lakes_queue else []

    def average(self, salary: list) -> float:
        """
        5432. 去掉最低工资和最高工资后的工资平均值
        :param salary:
        :return:
        """
        return (sum(salary) - max(salary) - min(salary)) / (len(salary) - 2)

    def longestSubarray(self, nums: list) -> int:
        """
        5434. 删掉一个元素以后全为 1 的最长子数组
        :param nums:
        :return:
        """
        last_zero_index = -1
        zero_index_list = []
        for i in range(len(nums)):
            if nums[i] == 0:
                zero_index_list.append(i - last_zero_index - 1)
                last_zero_index = i
        zero_index_list.append(len(nums) - last_zero_index - 1)

        print(zero_index_list)

        if len(zero_index_list) == 1:
            return len(nums) - 1

        max_length = 0
        for i in range(len(zero_index_list) - 1):
            max_length = max(max_length, zero_index_list[i] + zero_index_list[i + 1])

        return max_length

    def minSubArrayLen(self, s: int, nums: list) -> int:
        """
        209. 长度最小的子数组
        :see https://leetcode-cn.com/problems/minimum-size-subarray-sum/
        """
        '''# 滑动窗口
        if not nums:
            return 0

        length = len(nums)
        left = 0
        right = 0
        total = nums[0]

        if total >= s:
            return 1
        result = float('inf')

        while left < length:
            if total >= s or right == length - 1:
                total -= nums[left]
                left += 1
            elif right < length - 1:
                right += 1
                total += nums[right]

            if total >= s:
                result = min(result, right - left + 1)
                if result == 1:
                    return 1
            print(left, right, nums[left:right + 1], total)

        return result if result <= length else 0
        '''
        # 前缀和 + 二分查找
        total = 0
        total_list = []
        for i in nums:
            total += i
            total_list.append(total)

        if total < s:
            return 0

        result = float('inf')
        for i, v in enumerate(total_list):
            if v >= s:
                result = min(result, i - bisect.bisect_right(total_list, v - s, 0, i) + 1)
            # print(i, v, target, index, result)

        return result

    def canArrange(self, arr: list, k: int) -> bool:
        """
        5449. 检查数组对是否可以被 k 整除
        :see https://leetcode-cn.com/problems/check-if-array-pairs-are-divisible-by-k/
        """
        k_list = [0 for i in range(k)]
        for i in arr:
            k_list[i % k] += 1

        print(k_list)
        if k_list[0] % 2 != 0:
            return False

        for i in range(1, k // 2):
            if k_list[i] != k_list[k - i]:
                return False

        return True

    def findMaxValueOfEquation(self, points: list, k: int) -> int:
        """
        5451. 满足不等式的最大值
        :see https://leetcode-cn.com/problems/max-value-of-equation/
        """
        # 单调栈
        queue = [(points[0][0], points[0][0] - points[0][1])]
        result = -float('inf')

        for i in range(1, len(points)):
            # 移除 xj - xi > k 的节点
            while queue and points[i][0] - queue[0][0] > k:
                queue.pop(0)

            # 如果全部节点都被移除了，说明该节点找不到前面的一个节点满足 xj - xi <= k
            if queue:
                result = max(result, points[i][0] + points[i][1] - queue[0][1])

            # 将这个节点的 xj - xi 加入单调栈
            a = points[i][0] - points[i][1]
            while queue and queue[-1][1] > a:
                queue.pop()
            queue.append((points[i][0], a))

            # print(i, result, queue)

        return result

    def maximumGap(self, nums: list) -> int:
        """
        164. 最大间距
        :see https://leetcode-cn.com/problems/maximum-gap/
        """
        nums.sort()
        result = 0
        for i in range(len(nums) - 1):
            result = max(result, nums[i + 1] - nums[i])
        return result

    def kthSmallest(self, matrix: list, k: int) -> int:
        """
        378. 有序矩阵中第K小的元素
        :see https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/
        """
        '''
        # 利用最大堆实现
        heap = []
        for i in matrix:
            for j in i:
                if len(heap) < k:
                    heapq.heappush(heap, -j)
                elif j < -heap[0]:
                    heapq.heapreplace(heap, -j)
        return -heap[0]
        '''
        '''
        # 利用二分查找实现
        def count(target: int) -> int:
            """ 判断矩阵中比Target小的元素有多少个 """
            i, j = 0, len(matrix[0]) - 1
            result = 0
            while i < len(matrix) and j >= 0:
                if matrix[i][j] <= target:
                    result += j + 1
                    i += 1
                else:
                    j -= 1
            return result

        left, right = matrix[0][0], matrix[-1][-1]
        while left < right:
            mid = (left + right) // 2
            if count(mid) < k:
                left = mid + 1
            else:
                right = mid
        return left
        '''
        # 利用优先队列实现
        length = len(matrix)
        queue = PriorityQueue()
        for i in range(length):
            queue.put((matrix[i][0], i, 0))

        for i in range(k - 1):
            line = queue.get()
            if (j := line[2] + 1) < length:
                queue.put((matrix[line[1]][j], line[1], j))
        return queue.get()[0]

    def canMakeArithmeticProgression(self, arr: list) -> bool:
        """
        5452. 判断能否形成等差数列
        :see https://leetcode-cn.com/problems/can-make-arithmetic-progression-from-sequence/
        """
        if len(arr) < 2:
            return True
        arr.sort()
        diff = arr[1] - arr[0]
        for i in range(1, len(arr) - 1):
            if arr[i + 1] - arr[i] != diff:
                return False
        return True

    def getLastMoment(self, n: int, left: list, right: list) -> int:
        """
        5453. 所有蚂蚁掉下来前的最后一刻
        :see https://leetcode-cn.com/problems/last-moment-before-all-ants-fall-out-of-a-plank/
        """
        return max(max(left) if left else 0, max([n - i for i in right]) if right else 0)

    def numSubmat(self, mat: list) -> int:
        """
        5454. 统计全 1 子矩形
        :see https://leetcode-cn.com/problems/count-submatrices-with-all-ones/
        """
        dp = [[(0, 0)] * len(m) for m in mat]

        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if mat[i][j] == 1:
                    if i == 0 and j == 0:
                        dp[0][0] = (1, 1)
                    elif i == 0:
                        dp[0][j] = (dp[0][j - 1][0] + 1, 1)
                    elif j == 0:
                        dp[i][0] = (1, dp[i - 1][0][1] + 1)
                    else:
                        dp[i][j] = (dp[i][j - 1][0] + 1, dp[i - 1][j][1] + 1)

        result = 0
        for i in range(len(dp)):
            for j in range(len(dp[i])):
                max_width = dp[i][j][0]
                for k in range(dp[i][j][1]):
                    max_width = min(max_width, dp[i - k][j][0])
                    result += max_width

        return result

    def divingBoard(self, shorter: int, longer: int, k: int) -> list:
        """
        面试题 16.11. 跳水板
        :see https://leetcode-cn.com/problems/diving-board-lcci/
        """
        # f(n) = [shorter * i + longer * (n - i)], 0 <= i <= n
        if k == 0:
            return []
        elif shorter == longer:
            return [shorter * k]
        else:
            return [shorter * (k - i) + longer * i for i in range(k + 1)]

    def rangeSum(self, nums: list, n: int, left: int, right: int) -> int:
        """
        5445. 子数组和排序后的区间和
        :see
        """
        sum_list = []
        for i in range(n):
            start = 0
            for j in range(i, n):
                start += nums[j]
                sum_list.append(start)
        sum_list.sort()
        print(sum_list)
        return sum(sum_list[left - 1:right])

    def minDifference(self, nums: list) -> int:
        """
        5446. 三次操作后最大值与最小值的最小差
        :see
        """
        # 情况一，删除 nums[0], nums[1], nums[2]
        # 情况二，删除 nums[0], nums[1], nums[-1]
        # 情况三，删除 nums[0], nums[-2], nums[-1]
        # 情况四，删除 nums[-3], nums[-2], nums[-1]
        if len(nums) < 5:
            return 0
        nums.sort()
        return min([nums[-1] - nums[3], nums[-2] - nums[2], nums[-3] - nums[1], nums[-4] - nums[0]])

    def numIdenticalPairs(self, nums: list) -> int:
        """
        好数对的数目
        :see
        """
        result = 0
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                if nums[i] == nums[j]:
                    result += 1
        return result

    def numSub(self, s: str) -> int:
        """
        仅含 1 的子串数
        :see
        """
        times = []
        a = 0
        for i in s:
            if i == '0':
                times.append(a)
                a = 0
            else:
                a += 1
        times.append(a)

        result = 0
        for i in times:
            result += (1 + i) * i // 2
        return result % (10 ** 9 + 7)


if __name__ == '__main__':
    s = Solution()
    # print(s.avoidFlood([3, 0, 2, 0, 2, 3]))
    # print(s.avoidFlood([2, 3, 0, 3, 0, 2]))
    print(s.numSub(''))
