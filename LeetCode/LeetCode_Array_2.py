import heapq


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


if __name__ == '__main__':
    s = Solution()
    print(s.longestConsecutive([3, 1, 4, 2]))
