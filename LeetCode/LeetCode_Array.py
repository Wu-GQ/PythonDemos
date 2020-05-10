import bisect
import heapq
from functools import reduce


class Solution:
    def move_zeroes(self, nums: list) -> None:
        """
        移动零
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/264/array/1130/
        """

        def update_first_zero_index(nums: list, start_index: int) -> int:
            for i in range(start_index, len(nums)):
                if nums[i] == 0:
                    return i
            return len(nums) - 1

        # 检查第一个0的位置
        zero_start = update_first_zero_index(nums, 0)

        # 从第一个0的位置开始遍历
        for i in range(zero_start, len(nums)):
            if zero_start == len(nums) - 1:
                break
            elif nums[i] != 0:
                nums[zero_start] = nums[i]
                nums[i] = 0

                # 更新第一个0的位置
                zero_start = update_first_zero_index(nums, zero_start)

    def increasing_triplet(self, nums: list) -> bool:
        """
        递增的三元子序列
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/264/array/1133/
        """
        count_dict = {}
        for i in nums:
            max_count = 1
            for j in count_dict:
                if j < i and max_count <= count_dict[j]:
                    max_count = count_dict[j] + 1

            if max_count == 3:
                return True

            count_dict[i] = max_count

        return False

    def contains_duplicate(self, nums: list) -> bool:
        """
        存在重复元素
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/264/array/1129/
        """
        nums_set = set()

        for i in nums:
            if i in nums_set:
                return True
            else:
                nums_set.add(i)
        return False

    def rotate(self, nums: list, k: int) -> None:
        """
        旋转数组
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/264/array/1128/
        """
        # MARK: 不符合空间复杂度的要求
        step = k % len(nums)
        other_list = nums[:len(nums) - step]
        for i in range(len(nums)):
            nums[i] = nums[i - step] if i < step else other_list[i - step]

    def intersect(self, nums1: list, nums2: list) -> list:
        """
        两个数组的交集II
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/264/array/1132/
        """
        if nums1 == [] or nums2 == []:
            return []
        count_dict = {}
        for i in nums1:
            if i in count_dict:
                count_dict[i] += 1
            else:
                count_dict[i] = 1

        result_list = []
        for i in nums2:
            if i in count_dict and count_dict[i] > 0:
                result_list.append(i)
                count_dict[i] -= 1

        return result_list

    def product_except_self(self, nums: list) -> list:
        """
        除自身以外数组的乘积
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/264/array/1135/
        """
        left_mul_list = []
        right_mul_list = [0 for i in range(len(nums))]

        for i in range(len(nums)):
            left_mul_list.append(nums[i] if i == 0 else nums[i] * left_mul_list[i - 1])

        for i in range(len(nums) - 1, -1, -1):
            right_mul_list[i] = nums[i] if i == len(nums) - 1 else nums[i] * right_mul_list[i + 1]

        result_list = []
        for i in range(len(nums)):
            if i == 0:
                result_list.append(right_mul_list[i + 1])
            elif i == len(nums) - 1:
                result_list.append(left_mul_list[i - 1])
            else:
                result_list.append(left_mul_list[i - 1] * right_mul_list[i + 1])

        return result_list

    def removeDuplicates(self, nums: list) -> int:
        """
        从排序数组中删除重复项
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/1/array/21/
        """
        slow_index = 0
        fast_index = 1
        while fast_index < len(nums):
            if nums[fast_index] != nums[slow_index]:
                slow_index += 1
                nums[slow_index] = nums[fast_index]
            fast_index += 1
        return slow_index + 1

    def plusOne(self, digits: list) -> list:
        """
        加一
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/1/array/27/
        """
        digits[len(digits) - 1] += 1
        for i in range(len(digits) - 1, 0, -1):
            if digits[i] > 9:
                digits[i - 1] += digits[i] // 10
                digits[i] %= 10
            else:
                break
        if digits[0] > 9:
            digits.insert(0, digits[0] // 10)
            digits[1] %= 10
        return digits

    def twoSum(self, nums: list, target: int) -> list:
        """
        两数之和
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/1/array/29/
        """
        nums_set = dict()

        for i in range(len(nums)):
            other_num = target - nums[i]
            if other_num in nums_set:
                return [nums_set[other_num], i]
            nums_set[nums[i]] = i
        return []

    def setZeroes(self, matrix: list):
        """
        矩阵置零
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-medium/29/array-and-strings/76/
        """
        if len(matrix) == 0:
            return

        zero_x_list = set()
        zero_y_list = set()

        m = len(matrix)
        n = len(matrix[0])

        # 遍历获得所有0的位置
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    zero_x_list.add(i)
                    zero_y_list.add(j)

        # 横向置0
        for x in zero_x_list:
            for i in range(n):
                matrix[x][i] = 0

        # 纵向置0
        for y in zero_y_list:
            for j in range(m):
                matrix[j][y] = 0

    def threeSum(self, nums: list) -> list:
        """
        三数之和
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-medium/29/array-and-strings/75/
        """
        result_list = []

        nums.sort()

        num_dict = {}
        for i in nums:
            if i in num_dict:
                num_dict[i] += 1
            else:
                num_dict[i] = 1

        search_set = set()

        for i in range(len(nums)):
            a = nums[i]

            old_b = float('inf')
            for j in range(i + 1, len(nums)):
                if i == j or old_b == nums[j]:
                    continue

                old_b = nums[j]

                b = nums[j]
                c = -a - b

                if c in num_dict:
                    num_dict[a] -= 1
                    num_dict[b] -= 1

                    if num_dict[c] > 0:
                        result = sorted([a, b, c])
                        result_string = f'{result[0]}_{result[1]}_{result[2]}'
                        if result_string not in search_set:
                            result_list.append(result)
                            search_set.add(result_string)

                    num_dict[a] += 1
                    num_dict[b] += 1

        return result_list

    def trap(self, height: list) -> int:
        """
        接雨水
        :see https://leetcode-cn.com/problems/trapping-rain-water/
        """
        max_height_list = []

        # 正序找出左侧的最大值
        max_height = 0
        for i in height:
            if i > max_height:
                max_height = i
            max_height_list.append(max_height)

        # 逆序找出右侧的最大值，并和左侧的最大值比较，保留两者较小的最大值
        max_height = 0
        for i in range(len(height) - 1, -1, -1):
            if height[i] > max_height:
                max_height = height[i]
            max_height_list[i] = min(max_height, max_height_list[i])

        # 求和
        return sum(max_height_list[i] - height[i] for i in range(0, len(height)))

    def groupAnagrams(self, strs: list) -> list:
        """
        字谜分组
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-medium/29/array-and-strings/77/
        """
        str_dict = {}
        for string in strs:
            sorted_str = "".join(sorted(string))
            if sorted_str in str_dict:
                str_dict[sorted_str].append(string)
            else:
                str_dict[sorted_str] = [string]

        return [str_dict[i] for i in str_dict]

    def permute(self, nums: list) -> list:
        """
        全排列
        :see https://leetcode-cn.com/problems/permutations/
        """
        if len(nums) == 0:
            return []
        result = []

        def permutation(nums: list, start_index: int):
            if len(nums) == start_index:
                result.append(nums.copy())

            used_index_set = set()
            for i in range(start_index, len(nums)):
                if nums[i] in used_index_set:
                    continue
                else:
                    used_index_set.add(nums[i])

                # 把第i位放到最前面，然后对剩下的部分进行全排列
                nums[start_index], nums[i] = nums[i], nums[start_index]

                print(f'start_index: {start_index}, i: {i}, result: {nums}')

                # 注释掉下面这句递归的语句，再理解代码原理比较简单
                permutation(nums, start_index + 1)

                nums[start_index], nums[i] = nums[i], nums[start_index]

        permutation(nums, 0)

        return result

    def nextPermutation(self, nums: list) -> None:
        """
        下一个排列
        :see https://leetcode-cn.com/problems/next-permutation/
        """
        # 从数组的末尾开始寻找第一组数字，使得 i > j 且 nums[i] > nums[j]
        length: int = len(nums)
        if length < 2:
            return

        # 从后往前，找第一次变小的数字，如果没有，则翻转数组
        tmp_num: int = nums[-1]
        i: int = length
        while i > 0:
            i -= 1

            if nums[i] >= tmp_num:
                tmp_num = nums[i]
            else:
                break

        if i == 0 and nums[0] >= tmp_num:
            nums.reverse()
            return

        # 从i的位置开始寻找，找到最接近tmp_num且大于tmp_num的数字
        tmp_num = nums[i]
        j = i
        while j < length - 1:
            if nums[j + 1] <= tmp_num:
                break
            j += 1

        # 交换nums[i]和nums[j]
        nums[i], nums[j] = nums[j], nums[i]

        # 旋转nums[i + 1:]的数组
        right_nums = nums[i + 1:]
        right_nums.reverse()
        nums[i + 1:] = nums[length - 1:i:-1]

    def solveNQueens(self, n: int) -> list:
        """
        N皇后
        :see https://leetcode-cn.com/problems/n-queens/
        """
        if n < 1:
            return []

        result_list = []

        def is_exist_queen(queens_list: list, row: int, column: int) -> bool:
            """ 判断对角线和纵线上是否存在皇后 """
            for i in range(0, row):
                diff = row - i
                if (column - diff >= 0 and queens_list[i][column - diff] == 'Q') \
                        or queens_list[i][column] == 'Q' \
                        or (column + diff < n and queens_list[i][column + diff] == 'Q'):
                    return True
            return False

        def backtrace(line: int, queens_list: list):
            """ 回溯第line行的可能性 """
            if line == n:
                result_list.append(queens_list.copy())
                return

            # 在第line行从0~n逐个试，能否加入皇后。如果可以，则加入皇后，继续回溯；否则i++
            for i in range(n):
                # 判断能否加入皇后
                if is_exist_queen(queens_list, line, i):
                    continue

                # 最后一行加入皇后
                queen_string: str = f"{'.' * i}Q{'.' * (n - i - 1)}"
                queens_list.append(queen_string)

                # 回溯
                backtrace(line + 1, queens_list)

                # 删除最后一行
                queens_list.pop()

        backtrace(0, [])

        return result_list

    def combine(self, n: int, k: int) -> list:
        """
        组合
        :see https://leetcode-cn.com/problems/combinations/ 
        """
        if k < 0 or n < 0:
            return []

        if k >= n:
            return [[i for i in range(1, n + 1)]]

        result_list = []

        def backtrace(index: int, num_list: list):
            if len(num_list) == k:
                result_list.append(num_list.copy())
                return

            for i in range(index, n + 1):
                num_list.append(i)

                backtrace(i + 1, num_list)

                num_list.pop()

        backtrace(1, [])

        return result_list

    def generateParenthesis(self, n: int) -> list:
        """
        22. 括号生成
        :see https://leetcode-cn.com/problems/generate-parentheses/
        """
        if n < 1:
            return []

        result_list = []

        def backtrace(string: str, left_count: int, right_count: int):
            if len(string) == 2 * n:
                result_list.append(string)
                return

            if right_count < left_count < n:
                backtrace(f'{string}(', left_count + 1, right_count)
                backtrace(f'{string})', left_count, right_count + 1)
            elif left_count == 0 or left_count == right_count:
                backtrace(f'{string}(', left_count + 1, right_count)
            elif left_count == n:
                backtrace(f'{string})', left_count, right_count)

        backtrace("", 0, 0)

        return result_list

    def validateStackSequences(self, pushed: list, popped: list) -> bool:
        """
        验证栈序列
        :see https://leetcode-cn.com/problems/validate-stack-sequences/
        """
        if len(pushed) != len(popped):
            return False

        # -1时为了防止temp_stack_list[-1]崩溃
        temp_stack_list = [-1]

        for j in popped:
            if temp_stack_list[-1] == j:
                # 当栈顶元素相同时，先出栈
                temp_stack_list.pop()
                continue

            # 栈顶元素不匹配，则将pushed的栈顶元素入栈；否则将pushed的栈顶元素去除
            while len(pushed) > 0:
                value = pushed.pop(0)
                if value != j:
                    temp_stack_list.append(value)
                else:
                    break

        # 成功匹配的情况下，栈内只剩下一个-1
        return len(temp_stack_list) == 1

    def maxSlidingWindow(self, nums: list, k: int) -> list:
        """
        滑动窗口最大值
        :see https://leetcode-cn.com/problems/sliding-window-maximum/
        """
        if k > len(nums) or len(nums) == 0:
            return []

        # 用来保存最大值
        result_list = []
        # 用来保存每个窗口可能的最大值及其下标
        index_value_list = [(-1, -float('inf'))]

        # 先将前k个入队列
        for i in range(0, k):
            while len(index_value_list) > 0 and index_value_list[-1][1] <= nums[i]:
                index_value_list.pop()
            index_value_list.append((i, nums[i]))

        result_list.append(index_value_list[0][1])

        # 从第k个开始遍历
        for i in range(k, len(nums)):
            # 先去掉最前面的失效的元素
            while len(index_value_list) > 0 and index_value_list[0][0] <= i - k:
                index_value_list.pop(0)

            # 再去掉末尾小于当前元素的元素
            while len(index_value_list) > 0 and index_value_list[-1][1] <= nums[i]:
                index_value_list.pop()
            index_value_list.append((i, nums[i]))

            # 队列中的首元素即为最大元素
            result_list.append(index_value_list[0][1])

        return result_list

    def canJump(self, nums: list) -> bool:
        """
        55. 跳跃游戏
        :see https://leetcode-cn.com/problems/jump-game/
        """
        # # 参考深度优先遍历（超时）
        # length = len(nums)
        # if length < 2:
        #     return True
        #
        # # 判断是否被检测过
        # result_list = [False] * length
        #
        # def jump(start_index: int) -> bool:
        #     if start_index >= length - 1:
        #         return True
        #
        #     for i in range(start_index + nums[start_index], start_index, -1):
        #         if i >= length - 1:
        #             return True
        #
        #         if result_list[i]:
        #             return False
        #
        #         if jump(i):
        #             return True
        #         else:
        #             result_list[i] = True
        #
        #     return False
        #
        # return jump(0)

        # 贪心算法，只保存右侧最大可达的坐标
        max_length = 0
        for i in range(len(nums)):
            if i > max_length:
                break
            max_length = max(max_length, i + nums[i])
            if max_length >= len(nums) - 1:
                return True
        return False

    def jump(self, nums: list) -> int:
        """
        跳跃游戏 II
        :see https://leetcode-cn.com/problems/jump-game-ii/
        """
        # 贪心算法，与跳跃游戏类似
        # 当前步数
        step = 0
        # 这一步的最远距离
        distance = 0
        # 下一步可以跳到的最远距离
        max_length = 0

        for i in range(len(nums) - 1):
            max_length = max(max_length, i + nums[i])
            if i == distance:
                distance = max_length
                step += 1

        return step

    def canReach(self, arr: list, start: int) -> bool:
        """
        跳跃游戏 III
        :see https://leetcode-cn.com/problems/jump-game-iii/
        """
        length = len(arr)
        if start < 0 or start >= length:
            return False

        # False代表未检测， True代表已检测且不能到终点
        can_reach_list = [False] * length

        def reach(index: int) -> bool:
            if index < 0 or index >= length or can_reach_list[index]:
                return False
            if arr[index] == 0:
                return True

            # index已被检测，不再重复检测
            can_reach_list[index] = True

            return reach(index + arr[index]) or reach(index - arr[index])

        return reach(start)

    def maxJumps(self, arr: list, d: int) -> int:
        """
        跳跃游戏 V
        :see https://leetcode-cn.com/problems/jump-game-v/
        """
        if d < 1:
            return 0

        length = len(arr)
        if length < 1:
            return 0
        elif length == 1:
            return 1

        max_step_list = [-1] * length

        def max_step_for_index(index: int) -> int:
            if index < 0 or index >= length:
                return 0

            if max_step_list[index] >= 0:
                return max_step_list[index]

            max_step = 0
            # 先求左边的最大步数
            for i in range(index - 1, index - d - 1, -1):
                if i < 0 or arr[i] >= arr[index]:
                    break
                else:
                    max_step = max(max_step, max_step_for_index(i))

            # 再求右边的最大步数
            for i in range(index + 1, index + d + 1):
                if i >= length or arr[i] >= arr[index]:
                    break
                else:
                    max_step = max(max_step, max_step_for_index(i))

            max_step_list[index] = max_step + 1

            return max_step_list[index]

        max_step = 0
        for i in range(0, length):
            if max_step_list[i] == -1:
                max_step_list[i] = max_step_for_index(i)
            max_step = max(max_step, max_step_list[i])

        return max_step

    def canCross(self, stones: list) -> bool:
        """
        青蛙过河
        :see https://leetcode-cn.com/problems/frog-jump/
        """
        length = len(stones)
        if length < 2:
            return False

        # 元组的第一位表示index，第二位表示打到该位置所用的距离
        stone_list = [(0, 0)]
        checked_stone_set = set()

        while len(stone_list) > 0:
            next_stone = stone_list.pop(0)

            for i in range(next_stone[0] + 1, stones[next_stone[0]] + next_stone[1] + 2):
                if i >= length:
                    break
                distance = stones[i] - stones[next_stone[0]]
                if (i, distance) in checked_stone_set:
                    continue

                if distance == next_stone[1] - 1 or distance == next_stone[1] or distance == next_stone[1] + 1:
                    if i == length - 1:
                        return True
                    stone_list.append((i, distance))
                    checked_stone_set.add((i, distance))

            print(next_stone[0], next_stone[1])

        return False

    def wiggleSort(self, nums: list) -> None:
        """
        324. 摆动排序 II
        :see https://leetcode-cn.com/problems/wiggle-sort-ii/
        """
        nums.sort(reverse=True)

        big_nums = nums[:len(nums) // 2]
        small_nums = nums[len(nums) // 2:]
        for i in range(0, len(nums)):
            nums[i] = big_nums[i >> 1] if i & 1 == 1 else small_nums[i >> 1]

    def merge(self, A: list, m: int, B: list, n: int) -> None:
        """
        面试题 10.01. 合并排序的数组
        :see https://leetcode-cn.com/problems/sorted-merge-lcci/
        """
        if len(A) < m + n:
            return

        i, j = m - 1, n - 1
        while j >= 0:
            if i >= 0 and A[i] >= B[j]:
                A[i + j + 1] = A[i]
                i -= 1
            else:
                A[i + j + 1] = B[j]
                j -= 1
        print(A)

    def findContinuousSequence(self, target: int) -> list:
        """
        面试题57 - II. 和为s的连续正数序列
        :see https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/
        """
        if target < 3:
            return []

        start = 1
        end = 2

        result = []

        while start <= target and start <= end:
            list_sum = sum(range(start, end + 1))

            if list_sum == target:
                result.append([i for i in range(start, end + 1)])
                start += 1
            elif list_sum > target:
                end -= 1
            elif list_sum < target < list_sum + end + 1:
                start += 1
            else:
                end += 1

        return result

    def canThreePartsEqualSum(self, A: list) -> bool:
        """
        1013. 将数组分成和相等的三个部分
        :see https://leetcode-cn.com/problems/partition-array-into-three-parts-with-equal-sum/
        """
        if len(A) < 3:
            return False

        sum_list = []
        for i in range(len(A)):
            sum_list.append(A[i] if i == 0 else (A[i] + sum_list[i - 1]))

        total_sum = sum_list[-1]
        if total_sum % 3 != 0:
            return False

        result = int(total_sum / 3)
        first_index = -1
        second_index = -1

        for i in range(len(sum_list)):
            if sum_list[i] == result and first_index == -1 and second_index == -1:
                first_index = i
            elif sum_list[i] == 2 * result and first_index != -1:
                second_index = i

            if second_index != -1 and second_index < len(sum_list) - 1:
                return True
        return False

    def majorityElement(self, nums: list) -> int:
        """
        169. 多数元素
        :see https://leetcode-cn.com/problems/majority-element/
        """
        # 最大次数的数字和出现的次数
        max_count_num = 0
        max_count = 0

        for i in nums:
            if max_count == 0:
                max_count_num = i
                max_count += 1
            elif i == max_count_num:
                max_count += 1
            else:
                max_count -= 1

        return max_count_num

    count = 0

    def reversePairs(self, nums: list) -> int:
        """
        面试题51. 数组中的逆序对
        :see https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/
        """

        # 利用归并排序的思想
        def part_list(sub_nums: list) -> list:
            if len(sub_nums) < 2:
                return sub_nums
            return merge_list(part_list(sub_nums[:len(sub_nums) // 2]), part_list(sub_nums[len(sub_nums) // 2:]))

        def merge_list(nums1: list, nums2: list) -> list:
            result_list = []
            i, j = 0, 0

            while i < len(nums1) or j < len(nums2):
                if i == len(nums1):
                    result_list += nums2[j:]
                    break
                elif j == len(nums2):
                    result_list += nums1[i:]
                    break
                elif nums1[i] > nums2[j]:
                    result_list.append(nums1[i])
                    i += 1
                    self.count += len(nums2) - j
                else:
                    result_list.append(nums2[j])
                    j += 1

            # print(self.count, result_list)

            return result_list

        # print(part_list(nums))
        part_list(nums)

        return self.count
        """
        # 利用二分插入的原理
        num_list = []
        count = 0
        for i in nums[::-1]:
            index = bisect.bisect_left(num_list, i)
            count += index
            num_list.insert(index, i)
        return count
        """

    def reversePairs2(self, nums: list) -> int:
        """
        面试题51. 数组中的逆序对
        :see https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/
        """
        num_list = []

        count = 0
        # result_list = []

        nums.reverse()

        for i in nums:
            index = bisect.bisect_left(num_list, i)
            num_list.insert(index, i)
            count += index
            # result_list.append(index)

        return count

    def getLeastNumbers(self, arr: list, k: int) -> list:
        """
        最小的k个数
        :see https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/
        """
        # return [] if k < 1 else heapq.nsmallest(k, arr)
        if k < 1:
            return []

        result = []
        for i in arr:
            if len(result) < k:
                heapq.heappush(result, -i)
            elif result[0] < -i:
                heapq.heapreplace(result, -i)
            print(result)
        return [-i for i in result]

    def quick_sort(self, array: list) -> None:
        """
        快速排序
        """

        # if len(array) < 2:
        #     return array
        # left_array = [i for i in array if i < array[0]]
        # right_array = [i for i in array if i > array[0]]
        # return self.quick_sort(left_array) + [array[0]] + self.quick_sort(right_array)

        def quick_sort(start: int, end: int) -> None:
            if end - start < 1:
                return

            target = array[start]
            left_index = start
            right_index = end

            while left_index < right_index:
                while right_index >= 0 and target <= array[right_index]:
                    right_index -= 1

                if left_index >= right_index:
                    break
                array[left_index], array[right_index] = array[right_index], array[left_index]
                # print(f'1: {left_index}, {right_index}, {array}')

                while left_index < len(array) and target > array[left_index]:
                    left_index += 1

                if left_index >= right_index:
                    break
                array[left_index], array[right_index] = array[right_index], array[left_index]
                # print(f'2: {left_index}, {right_index}, {array}')

            quick_sort(start, min(left_index, right_index))
            quick_sort(max(left_index, right_index) + 1, end)

        quick_sort(0, len(array) - 1)

    def massage(self, nums: list) -> int:
        """
        按摩师
        :see https://leetcode-cn.com/problems/the-masseuse-lcci/
        """
        # f(x) = max(f(x - 2), f(x - 3)) + nums[x]
        a, b, c = 0, 0, 0
        for i in nums: a, b, c = b, c, max(a, b) + i
        return max(b, c)

    def surfaceArea(self, grid: list) -> int:
        """
        892. 三维形体的表面积
        :see https://leetcode-cn.com/problems/surface-area-of-3d-shapes/
        """
        if not grid or not grid[0]:
            return 0

        # 每行的总面积
        total_height_per_row = [0] * len(grid)
        # 每列的总面积
        total_height_per_column = [0] * len(grid[0])
        # 非0的数量
        not_zero_count = 0

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                height = grid[i][j]

                total_height_per_row[i] += height if j == 0 else abs(height - grid[i][j - 1])
                if j == len(grid[i]) - 1:
                    total_height_per_row[i] += height

                total_height_per_column[j] += height if i == 0 else abs(height - grid[i - 1][j])
                if i == len(grid) - 1:
                    total_height_per_column[j] += height

                if height != 0:
                    not_zero_count += 1

        return sum(total_height_per_row) + sum(total_height_per_column) + 2 * not_zero_count

    def numRookCaptures(self, board: list) -> int:
        """
        999. 车的可用捕获量
        :see https://leetcode-cn.com/problems/available-captures-for-rook/
        """
        # 查找车的位置
        rook_x, rook_y = -1, -1
        for i in range(8):
            for j in range(8):
                if board[i][j] == 'R':
                    rook_x, rook_y = i, j
                    break

        # 四个方向查找是否有卒，顺序：上下左右
        count_list = ['.', '.', '.', '.']

        step = 1
        while step < 8:
            # 向上查找
            if count_list[0] == '.' and rook_y - step >= 0: count_list[0] = board[rook_x][rook_y - step]
            # 向下查找
            if count_list[1] == '.' and rook_y + step < 8: count_list[1] = board[rook_x][rook_y + step]
            # 向左查找
            if count_list[2] == '.' and rook_x - step >= 0: count_list[2] = board[rook_x - step][rook_y]
            # 向右查找
            if count_list[3] == '.' and rook_x + step < 8: count_list[3] = board[rook_x + step][rook_y]

            step += 1

        return count_list.count('p')

    def sortArray(self, nums: list) -> list:
        """
        912. 排序数组
        :see https://leetcode-cn.com/problems/sort-an-array/
        """
        if len(nums) < 2:
            return nums
        target = nums.pop(0)
        return self.sortArray([i for i in nums if i < target]) + [target] + self.sortArray(
            [i for i in nums if i >= target])

    def maxDepthAfterSplit(self, seq: str) -> list:
        """
        1111. 有效括号的嵌套深度
        :see https://leetcode-cn.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings/
        """
        """
        以下方法可进行简化，奇数层分配给a，偶数层给b即可
        """
        # a 和 b 的深度
        a_depth, b_depth = 0, 0
        # 上一个括号属于 a 或者 b
        a_or_b_stack = []

        result_list = []

        for i in seq:
            if i == '(':
                if a_depth <= b_depth:
                    result_list.append(0)
                    a_or_b_stack.append(0)
                    a_depth += 1
                else:
                    result_list.append(1)
                    a_or_b_stack.append(1)
                    b_depth += 1
            else:
                if not a_or_b_stack:
                    print('Error: ' + str(result_list))
                elif a_or_b_stack[-1] == 1:
                    result_list.append(1)
                    a_or_b_stack.pop()
                    b_depth -= 1
                elif a_or_b_stack[-1] == 0:
                    result_list.append(0)
                    a_or_b_stack.pop()
                    a_depth -= 1
            # print(i, result_list, a_or_b_stack)

        return result_list

    def expectNumber(self, scores: list) -> int:
        """
        LCP 11. 期望个数统计
        :see https://leetcode-cn.com/problems/qi-wang-ge-shu-tong-ji/
        """
        # 当同一相同的数字的期望为1，不同的每个数字的期望也为1
        # 因此，本题是求不同的数字个数
        return len(set(scores))

    def maxSatisfaction(self, satisfaction: list) -> int:
        """
        做菜顺序
        """
        max_result = -float('inf')
        satisfaction.sort()

        for i in range(len(satisfaction)):
            result = 0
            for j in range(len(satisfaction) - i):
                result += satisfaction[i + j] * (j + 1)

            if result >= max_result:
                max_result = result
            else:
                break

        return max_result if max_result > 0 else 0

    def minSubsequence(self, nums: list) -> list:
        """
        非递增顺序的最小子序列
        :param nums:
        :return:
        """
        nums_sum = sum(nums)
        nums.sort(reverse=True)

        result = []
        part_sum = 0

        for i in nums:
            part_sum += i
            result.append(i)
            if nums_sum < part_sum * 2:
                break
        return result

    def stringMatching(self, words: list) -> list:
        """
        5380. 数组中的字符串匹配
        :param words:
        :return:
        """
        words.sort(key=lambda word: len(word))

        result = []
        for i in range(0, len(words)):
            for j in range(i + 1, len(words)):
                if words[i] in words[j]:
                    result.append(words[i])
                    break
        return result

    def processQueries(self, queries: list, m: int) -> list:
        """
        5381. 查询带键的排列
        :param queries:
        :param m:
        :return:
        """
        num_index_dict = {i + 1: i for i in range(m)}

        result = []
        for num in queries:
            old_index = num_index_dict[num]
            for i in num_index_dict:
                if num_index_dict[i] < old_index:
                    num_index_dict[i] += 1
            num_index_dict[num] = 0
            result.append(old_index)
        return result

    def merge_intervals(self, intervals: list) -> list:
        """
        56. 合并区间
        :see https://leetcode-cn.com/problems/merge-intervals/
        """
        intervals.sort(key=lambda interval: (interval[0], interval[1]))
        result = []

        while intervals:
            interval = intervals.pop(0)
            if not result:
                result.append(interval)
                continue

            last_interval = result[-1]
            if last_interval[0] <= interval[0] <= last_interval[1]:
                last_interval[1] = max(interval[1], last_interval[1])
            else:
                result.append(interval)

        return result

    def insert_interval(self, intervals: list, newInterval: list) -> list:
        """
        57. 插入区间
        :see https://leetcode-cn.com/problems/insert-interval/
        """

        def append_interval(interval: list) -> None:
            if not result:
                result.append(interval)
                return

            last_interval = result[-1]
            if last_interval[0] <= interval[0] <= last_interval[1]:
                last_interval[1] = max(interval[1], last_interval[1])
            else:
                result.append(interval)

        # 获得插入的位置
        index = bisect.bisect_left(intervals, newInterval)

        result = intervals[:index]
        append_interval(newInterval)

        for i in range(index, len(intervals)):
            append_interval(intervals[i])

        return result

    def minCount(self, coins: list) -> int:
        """
        拿硬币
        :param coins:
        :return:
        """
        result = 0
        for i in coins:
            if i % 2 == 1:
                result += i // 2 + 1
            else:
                result += i // 2
        return result

    def getTriggerTime(self, increase: list, requirements: list) -> list:
        """剧情触发时间"""
        # result = [-1] * len(requirements)
        # current = [0, 0, 0]
        # for i in range(len(increase)):
        #     current[0] += increase[i][0]
        #     current[1] += increase[i][1]
        #     current[2] += increase[i][2]
        #
        #     for j in range(len(requirements)):
        #         if result[j] < 0 and requirements[j][0] <= current[0] and requirements[j][1] <= current[1] and requirements[j][2] <= current[2]:
        #             result[j] = i + 1
        #
        # return result
        property_a_list = [0]
        property_b_list = [0]
        property_c_list = [0]
        for i in range(len(increase)):
            property_a_list.append(property_a_list[-1] + increase[i][0])
            property_b_list.append(property_b_list[-1] + increase[i][1])
            property_c_list.append(property_c_list[-1] + increase[i][2])

        result = []
        for i in requirements:
            a = bisect.bisect_left(property_a_list, i[0])
            b = bisect.bisect_left(property_b_list, i[1])
            c = bisect.bisect_left(property_c_list, i[2])

            index = max(a, b, c)
            if index > len(increase):
                index = -1
            result.append(index)

        return result

    def minJump(self, jump: list) -> int:
        """最小跳跃次数"""
        # step = 1
        # last_max_length = 0
        # max_length = jump[0]
        #
        # while max_length < len(jump):
        #     length = max_length
        #     for i in range(last_max_length + 1, max_length):
        #         length = max(length, jump[i] + i)
        #
        #     last_max_length = max_length
        #     if jump[max_length] + max_length >= length:
        #         max_length = jump[max_length] + max_length
        #         step += 1
        #     else:
        #         max_length = length
        #         step += 2
        #
        #     print(last_max_length, max_length)
        # return step

        result = [1E6] * len(jump)
        result[0] = 0
        step = 1E6

        for i in range(len(jump)):
            max_length = i + jump[i]
            if max_length >= len(jump):
                step = min(step, result[i] + 1)
                continue

            result[max_length] = min(result[max_length], result[i] + 1)

            for j in range(i + 1, max_length):
                result[j] = min(result[j], result[i] + 2)

        return step

    def minStartValue(self, nums: list) -> int:
        """逐步求和得到正数的最小值"""
        min_value = nums[0]
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
            min_value = min(min_value, nums[i])
        return -min_value + 1 if min_value < 0 else 1

    fibonacci_list = [1, 1]

    def findMinFibonacciNumbers(self, k: int) -> int:
        """和为 K 的最少斐波那契数字数目"""
        while self.fibonacci_list[-1] < k:
            self.fibonacci_list.append(self.fibonacci_list[- 1] + self.fibonacci_list[- 2])

        step = 0
        while k > 0:
            index = bisect.bisect_right(self.fibonacci_list, k)
            k -= self.fibonacci_list[index - 1]
            step += 1
        return step

    def displayTable(self, orders: list) -> list:
        """点菜展示表"""
        tables = [dict() for _ in range(501)]
        menus = set()
        for i in orders:
            table = int(i[1])
            food = i[2]
            tables[table][food] = tables[table].get(food, 0) + 1

            menus.add(food)

        menus_list = list(menus)
        menus_list.sort()

        result = [['Table']]
        for i in menus_list:
            result[0].append(i)

        for i in range(501):
            if len(tables[i]) == 0:
                continue

            table_menu = [str(i)]
            for j in menus_list:
                table_menu.append(str(tables[i].get(j, 0)))

            result.append(table_menu)
        return result

    def minNumberOfFrogs(self, croakOfFrogs: str) -> int:
        """数青蛙"""
        croak_list = []
        frogs_num = 0

        for i in croakOfFrogs:
            if i == 'c':
                croak_list.insert(0, 1)
            elif i == 'r':
                index = bisect.bisect_right(croak_list, 1)
                if index == -1 or croak_list[index - 1] != 1:
                    return -1
                else:
                    croak_list[index - 1] += 1
            elif i == 'o':
                index = bisect.bisect_right(croak_list, 2)
                if index == 0 or croak_list[index - 1] != 2:
                    return -1
                else:
                    croak_list[index - 1] += 1
            elif i == 'a':
                index = bisect.bisect_right(croak_list, 3)
                if index == 0 or croak_list[index - 1] != 3:
                    return -1
                else:
                    croak_list[index - 1] += 1
            elif i == 'k':
                index = bisect.bisect_right(croak_list, 4)
                if index == 0 or croak_list[index - 1] != 4:
                    return -1
                else:
                    frogs_num = max(frogs_num, len(croak_list))
                    del croak_list[index - 1]
            else:
                return -1

            # print(croak_list)

        return frogs_num if not croak_list else -1

    def numOfArrays(self, n: int, m: int, k: int) -> int:
        """生成数组"""

        def add_one() -> bool:
            num[-1] += 1
            carry = 0
            for i in range(len(num) - 1, -1, -1):
                a = num[i] + carry
                if a > m:
                    carry = a // m
                    num[i] = a % m
                else:
                    num[i] = a
                    carry = 0
                    break

            return carry == 0

        if k > m or k > n:
            return 0

        num = [1] * n
        while add_one():
            print(num)

        return 0

    def numberOfSubarrays(self, nums: list, k: int) -> int:
        """
        1248. 统计「优美子数组」
        :see https://leetcode-cn.com/problems/count-number-of-nice-subarrays/
        """
        """ 此方法不够简洁，仅次于暴力。只需要统计奇数所在的下标，根据左右两侧的可扩展距离相乘后相加即可。 """
        # 滑动窗口的左右下标
        left_index = 0
        right_index = -1
        # 奇数的数量
        odd_number_count = 0
        # 子数组的数量
        sub_array_count = 0
        # 统计过的右下标
        checked_right_index = -1

        while left_index < len(nums):
            # 若满足奇数的数量，则统计以当前右下标结尾的子序列数量
            if odd_number_count == k and checked_right_index != right_index:
                checked_right_index = right_index
                # 遍历所有[left_index, right_index]之间满足条件的情况
                for i in range(left_index, right_index + 1):
                    sub_array_count += 1
                    # print(f'{i}, {right_index}: {nums[i:right_index + 1]}')
                    if nums[i] & 1 == 1:
                        break

            if odd_number_count <= k and right_index < len(nums) - 1:
                right_index += 1
                # 判断新加入数字的奇偶性
                if nums[right_index] & 1 == 1:
                    odd_number_count += 1
            else:
                # 判断被删除数字的奇偶性
                if nums[left_index] & 1 == 1:
                    odd_number_count -= 1
                left_index += 1

        return sub_array_count

    def singleNumbers(self, nums: list) -> list:
        """
        面试题56 - I. 数组中数字出现的次数
        :see https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/
        """
        # 通过第一遍历，获得所有数字的异或结果。根据异或结果中某一个‘1’的位置，将数组分成两组。对两组数据分别异或，即可获得两个只出现一次的数字
        all_xor_result = reduce(lambda x, y: x ^ y, nums)

        index = 0
        while all_xor_result & 1 == 0:
            index += 1
            all_xor_result >>= 1

        target = 1 << index

        a_xor_result = 0
        b_xor_result = 0
        for i in nums:
            if i & target == target:
                a_xor_result ^= i
            else:
                b_xor_result ^= i

        return [a_xor_result, b_xor_result]

    def maxScore(self, cardPoints: list, k: int) -> int:
        """
        5393. 可获得的最大点数
        :param cardPoints:
        :param k:
        :return:
        """
        # 找到和为最小的连续子数组
        if k >= len(cardPoints) or k < 1:
            return sum(cardPoints)

        left_card = len(cardPoints) - k
        total_sum = 0

        # cardPoints[i...i+left_card] 的点数和
        card_sum = sum(cardPoints[:left_card])
        # 最小点数和
        min_card_sum = card_sum

        for i in range(len(cardPoints)):
            total_sum += cardPoints[i]
            if 0 < i < len(cardPoints) - left_card + 1:
                card_sum = card_sum - cardPoints[i - 1] + cardPoints[i + left_card - 1]
                min_card_sum = min(min_card_sum, card_sum)
            print(i, card_sum, min_card_sum)

        return total_sum - min_card_sum

    def firstMissingPositive(self, nums: list) -> int:
        """
        41. 缺失的第一个正数
        :see https://leetcode-cn.com/problems/first-missing-positive/
        """
        if not nums:
            return 1

        result_list = [0] * (len(nums) + 1)
        for i in nums:
            if i <= 0 or i >= len(result_list):
                result_list[0] = 1
            else:
                result_list[i] = 1
        print(result_list)

        for i in range(1, len(result_list)):
            if result_list[i] != 1:
                return i
        return len(result_list)

    def singleNumbers(self, nums: list) -> list:
        """
        面试题56 - I. 数组中数字出现的次数
        :see https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/
        """
        # 通过第一遍历，获得所有数字的异或结果。根据异或结果中某一个‘1’的位置，将数组分成两组。对两组数据分别异或，即可获得两个只出现一次的数字
        all_xor_result = reduce(lambda x, y: x ^ y, nums)

        index = 0
        while all_xor_result & 1 == 0:
            index += 1
            all_xor_result >>= 1

        target = 1 << index

        a_xor_result = 0
        b_xor_result = 0
        for i in nums:
            if i & target == target:
                a_xor_result ^= i
            else:
                b_xor_result ^= i

        return [a_xor_result, b_xor_result]

    def candy(self, ratings: list) -> int:
        """
        135. 分发糖果
        :see https://leetcode-cn.com/problems/candy/
        """
        left_candy_list = [0] * len(ratings)
        right_candy_list = [0] * len(ratings)

        for i in range(len(ratings)):
            if i == 0 or ratings[i] <= ratings[i - 1]:
                left_candy_list[i] = 1
            else:
                left_candy_list[i] = left_candy_list[i - 1] + 1

        for i in range(len(ratings) - 1, -1, -1):
            if i == len(ratings) - 1 or ratings[i] <= ratings[i + 1]:
                right_candy_list[i] = 1
            else:
                right_candy_list[i] = right_candy_list[i + 1] + 1

        return sum([max(left_candy_list[i], right_candy_list[i]) for i in range(len(ratings))])

    def kidsWithCandies(self, candies: list, extraCandies: int) -> list:
        """
        5384. 拥有最多糖果的孩子
        :param candies:
        :param extraCandies:
        :return:
        """
        max_candy = max(candies)
        return [candy + extraCandies >= max_candy for candy in candies]

    def numberWays(self, hats: list) -> int:
        """
        5387. 每个人戴不同帽子的方案数
        :param hats:
        :return:
        """

        # 回溯会超时
        def backtrace(index: int, wearing_hats_set: set):
            if index == len(hats):
                self.result_way += 1
                # print(wearing_hats_set)
                return

            for color in hats[index]:
                if color not in wearing_hats_set:
                    wearing_hats_set.add(color)
                    backtrace(index + 1, wearing_hats_set)
                    wearing_hats_set.remove(color)

        self.result_way = 0
        backtrace(0, set())
        return self.result_way % 100000007

    def kLengthApart(self, nums: list, k: int) -> bool:
        """
        5401. 是否所有 1 都至少相隔 k 个元素
        :param nums:
        :param k:
        :return:
        """
        if k < 1:
            return True

        last_one_index = -float('inf')
        for i in range(len(nums)):
            if nums[i] == 0:
                continue
            if i - last_one_index <= k:
                return False
            last_one_index = i

        return True

    def longestSubarray(self, nums: list, limit: int) -> int:
        """
        5402. 绝对差不超过限制的最长连续子数组
        :param nums:
        :param limit:
        :return:
        """
        left = 0
        right = 1
        max_distance = 0

        # 最大数栈和最小数栈
        max_nums_stack = [(left, nums[left])]
        min_nums_stack = [(left, nums[left])]

        while right < len(nums):
            # 将右侧数加入最大数栈和最小数栈
            while max_nums_stack and nums[right] > max_nums_stack[-1][1]:
                max_nums_stack.pop()
            max_nums_stack.append((right, nums[right]))

            while min_nums_stack and nums[right] < min_nums_stack[-1][1]:
                min_nums_stack.pop()
            min_nums_stack.append((right, nums[right]))

            # 检查最大绝对值是否大于 limit
            while max_nums_stack[0][1] - min_nums_stack[0][1] > limit:
                if max_nums_stack[0][0] < min_nums_stack[0][0]:
                    left = max_nums_stack[0][0] + 1
                    max_nums_stack.pop(0)
                elif max_nums_stack[0][0] == min_nums_stack[0][0]:
                    left = max_nums_stack[0][0] + 1
                    max_nums_stack.pop(0)
                    min_nums_stack.pop(0)
                else:
                    left = min_nums_stack[0][0] + 1
                    min_nums_stack.pop(0)

            # 更新右坐标和左坐标的最大距离
            max_distance = max(max_distance, right - left)
            right += 1

        # + 1 的原因是，子数组内元素个数 = 左右下标的差值 + 1，例如 left = 0, right = 0 时，其实含有 1 个元素
        return max_distance + 1

    def buildArray(self, target: list, n: int) -> list:
        """
        5404. 用栈操作构建数组
        :param target:
        :param n:
        :return:
        """
        target_set = set(target)
        result = []
        for i in range(1, target[-1] + 1):
            result.append('Push')
            if i not in target_set:
                result.append('Pop')
        return result

    def countTriplets(self, arr: list) -> int:
        """
        5405. 形成两个异或相等数组的三元组数目
        :param arr:
        :return:
        """
        result = 0
        for i in range(len(arr) - 1):
            a = arr[i]
            for j in range(i + 1, len(arr)):
                a ^= arr[j]
                b = arr[j]
                for k in range(j, len(arr)):
                    b ^= arr[k]
                    if a == b:
                        result += 1

                    # print(i, j, k, a, b)

        return result


if __name__ == "__main__":
    s = Solution()
    print(s.countTriplets([1, 3, 5, 7, 9]))
    # print(a)
