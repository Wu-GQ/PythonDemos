class MinStack:
    """
    最小栈
    :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/24/design/59/
    """

    def __init__(self):
        """
        initialize your data structure here.
        """
        # 用来存正常的数据
        self.stack = []
        # 用来存最小的数据
        self.min_stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        # 比较前一存入的数据,如果新加入的数据比较小，则在另一栈中存入新的数据
        self.min_stack.append(self.min_stack[-1] if len(self.min_stack) > 0 and self.min_stack[-1] < x else x)

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


class Solution:
    def move_zeroes(self, nums: list) -> None:
        """
        移动零
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/264/array/1130/
        """

        def update_first_zero_index(self, nums: list, start_index: int) -> int:
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
        括号生成
        :see https://leetcode-cn.com/problems/generate-parentheses/
        """
        if n < 1:
            return []

        result_list = []

        def backtrace(string: str, left_count: int, right_count: int):
            if len(string) == 2 * n:
                result_list.append(string)
                return

            if 0 < left_count < n and left_count > right_count:
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
        跳跃游戏
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
        length = len(nums)
        if length < 1:
            return False
        elif length == 1:
            return True

        max_index = nums[0]
        i = 0
        while i < length - 1 and i <= max_index:
            max_index = max(max_index, i + nums[i])
            if max_index >= length - 1:
                return True
            i += 1

        return False

    def jump(self, nums: list) -> int:
        """
        跳跃游戏 II
        :see https://leetcode-cn.com/problems/jump-game-ii/
        """
        # 参考广度优先遍历（超时）
        length = len(nums)
        if length < 2:
            return 0

        min_step_list = [-1] * length

        def bfs_jump(start_index: int) -> int:
            if start_index >= length - 1:
                return 0

            # 避免重复计算
            if min_step_list[start_index] >= 0:
                return min_step_list[start_index]

            # 递归
            min_jump_step = float('inf')
            for i in range(start_index + nums[start_index], start_index, -1):
                if i >= length - 1:
                    min_jump_step = 1
                    break
                else:
                    min_jump_step = min(min_jump_step, bfs_jump(i) + 1)

            # 保存最小步数
            min_step_list[start_index] = min_jump_step

            return min_jump_step

        return bfs_jump(0)

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


if __name__ == "__main__":
    s = Solution()
    num = [1, 1, 2, 2, 2, 3]
    s.wiggleSort(num)
    print(num)
