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
        for i in matrix:
            for j in matrix[i]:
                j = 0

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


if __name__ == "__main__":
    # array2 = [9, 0, 9, 1, 9]
    # print(twoSum(array2, 9))

    minStack = MinStack()
    minStack.push(-2)
    minStack.push(0)
    minStack.push(-3)
    print(minStack.getMin())

    print(minStack.stack, minStack.min_stack)

    minStack.pop()
    print(minStack.top())

    minStack.getMin()
    print(minStack.stack, minStack.min_stack)
