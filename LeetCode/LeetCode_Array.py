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

                nums[start_index], nums[i] = nums[i], nums[start_index]

                permutation(nums, start_index + 1)

                nums[start_index], nums[i] = nums[i], nums[start_index]

        permutation(nums, 0)

        return result


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

    matrix = [1, 1, 2]

    s = Solution()
    print(s.permute(matrix))

    # print(matrix)
