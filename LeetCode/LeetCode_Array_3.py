class Solution:

    def findDiagonalOrder(self, matrix: list) -> list:
        """
        498. 对角线遍历
        :see https://leetcode-cn.com/problems/diagonal-traverse/
        """
        if not matrix or not matrix[0]:
            return []
        m, n = len(matrix), len(matrix[0])
        result = []
        for i in range(m + n - 1):
            if i & 1 == 0:
                for j in range(min(i, m - 1), max(i - n, -1), -1):
                    result.append(matrix[j][i - j])
            else:
                for j in range(max(i - n + 1, 0), min(i + 1, m)):
                    result.append(matrix[j][i - j])
        return result

    def findMin(self, nums: list) -> int:
        """
        154. 寻找旋转排序数组中的最小值 II
        :see https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/
        """
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            print(l, mid, r, nums[l], nums[mid], nums[r])
            if nums[l] <= nums[mid] < nums[r] or nums[l] < nums[mid] <= nums[r]:
                break
            elif nums[mid] == nums[r] or nums[l] == nums[r]:
                r -= 1
            elif nums[l] == nums[mid]:
                l += 1
            elif nums[r] < nums[l] < nums[mid] or nums[r] < nums[l] < nums[mid]:
                l = mid
            elif nums[mid] < nums[r] < nums[l] or nums[mid] < nums[r] < nums[l]:
                r = mid
        return nums[l]

    def numOfSubarrays(self, arr: list) -> int:
        """
        5457. 和为奇数的子数组数目
        :param arr:
        :return:
        """
        # 记录从第一个数字开始的奇数和出现的次数
        count = 0
        # 记录从第一个数字开始的和
        total = 0

        result = 0
        for i in range(len(arr)):
            total += arr[i]
            # 当sum[:i + 1]为奇数时，需要加上arr[:i]的奇数和出现的次数，即i - count + 1
            # 当sum[:i + 1]为偶数时，需要加上arr[:i]的偶数和出现的次数，即count
            if total % 2 == 1:
                result += i - count + 1
                count += 1
            else:
                result += count
            # print(i, total, result, count)

        return result % 1000000007

    def minNumberOperations(self, target: list) -> int:
        """
        5459. 形成目标数组的子数组最少增加次数
        :param target:
        :return:
        """
        target.append(0)
        stack = [0]
        result = 0
        for i in range(len(target)):
            top = stack[-1]
            while stack and target[i] <= stack[-1]:
                stack.pop()
            result += max(top - target[i], 0)
            stack.append(target[i])
            # print(i, result, stack)
        return result


if __name__ == '__main__':
    s = Solution()
    print(s.numOfSubarrays([1, 3, 5]))
