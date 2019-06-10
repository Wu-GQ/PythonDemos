class Solution:
    _count_list = [0]

    def num_squares(self, n: int) -> int:
        """
        完全平方数
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/272/dynamic-programming/1178/
        """
        length = len(self._count_list)

        if length > n:
            return self._count_list[n]

        for i in range(length, n + 1):
            j = 1
            self._count_list.append(i)
            while i - j * j >= 0:
                self._count_list[i] = min(self._count_list[i], self._count_list[i - j * j] + 1)
                j += 1

        return self._count_list[n]

    def lengthOfLIS(self, nums: list) -> int:
        """
        最长上升子序列
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/272/dynamic-programming/1179/
        """
        # count_list = [1] * len(nums)
        #
        # max_length = 0
        #
        # for i in range(len(nums)):
        #     for j in range(i):
        #         if nums[j] < nums[i]:
        #             count_list[i] = max(count_list[i], count_list[j] + 1)
        #
        #     if count_list[i] > max_length:
        #         max_length = count_list[i]
        #
        # return max_length
        size = len(nums)
        if size < 2:
            return size

        tail = []
        for num in nums:
            # 找到大于等于 num 的第 1 个数
            l = 0
            r = len(tail)
            while l < r:
                mid = l + (r - l) // 2
                if tail[mid] < num:
                    l = mid + 1
                else:
                    r = mid
            if l == len(tail):
                tail.append(num)
            else:
                tail[l] = num

            print(tail)
        return len(tail)


if __name__ == '__main__':
    so = Solution()
    nums_list = [10, 9, 2, 5, 3, 7, 101, 18, 1, 2, 3, 4, 5]
    print(so.lengthOfLIS(nums_list))
