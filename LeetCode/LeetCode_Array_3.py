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

    def closestToTarget(self, arr: list, target: int) -> int:
        """
        5466. 最多的不重叠子字符串
        :param arr:
        :param target:
        :return:
        """
        result = abs(target - arr[0])
        for i in range(len(arr)):
            if i > 0 and arr[i] == arr[i - 1]:
                continue
            t = arr[i]
            for j in range(i, len(arr)):
                if j > 0 and arr[j] == arr[j - 1]:
                    continue
                t &= arr[j]
                print(i, j, t)
                if t == target:
                    return 0
                if abs(t - target) < result:
                    result = abs(t - target)
                if t == 0:
                    continue
        return result


if __name__ == '__main__':
    s = Solution()
    print(s.closestToTarget([70, 15, 21, 96], 4))
