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


if __name__ == '__main__':
    s = Solution()
    print(s.findDiagonalOrder([]))
