class Solution:

    def minScoreTriangulation(self, A: list) -> int:
        """
        1039. 多边形三角剖分的最低得分
        :see https://leetcode-cn.com/problems/minimum-score-triangulation-of-polygon/
        """
        # dp[i][j] 表示在 [i...j] 的范围内，可以得到的最低得分。
        # dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j] + A[i] * A[k] * A[j]), i <= k <= j
        # 以 k 点作为三角形的一个顶点，则新构成的三角形为 ikj，只要加上左侧的最小值和右侧的最小值，即 k 点作为顶点的最小值

        dp = [[float('inf')] * len(A) for _ in A]

        # 初始化 - 两个点以下不能形成三角形
        for i in range(len(A) - 1):
            dp[i][i] = 0
            dp[i][i + 1] = 0
        dp[len(A) - 1][len(A) - 1] = 0

        for d in range(2, len(A)):
            for i in range(len(A) - d):
                j = d + i
                for k in range(i + 1, j):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j] + A[i] * A[k] * A[j])

        return dp[0][-1]


if __name__ == '__main__':
    s = Solution()
    print(s.minScoreTriangulation([3, 7, 4, 5]))
