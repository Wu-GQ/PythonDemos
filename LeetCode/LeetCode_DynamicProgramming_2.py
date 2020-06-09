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

        return int(dp[0][-1])

    def canPartition(self, nums: list) -> bool:
        """
        416. 分割等和子集
        :see https://leetcode-cn.com/problems/partition-equal-subset-sum/
        """
        sum_set = set()
        for i in nums:
            new_sum = set()
            for j in sum_set:
                new_sum.add(i + j)
            sum_set.update(new_sum)

            sum_set.add(i)

        return sum(nums) / 2 in sum_set

    def findTargetSumWays(self, nums: list, S: int) -> int:
        """
        494. 目标和
        :see https://leetcode-cn.com/problems/target-sum/
        """
        dp = {0: 1}

        for i in nums:
            new_dp = {}
            for j in dp:
                new_dp[j + i] = new_dp.get(j + i, 0) + dp[j]
                new_dp[j - i] = new_dp.get(j - i, 0) + dp[j]
            dp = new_dp
            # print(dp)

        return dp.get(S, 0)

    def changeII(self, amount: int, coins: list) -> int:
        """
        518. 零钱兑换 II
        :see https://leetcode-cn.com/problems/coin-change-2/
        """
        dp = [0] * (amount + 1)
        dp[0] = 1

        for coin in coins:
            for i in range(amount + 1):
                dp[i] += dp[i - coin] if i - coin >= 0 else 0
            # print(dp)

        return dp[amount]

    def findMaxForm(self, strs: list, m: int, n: int) -> int:
        """
        474. 一和零
        :see https://leetcode-cn.com/problems/ones-and-zeroes/
        """
        # 动态规划结果存储，key 表示剩余的 0 和 1 的数量，value 表示 key 对应的数量下最大的字符串数量
        dp = {(m, n): 0}
        # 最大字符串的数量
        max_count = 0

        for string in strs:
            # 计算字符串中 0 和 1 的数量
            zero = string.count('0')
            one = len(string) - zero

            new_dp = {}
            for i, j in dp:
                if zero <= i and one <= j:
                    # 动态规划方程: f(i - zero, j - one) = max(f(i - zero, j - one), f(i, j) + 1)
                    new_dp[i - zero, j - one] = max(dp.get((i - zero, j - one), 0), dp[i, j] + 1)
                    # 更新最大值
                    max_count = max(max_count, new_dp[i - zero, j - one])
            # 合并
            dp.update(new_dp)

        return max_count

    def new21Game(self, N: int, K: int, W: int) -> float:
        """
        837. 新21点
        :see https://leetcode-cn.com/problems/new-21-game/
        """
        # 自顶向下的动态规划，参考题解https://leetcode-cn.com/problems/new-21-game/solution/huan-you-bi-zhe-geng-jian-dan-de-ti-jie-ma-tian-ge/
        # dp[i] 表示爱丽丝在i分时的胜率，即不少于N分获胜，若大于N分则失败
        # 当 K <= i < K + W 时，胜率为1
        # 当 0 <= i < K 时，胜率为(1 / W) * sum(dp[i + 1:i + w])
        if N > K + W:
            return 0

        dp = [1] * (N + 1) + [0] * (K + W - N)
        dp_part_sum = sum(dp[K:K + W])

        for i in range(K - 1, -1, -1):
            dp[i] = dp_part_sum / W
            dp_part_sum += dp[i] - dp[i + W]
        # print(dp)
        return dp[0]

    def translateNum(self, num: int) -> int:
        """
        面试题46. 把数字翻译成字符串
        :see https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/
        """
        string = str(num)
        dp = [0] * len(string)

        for i in range(len(string)):
            if i == 0:
                dp[i] = 1
            elif i == 1:
                dp[i] = 1 + (1 if int(string[:2]) < 26 else 0)
            else:
                dp[i] = dp[i - 1] + (dp[i - 2] if 9 < int(string[i - 1:i + 1]) < 26 else 0)
        return dp[-1]


if __name__ == '__main__':
    s = Solution()
    print(s.translateNum(100))
