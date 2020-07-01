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

    def splitArray(self, nums: list, m: int) -> int:
        """
        410. 分割数组的最大值
        :see https://leetcode-cn.com/problems/split-array-largest-sum/
        """
        # 二分查找解法参考 LeetCode_Search.py 的 splitArray
        length = len(nums)
        if length <= m:
            return max(nums)

        # dp[i][j] 表示 nums[:i + 1] 分成 j + 1 份时的最小和
        # dp[i][j] = min(dp[i][j], max(dp[k][j - 1], sum(nums[k + 1:i + 1]))), 0 <= k < i
        # 状态转移方程中，dp[k][j - 1] 表示前 k 个数字(包括第 k 个数字)已分成 j 份时的最小和，
        # 由于新加了一个数字 nums[i]，数组就分成了 "已经被分成 j 份的 num[:k + 1]" 和 "第 j + 1 份的 num[k + 1:i + 1]"
        # 这时候的最小和就是前半部分和后半部分数字和的最大值，即为 max(dp[k][j - 1], sum(nums[k + 1:i + 1])), 0<= k < i
        dp = [[float('inf')] * m for _ in range(length)]

        for i in range(length):
            # 不用考虑1个数字分成2份，2个数字分成3份的情况
            for j in range(min(i + 1, m)):
                if i == 0:
                    # 只有1个数字时，无论分几份，最小和都是这个数
                    dp[0][j] = nums[0]
                elif j == 0:
                    # 只分成1份时，无论有几个数，最小和都是这些数的和
                    dp[i][0] = nums[i] + dp[i - 1][0]
                else:
                    # 因为不用考虑1个数字分成2份，2个数字分成3份的情况，所以 k 从 j - 1 开始遍历即可
                    for k in range(j - 1, i + 1):
                        # 因为 dp[i][0] 就是 sum(num[:i + 1])，
                        # 所以状态转移方程中的 sum(nums[k + 1:i + 1])) 可以改为 dp[i][0] - dp[k][0]
                        dp[i][j] = min(dp[i][j], max(dp[k][j - 1], dp[i][0] - dp[k][0]))

        return dp[-1][-1]

    def minDifficulty(self, jobDifficulty: list, d: int) -> int:
        """
        1335. 工作计划的最低难度
        :see https://leetcode-cn.com/problems/minimum-difficulty-of-a-job-schedule/
        """
        # dp[i][j] 代表前 i 项任务(包括第 i 项)在 j + 1 天完成时的最小难度和
        # dp[i][j] = min(dp[i][j], dp[k][j - 1] + max(jD[k + 1:i + 1])), 0 <= k < i
        # 状态转移方程中，dp[k][j - 1] 表示前 k 个任务(包括第 k 个任务)已分成 j 天时的最小难度之和，
        # 由于新加了一个任务 jD[i]，数组就分成了 "已经 j 天内完成的 jD[:k + 1]" 和 "第 j + 1 天完成的 jD[k + 1:i + 1]"
        # 这时候的最小难度就是前半部分和后半部分的难度之和，即为 dp[k][j - 1] + max(jD[k + 1:i + 1]), 0 <= k < i
        # 时间复杂度为 O(len * d * len * len)
        length = len(jobDifficulty)
        if length == d:
            return sum(jobDifficulty)
        elif length < d:
            return -1

        # 预处理区间最大值，算法整体时间复杂度为 O(len * d * len + len * len * len)
        max_value_dict = {}
        for i in range(length):
            for j in range(i + 1, length + 1):
                max_value_dict[(i, j)] = max(jobDifficulty[i:j])

        # 0 <= jobDifficulty[i] <= 1000, 1 <= d <= 10, 所以最大值为 10000
        dp = [[10000] * d for _ in range(length)]

        for i in range(length):
            # 当 j < i + 1 时，即 j + 1 天分配最多 i 项任务，这是不可行的。因此从 i + 1 开始递推
            for j in range(min(i + 1, d)):
                if i == 0:
                    # 只有一项任务，无论几天，最小难度都是这个任务
                    dp[0][j] = jobDifficulty[0]
                elif j == 0:
                    # 只有一天，无论多少任务，最小难度都是最大难度的任务
                    dp[i][0] = max(dp[i - 1][0], jobDifficulty[i])
                else:
                    # k 表示第 k 个任务，每天至少有一项任务，所以从 j - 1 开始递推
                    for k in range(j - 1, i):
                        # dp[i][j] = min(dp[i][j], dp[k][j - 1] + max(jobDifficulty[k + 1:i + 1]))
                        dp[i][j] = min(dp[i][j], dp[k][j - 1] + max_value_dict[(k + 1, i + 1)])

        return dp[-1][-1]

    def getKthMagicNumber(self, k: int) -> int:
        """
        面试题 17.09. 第 k 个数
        :see https://leetcode-cn.com/problems/get-kth-magic-number-lcci/
        """
        # 考虑3个数列
        # i:  0, 1, 2, 3, ...
        # 3i: 1, 3, 9, 15, ...
        # 5i: 1, 5, 10, 15, ...
        # 7i: 1, 7, 14, 21, ...
        # 如何合并这三个数列，并得到第 k 个小的数

        three, five, seven = 0, 0, 0
        dp = [1]
        for i in range(k - 1):
            num = min((dp[three] * 3, dp[five] * 5, dp[seven] * 7))
            if num % 3 == 0:
                three += 1
            if num % 5 == 0:
                five += 1
            if num % 7 == 0:
                seven += 1
            dp.append(num)
        # print(dp)
        return dp[-1]

    def nthUglyNumber(self, n: int) -> int:
        """
        264. 丑数 II
        :see https://leetcode-cn.com/problems/ugly-number-ii/
        """
        # 与上面那题的解法相同
        two, three, five = 0, 0, 0
        dp = [1]
        for _ in range(n - 1):
            num = min((dp[two] * 2, dp[three] * 3, dp[five] * 5))
            if num % 2 == 0:
                two += 1
            if num % 3 == 0:
                three += 1
            if num % 5 == 0:
                five += 1
            dp.append(num)
        # print(dp)
        return dp[-1]

    def nthSuperUglyNumber(self, n: int, primes: list) -> int:
        """
        313. 超级丑数
        :see https://leetcode-cn.com/problems/super-ugly-number/
        """
        # 与上面那题的解法相同
        primes_length = len(primes)
        primes_times = [0] * primes_length
        dp = [1]

        for _ in range(n - 1):
            num = min([dp[primes_times[i]] * primes[i] for i in range(primes_length)])

            for i in range(primes_length):
                if num % primes[i] == 0:
                    primes_times[i] += 1

            dp.append(num)

        return dp[-1]

    def findLength(self, A: list, B: list) -> int:
        """
        718. 最长重复子数组
        :see https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/
        """
        dp = [[0] * len(B) for _ in A]
        result = 0
        for i in range(len(A)):
            for j in range(len(B)):
                if A[i] == B[j]:
                    dp[i][j] = dp[i - 1][j - 1] + 1 if i > 0 and j > 0 else 1
                    result = max(result, dp[i][j])
        # for i in dp:
        #     print(i)
        return result


if __name__ == '__main__':
    s = Solution()
    print(s.findLength([0, 0, 0, 0, 1], [1, 0, 0, 0, 0]))
