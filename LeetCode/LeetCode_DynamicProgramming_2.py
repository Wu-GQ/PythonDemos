from functools import lru_cache


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

    def winnerSquareGame(self, n: int) -> bool:
        """
        5446. 三次操作后最大值与最小值的最小差
        :see
        """
        # first[i] 表示先手从剩 i 个石子开始取时，能否获胜
        # first[i] = or(second[i + j * j]), i + j * j <= n
        # second[i] 表示后手从剩 i 个石子时开始取，能否获胜(若先手选择取 j * j 个石子)
        # second[i] = first[i + j * j]
        first = [False] * (n + 1)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i < j * j:
                    break
                if not first[i - j * j]:
                    first[i] = True
                    break
        return first[-1]

    def respace(self, dictionary: list, sentence: str) -> int:
        """
        面试题 17.13. 恢复空格
        :see https://leetcode-cn.com/problems/re-space-lcci/
        """
        length = len(sentence)
        dp = [0] * (length + 1)
        words = {i for i in dictionary if sentence.find(i) != -1}

        for i in range(length):
            dp[i + 1] = dp[i] + 1
            for word in words:
                if ((word_len := len(word)) <= i + 1) and sentence[:i + 1].endswith(word):
                    dp[i + 1] = min(dp[i + 1], dp[i + 1 - word_len])
        # print(dp)
        return dp[-1]

    def numTrees(self, n: int) -> int:
        """
        96. 不同的二叉搜索树
        :see https://leetcode-cn.com/problems/unique-binary-search-trees/
        """
        # dp[n] 表示 n 个节点时可以组成的二叉搜索树
        # 将 i 作为根节点，左子树有 i - 1 个节点，右子树有 n - i 个节点
        # dp[i] = sum(dp[j - 1] * dp[i - j], 1 <= j <= i)
        dp = [1, 1]
        for i in range(2, n + 1):
            total = 0
            for j in range(1, i + 1):
                total += dp[j - 1] * dp[i - j]
            dp.append(total)
        return dp[-1]

    def divisorGame(self, N: int) -> bool:
        """
        1025. 除数博弈
        :see https://leetcode-cn.com/problems/divisor-game/
        """
        '''
        假设当前数字为n
        A先手：
        假设A选了i，i满足以上条件
        A[n] = or(B[n - i]), 0 < i < N 且 N % i == 0
        因为A选择最优，所以A需要遍历所有满足条件的i，只要一个B[n - i]为True，A[n]即为True
        B后手:
        B[n] = A[n - i]
        
        由于A赢则B输，B赢则A输，所以只需要一个dp即可同时表示A和B的状态
        dp[n] = or(not dp[n - i]), 0 < i < N 且 N % i == 0
        '''
        dp = [False] * (N + 1)
        for n in range(2, N + 1):
            for i in range(1, n):
                if n % i == 0 and not dp[n - i]:
                    dp[n] = True
                    break
        # print(dp)
        return dp[N]
        # return not N & 1

    def longestIncreasingPath(self, matrix: list) -> int:
        """
        329. 矩阵中的最长递增路径
        :see https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/
        """

        def max_path(x: int, y: int) -> int:
            if dp[x][y] > 1:
                return dp[x][y]
            result = 1
            for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= i < len(matrix) and 0 <= j < len(matrix[i]) and matrix[i][j] < matrix[x][y]:
                    result = max(result, max_path(i, j) + 1)
            dp[x][y] = result
            return result

        if not matrix or not matrix[0]:
            return 0
        dp = [[1] * len(matrix[i]) for i in range(len(matrix))]
        ans = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                ans = max(ans, max_path(i, j))
        return ans

    def isSubsequence(self, s: str, t: str) -> bool:
        """
        392. 判断子序列
        :see https://leetcode-cn.com/problems/is-subsequence/
        """
        s = ' ' + s
        t = ' ' + t
        dp = [[False] * len(t) for _ in s]
        for i in range(len(t)):
            dp[0][i] = True
        for i in range(1, len(s)):
            for j in range(1, len(t)):
                dp[i][j] = dp[i][j - 1]
                if not dp[i][j] and s[i] == t[j]:
                    dp[i][j] = dp[i - 1][j - 1]
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

    def winnerSquareGame(self, n: int) -> bool:
        """
        5446. 三次操作后最大值与最小值的最小差
        :see
        """
        # first[i] 表示先手从剩 i 个石子开始取时，能否获胜
        # first[i] = or(second[i + j * j]), i + j * j <= n
        # second[i] 表示后手从剩 i 个石子时开始取，能否获胜(若先手选择取 j * j 个石子)
        # second[i] = first[i + j * j]
        first = [False] * (n + 1)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i < j * j:
                    break
                if not first[i - j * j]:
                    first[i] = True
                    break
        return first[-1]

    def respace(self, dictionary: list, sentence: str) -> int:
        """
        面试题 17.13. 恢复空格
        :see https://leetcode-cn.com/problems/re-space-lcci/
        """
        length = len(sentence)
        dp = [0] * (length + 1)
        words = {i for i in dictionary if sentence.find(i) != -1}

        for i in range(length):
            dp[i + 1] = dp[i] + 1
            for word in words:
                if ((word_len := len(word)) <= i + 1) and sentence[:i + 1].endswith(word):
                    dp[i + 1] = min(dp[i + 1], dp[i + 1 - word_len])
        # print(dp)
        return dp[-1]

    def numTrees(self, n: int) -> int:
        """
        96. 不同的二叉搜索树
        :see https://leetcode-cn.com/problems/unique-binary-search-trees/
        """
        # dp[n] 表示 n 个节点时可以组成的二叉搜索树
        # 将 i 作为根节点，左子树有 i - 1 个节点，右子树有 n - i 个节点
        # dp[i] = sum(dp[j - 1] * dp[i - j], 1 <= j <= i)
        dp = [1, 1]
        for i in range(2, n + 1):
            total = 0
            for j in range(1, i + 1):
                total += dp[j - 1] * dp[i - j]
            dp.append(total)
        return dp[-1]

    def divisorGame(self, N: int) -> bool:
        """
        1025. 除数博弈
        :see https://leetcode-cn.com/problems/divisor-game/
        """
        '''
        假设当前数字为n
        A先手：
        假设A选了i，i满足以上条件
        A[n] = or(B[n - i]), 0 < i < N 且 N % i == 0
        因为A选择最优，所以A需要遍历所有满足条件的i，只要一个B[n - i]为True，A[n]即为True
        B后手:
        B[n] = A[n - i]
        
        由于A赢则B输，B赢则A输，所以只需要一个dp即可同时表示A和B的状态
        dp[n] = or(not dp[n - i]), 0 < i < N 且 N % i == 0
        '''
        dp = [False] * (N + 1)
        for n in range(2, N + 1):
            for i in range(1, n):
                if n % i == 0 and not dp[n - i]:
                    dp[n] = True
                    break
        # print(dp)
        return dp[N]
        # return not N & 1

    def longestIncreasingPath(self, matrix: list) -> int:
        """
        329. 矩阵中的最长递增路径
        :see https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/
        """

        def max_path(x: int, y: int) -> int:
            if dp[x][y] > 1:
                return dp[x][y]
            result = 1
            for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= i < len(matrix) and 0 <= j < len(matrix[i]) and matrix[i][j] < matrix[x][y]:
                    result = max(result, max_path(i, j) + 1)
            dp[x][y] = result
            return result

        if not matrix or not matrix[0]:
            return 0
        dp = [[1] * len(matrix[i]) for i in range(len(matrix))]
        ans = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                ans = max(ans, max_path(i, j))
        return ans

    def isSubsequence(self, s: str, t: str) -> bool:
        """
        392. 判断子序列
        :see https://leetcode-cn.com/problems/is-subsequence/
        """
        s = ' ' + s
        t = ' ' + t
        dp = [[False] * len(t) for _ in s]
        for i in range(len(t)):
            dp[0][i] = True
        for i in range(1, len(s)):
            for j in range(1, len(t)):
                dp[i][j] = dp[i][j - 1]
                if not dp[i][j] and s[i] == t[j]:
                    dp[i][j] = dp[i - 1][j - 1]
        return dp[-1][-1]

    def minimalSteps(self, maze: list) -> int:
        """
        LCP 13. 寻宝
        :see https://leetcode-cn.com/problems/xun-bao/
        """

        def bfs(point: tuple, target_list: list) -> dict:
            """ 始发点到目标列表中所有节点的最短距离，若不可达，则为float('inf') """
            target_set = set(target_list)
            if point in target_set:
                target_set.remove(point)
            queue = [point]
            checked = set()
            distance_dict = {point: 0}
            step = 0
            while queue and target_set:
                length = len(queue)
                step += 1
                while length > 0 and target_set:
                    length -= 1
                    x, y = queue.pop(0)
                    checked.add((x, y))
                    for i, j in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if 0 <= i < len(maze) and 0 <= j < len(maze[i]) and maze[i][j] != '#' and (i, j) not in checked:
                            next_point = (i, j)
                            queue.append(next_point)
                            if next_point in target_set:
                                target_set.remove(next_point)
                                distance_dict[next_point] = step
            for i in target_set:
                distance_dict[i] = float('inf')
            return distance_dict

        # 1. 找到所有的关键点：起点、终点、石堆、机关
        start, end = (), ()
        stones, buttons = [], []
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if maze[i][j] == 'M':
                    buttons.append((i, j))
                elif maze[i][j] == 'O':
                    stones.append((i, j))
                elif maze[i][j] == 'S':
                    start = (i, j)
                elif maze[i][j] == 'T':
                    end = (i, j)

        # 2. 起点到其它关键点的最短距离
        start_all_dict = bfs(start, stones + buttons + [end])

        # 3. 如果不存在机关，可以直达终点
        if not buttons:
            return start_all_dict[end] if end in start_all_dict and start_all_dict[end] != float('inf') else -1

        # 4. 任意机关到石堆 + 终点的最短距离
        buttons_stones_dict = {}
        for button in buttons:
            buttons_stones_dict[button] = bfs(button, stones + [end])

        # 5. 任意机关到起点、其它机关、终点的最短距离
        buttons_all_dict = {}
        for button in buttons:
            # 5.1. 起点 -> 石堆 -> 该机关
            button_start_distance = float('inf')
            for stone in stones:
                button_start_distance = min(button_start_distance, start_all_dict[stone] + buttons_stones_dict[button][stone])
            if button in buttons_all_dict:
                buttons_all_dict[button][start] = button_start_distance
            else:
                buttons_all_dict[button] = {start: button_start_distance}

            # 5.2. 该机关 -> 石堆 -> 另一机关
            for other in buttons:
                if other in buttons_all_dict and button in buttons_all_dict[other]:
                    continue
                button_other_distance = float('inf')
                for stone in stones:
                    button_other_distance = min(button_other_distance, buttons_stones_dict[button][stone] + buttons_stones_dict[other][stone])
                buttons_all_dict[button][other] = button_other_distance
                if other not in buttons_all_dict:
                    buttons_all_dict[other] = {button: button_other_distance}
                else:
                    buttons_all_dict[other][button] = button_other_distance

            # 5.3. 该机关 -> 终点
            buttons_all_dict[button][end] = buttons_stones_dict[button][end]

        # 6. 确认每个机关皆可以到达起点和终点
        for button in buttons:
            if buttons_all_dict[button][start] == float('inf') or buttons_all_dict[button][end] == float('inf'):
                return -1

        # 7. 状态压缩，i-将已完成的任务置为1，未完成的任务置为0，j-当前在哪个机关
        dp = [[float('inf')] * len(buttons) for _ in range(1 << len(buttons))]
        for i in range(len(buttons)):
            dp[1 << i][i] = buttons_all_dict[buttons[i]][start]

        # 8. 动态规划，dp[i | 1 << k][k] = min(dp[i][j] + buttons_all_dict[buttons[j]][buttons[k]])，k-下一个机关
        for i in range(1, 1 << len(buttons)):
            for j in range(len(buttons)):
                # 确认 j 机关在这个范围内
                if i & (1 << j):
                    for k in range(len(buttons)):
                        # 确认 k 机关不在这个范围内
                        if not (i & (1 << k)):
                            next = i | (1 << k)
                            dp[next][k] = min(dp[next][k], dp[i][j] + buttons_all_dict[buttons[j]][buttons[k]])

        # 9. 走完所有的机关后，加上最后一个机关到终点的距离
        result = float('inf')
        all_buttons_status = (1 << len(buttons)) - 1
        for i in range(len(buttons)):
            result = min(result, dp[all_buttons_status][i] + buttons_all_dict[buttons[i]][end])
        return result

    def minCost(self, n: int, cuts: list) -> int:
        """
        1547. 切棍子的最小成本
        :see https://leetcode-cn.com/problems/minimum-cost-to-cut-a-stick/
        """
        # 把切棍子问题，转换成合并棍子的问题，然后用区间动归
        # dp[i][j] = (x, y)，i和j表示从第i段到第j段（包括第j段），x表示合并这几段的最小成本，y表示这几段的总长度
        # dp[i][j] = (min(dp[i][k][0] + dp[i][k][1] + dp[k+1][j][0] + dp[k+1][j][1]), dp[i][k][1] + dp[k+1][j][1])
        cuts.sort()
        nums = []
        old = 0
        for i in cuts:
            nums.append(i - old)
            old = i
        nums.append(n - old)
        print(nums)

        dp = [[(-1, 0) for _ in nums] for _ in nums]
        for i in range(len(nums)):
            dp[i][i] = (0, nums[i])

        for t in range(1, len(nums)):
            for i in range(len(nums) - t):
                j = t + i
                for k in range(i, j):
                    temp = dp[i][k][0] + dp[i][k][1] + dp[k + 1][j][0] + dp[k + 1][j][1]
                    if dp[i][j][0] < 0 or temp < dp[i][j][0]:
                        dp[i][j] = (temp, dp[i][k][1] + dp[k + 1][j][1])

        return dp[0][-1][0]

    def removeBoxes(self, boxes: list) -> int:
        """
        546. 移除盒子
        :see https://leetcode-cn.com/problems/remove-boxes/
        """

        # PS：很难，看不懂就再去看次题解吧，第一次先按照代码抄一遍
        # 对于每个盒子，有两种选择
        # 1. 直接把相邻相同的全部移除
        # 2. 把不相邻的盒子合并后一起移除，在这之前，先移除破坏相邻的所有盒子
        #
        # dfs(i, j, k) 表示
        # 当前处理的区间 [l, r], 目的是要移除区间最右边的盒子也就是box[r]
        # k 表示该区间外，即在区间右侧，有连续k个和box[r]相同颜色的盒子可以一起删除
        # f[i][j][k] 表示该操作的最大收益

        def dfs(l: int, r: int, k: int) -> int:
            # [l, r]区间内, 目前能够删除k个与 b[r]相同颜色盒子的的最大收益
            if l > r:
                return 0

            # k 说明 b[r] 后面有 k 个和 b[r] 相同的值
            # 也就是能连续删除 k + 1 个 b[r]
            while l < r and boxes[r] == boxes[r - 1]:
                r -= 1
                k += 1

            # 若已经有答案，直接返回
            if dp[l][r][k] > 0:
                return dp[l][r][k]

            # 决策1：直接删除
            # 删除 k + 1 个，递归处理前面的区间[l, r - 1]
            dp[l][r][k] = dfs(l, r - 1, 0) + (k + 1) ** 2

            # 决策2：在 [l, r-1] 区间找到和 b[r] 相同颜色的盒子，合并起来一起删除
            for i in range(l, r):
                if boxes[i] == boxes[r]:
                    # 若找到了相同颜色的盒子，将 [l, r - 1] 分成 [l, i] 和 [i + 1, r - 1]
                    dp[l][r][k] = max(dp[l][r][k], dfs(l, i, k + 1) + dfs(i + 1, r - 1, 0))

            return dp[l][r][k]

        n = len(boxes)
        dp = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        return dfs(0, n - 1, 0)

    def stoneGameV(self, stoneValue: list) -> int:
        """
        5498. 石子游戏 V
        :see https://leetcode-cn.com/problems/stone-game-v/
        """
        ''' 动态规划解法，超时
        # 前缀和，i到j的前缀和为p_sum_list[j + 1]-p_sum_list[i]
        p_sum_list = [0]
        for i in stoneValue:
            p_sum_list.append(p_sum_list[-1] + i)

        # dp[i][j]表示从i开始到j为止(包括j),可获得的最大分数
        # dp[i][j] = max(i到k的和较小 ? dp[i][k] + 线段和 : dp[k][j] + 线段和)
        dp = [[0] * len(stoneValue) for _ in stoneValue]
        for i in range(len(stoneValue) - 1):
            dp[i][i + 1] = min(stoneValue[i], stoneValue[i + 1])

        for t in range(2, len(stoneValue)):
            for i in range(len(stoneValue) - t):
                j = t + i

                for k in range(i, j + 1):
                    left = p_sum_list[k] - p_sum_list[i]
                    right = p_sum_list[j + 1] - p_sum_list[k]
                    if left < right:
                        dp[i][j] = max(dp[i][j], dp[i][k - 1] + left)
                    elif left > right:
                        dp[i][j] = max(dp[i][j], dp[k][j] + right)
                    else:
                        dp[i][j] = max(dp[i][j], max(dp[i][k - 1], dp[k][j]) + left)

        return 0
        '''
        # 记忆化递归
        # 前缀和，i到j的前缀和为p_sum_list[j + 1]-p_sum_list[i]
        p_sum_list = [0]
        for i in stoneValue:
            p_sum_list.append(p_sum_list[-1] + i)

        @lru_cache(None)
        def dfs(start: int, end: int) -> int:
            """ [start, end]区间内的最大得分 """
            if start >= end:
                return 0
            elif start + 1 == end:
                return min(stoneValue[start], stoneValue[end])

            value = 0
            for k in range(start, end + 1):
                left = p_sum_list[k] - p_sum_list[start]
                right = p_sum_list[end + 1] - p_sum_list[k]
                if left < right:
                    value = max(value, dfs(start, k - 1) + left)
                elif left > right:
                    value = max(value, dfs(k, end) + right)
                else:
                    value = max(value, max(dfs(start, k - 1), dfs(k, end)) + left)
            return value

        return dfs(0, len(stoneValue) - 1)

    def minSteps(self, n: int) -> int:
        """
        650. 只有两个键的键盘
        :see https://leetcode-cn.com/problems/2-keys-keyboard/
        """
        # dp[i][j] 表示打印i个字符，粘贴长度为j个字符时的最小步数
        dp = [[float('inf')] * (n + 1) for _ in range(n + 1)]
        dp[1][0] = 0
        dp[1][1] = 1

        for i in range(2, n + 1):
            # min_time 表示打印i个字符时的最小步数
            min_time = float('inf')

            for j in range(1, i):
                dp[i][j] = dp[i - j][j] + 1
                min_time = min(min_time, dp[i][j])

            dp[i][i] = min_time + 1

        return min(dp[n])

    def findRotateSteps(self, ring: str, key: str) -> int:
        """
        514. 自由之路
        :see https://leetcode-cn.com/problems/freedom-trail/
        """
        # 存储每个字母所在的位置
        char_index_dict = {}
        for i in range(len(ring)):
            if ring[i] in char_index_dict:
                char_index_dict[ring[i]].append(i)
            else:
                char_index_dict[ring[i]] = [i]

        # 存储当前所在的位置和已经使用步数
        dp = {0: 0}
        # 上一个字母
        last = ''

        for ch in key:
            # 当字母相同时，不需要移动
            if ch == last:
                for i in dp:
                    dp[i] += 1
                continue

            last = ch

            # 从 dp 所在的位置 到 ch字母所有的位置
            next_dp = {}

            for end_index in char_index_dict[ch]:
                min_step = float('inf')

                for start_index in dp:
                    distance = abs(end_index - start_index)
                    min_step = min(min_step, dp[start_index] + min(distance, len(ring) - distance))

                    # 相邻的，肯定是最小的步数
                    if min_step == 1:
                        break

                # 加上打印字母的一步
                next_dp[end_index] = min_step + 1

            # 更新
            dp = next_dp

            # print(dp)

        return min(dp.values())


if __name__ == '__main__':
    s = Solution()
    # print(s.stoneGameV([1, 1, 2]))
    print(s.findRotateSteps("xrrakuulnczywjs", "jrlucwzakzussrlckyjjsuwkuarnaluxnyzcnrxxwruyr"))
