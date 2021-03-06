from typing import List


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

    def searchRange(self, nums: list, target: int) -> list:
        """
        34. 在排序数组中查找元素的第一个和最后一个位置
        :see https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/
        """
        # 仿bisect_left找左边界
        left1, right1 = 0, len(nums)
        while left1 < right1:
            mid = left1 + (right1 - left1) // 2
            if nums[mid] < target:
                left1 = mid + 1
            else:
                right1 = mid

        # 仿bisect_right找右边界
        left2, right2 = 0, len(nums)
        while left2 < right2:
            mid = left2 + (right2 - left2) // 2
            if nums[mid] <= target:
                left2 = mid + 1
            else:
                right2 = mid

        return [left1, left2 - 1] if left1 < left2 else [-1, -1]

    def isValidSudoku(self, board: list) -> bool:
        """
        36. 有效的数独
        :see https://leetcode-cn.com/problems/valid-sudoku/
        """

        def check(nums: list) -> bool:
            nums.sort()
            for i in range(1, len(nums)):
                if nums[i] == nums[i - 1] and nums[i] != '.':
                    # print(nums)
                    return False
            return True

        # 校验每行
        for row in board:
            if not check(row[:]):
                return False

        # 校验每列
        for j in range(9):
            if not check([board[i][j] for i in range(9)]):
                return False

        # 校验每小格
        for i in range(9):
            x = i // 3 * 3 + 1
            y = (3 * i + 1) % 9
            # print(x, y)
            if not check([board[x - 1][y - 1], board[x - 1][y], board[x - 1][y + 1],
                          board[x][y - 1], board[x][y], board[x][y + 1],
                          board[x + 1][y - 1], board[x + 1][y], board[x + 1][y + 1]]):
                return False

        return True

    def combinationSum(self, candidates: list, target: int) -> list:
        """
        39. 组合总和
        :see https://leetcode-cn.com/problems/combination-sum/
        """

        def backtrace(index: int, total: int, nums: list):
            if total == target:
                result.append(nums[:])
                return
            elif total > target:
                return

            for i in range(index, len(candidates)):
                nums.append(candidates[i])
                backtrace(i, total + candidates[i], nums)
                nums.pop()

        result = []
        backtrace(0, 0, [])
        return result

    def combinationSum2(self, candidates: list, target: int) -> list:
        """
        40. 组合总和 II
        :see https://leetcode-cn.com/problems/combination-sum-ii/
        """

        def backtrace(index, total: int, nums: list):
            # 终止条件
            if total == target:
                result.append(nums[:])
                return
            elif total > target:
                return

            for i in range(index, len(candidates)):
                # 去重
                if i > index and candidates[i] == candidates[i - 1]:
                    continue
                # 先加入这个值
                nums.append(candidates[i])
                # 回溯入口。注意，此处 i + 1 表示不能重复利用同一个数字，这是跟第39题的核心差别
                backtrace(i + 1, total + candidates[i], nums)
                # 这个值的回溯结束后，把这个值移除
                nums.pop()

        # 排序，以便回溯中的去重操作
        candidates.sort()
        result = []
        backtrace(0, 0, [])
        return result

    def smallestRange(self, nums: list) -> list:
        """
        632. 最小区间
        :see https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/
        """
        # 解题：思路将所有数组的数字升序放到一个大数组中，然后找到这个大数组中包含所有数组数字的最小区间即可
        # 1. 实现所有数组的合并和排序
        all_nums = []
        for i in range(len(nums)):
            for j in range(len(nums[i])):
                all_nums.append((nums[i][j], i, j))
        all_nums.sort()

        # 2. 使用滑动窗口，找到包含所有数组数组的子数组
        # 包左包右
        left, right = 0, 0
        # 用{数字所在数组: 该数组数字出现的次数}表示是否所有的数组都在这个all_nums[left:right+1]的子数组内
        index_dict = {all_nums[0][1]: 1}

        result = [all_nums[0][0], all_nums[-1][0]]
        while left <= right:
            # 右指针移动末尾时，结束即可
            if right == len(all_nums) - 1:
                break

            # 右指针右移，对应的数组数字出现次数+1
            right += 1
            index_dict[all_nums[right][1]] = index_dict.get(all_nums[right][1], 0) + 1

            # 左指针所在数组的数字若已经出现多次，则左指针右移
            while left < right and index_dict[all_nums[left][1]] > 1:
                index_dict[all_nums[left][1]] -= 1
                left += 1

            # 若此时窗口内的数字已包含所有数组，则更新result
            if len(index_dict) == len(nums) and all_nums[right][0] - all_nums[left][0] < result[1] - result[0]:
                result = [all_nums[left][0], all_nums[right][0]]

        return result

    def countGoodTriplets(self, arr: list, a: int, b: int, c: int) -> int:
        """
        统计好三元组
        :see
        """
        result = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                for k in range(j + 1, len(arr)):
                    if abs(arr[i] - arr[j]) <= a and abs(arr[j] - arr[k]) <= b and abs(arr[i] - arr[k]) <= c:
                        result += 1
        return result

    def getWinner(self, arr: list, k: int) -> int:
        """
        找出数组游戏的赢家
        :see
        """
        times = 0
        index = 0
        for i in range(1, len(arr)):
            if arr[i] < arr[index]:
                times += 1
            else:
                times = 1
                index = i

            if times == k:
                return arr[index]

        return arr[index]

    def maxSum(self, nums1: list, nums2: list) -> int:
        """
        最大得分
        :see
        """
        num1_dict = {}
        for i in range(len(nums1)):
            num1_dict[nums1[i]] = i

        nums2_dict = {}
        for i in range(len(nums2)):
            nums2_dict[nums2[i]] = i

        index1 = 0
        num1_s = 0
        index2 = 0
        num2_s = 0
        while index1 < len(nums1) or index2 < len(nums2):
            if index1 == len(nums1):
                num2_s += nums2[index2]
                index2 += 1
            elif index2 == len(nums2):
                num1_s += nums1[index1]
                index1 += 1
            elif nums1[index1] == nums2[index2]:
                r = nums1[index1] + max(num1_s, num2_s)
                num1_s = r
                num2_s = r
                index1 += 1
                index2 += 1
            elif nums1[index1] < nums2[index2]:
                num1_s += nums1[index1]
                index1 += 1
            else:
                num2_s += nums2[index2]
                index2 += 1

        return max(num1_s, num2_s) % (pow(10, 9) + 7)

    def minSwaps(self, grid: list) -> int:
        """
        排布二进制网格的最少交换次数
        :see
        """
        one_list = [0] * len(grid)
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    one_list[i] = j

            if one_list[i] == len(grid[i]):
                return -1

        # print(one_list)

        count = 0
        index = 0
        while index < len(one_list):
            if one_list[index] > index:
                tmp = index
                for i in range(index + 1, len(one_list)):
                    if one_list[i] <= index:
                        tmp = i
                        break
                # print(tmp)
                if tmp == index:
                    return -1
                count += tmp - index
                a = one_list.pop(tmp)
                one_list.insert(index, a)
            # print(index, one_list)

            index += 1

        return count

    def maxNonOverlapping(self, nums: list, target: int) -> int:
        """
        5471. 和为目标值的最大数目不重叠非空子数组数目
        :see https://leetcode-cn.com/problems/maximum-number-of-non-overlapping-subarrays-with-sum-equals-target/
        """
        prefix_sum = 0
        prefix_index_dict = {0: 0}
        dp = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_sum += nums[i]
            dp[i + 1] = dp[i]
            if prefix_sum - target in prefix_index_dict:
                index = prefix_index_dict[prefix_sum - target]
                dp[i + 1] = max(dp[i + 1], dp[index] + 1)
            prefix_index_dict[prefix_sum] = i + 1
        # print(prefix_index_dict)
        # print(dp)

        return dp[-1]

    def findKthPositive(self, arr: list, k: int) -> int:
        """
        5468. 第 k 个缺失的正整数
        :see https://leetcode-cn.com/problems/kth-missing-positive-number/
        """
        index = 1
        for i in range(len(arr)):
            while arr[i] != index:
                k -= 1
                if k == 0:
                    return index
                index += 1
            index += 1

        return index + k - 1 if index > 1 else index + k

    def maxDistance(self, position: list, m: int) -> int:
        """
        1552. 两球之间的磁力
        :see https://leetcode-cn.com/problems/magnetic-force-between-two-balls/
        """
        # 对position进行排序
        position.sort()

        low = 0
        high = position[-1]
        print('->', low, high)
        while low < high:
            # 使用二分法开始测试最多可以放几个球
            mid = (low + high) // 2

            # 以mid作为间距，来推测可以放多少个球
            count = 1
            old = position[0]
            for i in range(1, len(position)):
                if position[i] - old >= mid:
                    count += 1
                    old = position[i]

                if count > m:
                    break

            # print(mid, count)

            # 根据可放的count数量，来改变间距。可放的球数量count大于等于m，说明间距过小，应该增大间距，反之则减少
            if count >= m:
                low = mid + 1
            else:
                high = mid

            # print('->', low, high)

        return low - 1

    def judgePoint24(self, nums: list) -> bool:
        """
        679. 24 点游戏
        :see https://leetcode-cn.com/problems/24-game/
        """

        def four(nums: list):
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    left = []
                    for k in range(len(nums)):
                        if k != i and k != j:
                            left.append(nums[k])

                    if three(left + [nums[i] + nums[j]]) or three(left + [nums[i] - nums[j]]) or three(left + [nums[j] - nums[i]]) or three(
                            left + [nums[i] * nums[j]]) or (nums[j] != 0 and three(left + [nums[i] / nums[j]])) or (
                            nums[i] != 0 and three(left + [nums[j] / nums[i]])):
                        return True
            return False

        def three(nums: list):
            print(f'three: {nums}')
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    left = nums[3 - i - j]
                    if two(left, nums[i] + nums[j]) or two(left, nums[i] - nums[j]) or two(left, nums[j] - nums[i]) or two(left,
                                                                                                                           nums[i] * nums[j]) or (
                            nums[j] != 0 and two(left, nums[i] / nums[j])) or (nums[i] != 0 and two(left, nums[j] / nums[i])):
                        return True
            return False

        def two(a, b):
            print(f'two: {a}, {b}')
            return a + b == 24 or a - b == 24 or b - a == 24 or a * b == 24 or (b != 0 and abs(a / b - 24) < 0.00001) or (
                    a != 0 and abs(b / a - 24) < 0.00001)

        return four(nums)

    def minOperations(self, nums: list) -> int:
        """
        5481. 得到目标数组的最少函数调用次数
        :param nums:
        :return:
        """
        result = 0
        max_length = 0
        for i in nums:
            s = bin(i)
            max_length = max(max_length, len(s))
            result += s.count('1')
        return result + max_length - 3

    def mostVisited(self, n: int, rounds: list) -> list:
        """
        5495. 圆形赛道上经过次数最多的扇区
        :see https://leetcode-cn.com/problems/most-visited-sector-in-a-circular-track/
        """
        nums = [0] * (n + 1)
        nums[rounds[0]] += 1
        for i in range(1, len(rounds)):
            if rounds[i] > rounds[i - 1]:
                for j in range(rounds[i - 1] + 1, rounds[i] + 1):
                    nums[j] += 1
            else:
                for j in range(rounds[i - 1] + 1, n + 1):
                    nums[j] += 1
                for j in range(1, rounds[i] + 1):
                    nums[j] += 1

        sorted_nums = sorted([(nums[i], i) for i in range(len(nums))])[::-1]
        # print(sorted_nums)
        result = [sorted_nums[0][1]]
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i][0] == sorted_nums[0][0]:
                result.append(sorted_nums[i][1])
            else:
                break
        return sorted(result)

    def maxCoins(self, piles: list) -> int:
        """
        5496. 你可以获得的最大硬币数目
        :see https://leetcode-cn.com/problems/maximum-number-of-coins-you-can-get/
        """
        piles.sort(reverse=True)
        result = 0
        for i in range(2 * len(piles) // 3):
            if i % 2 == 1:
                result += piles[i]
        return result

    def findLatestStep(self, arr: list, m: int) -> int:
        """
        5497. 查找大小为 M 的最新分组
        :see https://leetcode-cn.com/problems/find-latest-group-of-size-m/
        """

    def findSubsequences(self, nums: list) -> list:
        """
        491. 递增子序列
        :see https://leetcode-cn.com/problems/increasing-subsequences/
        """
        # 用字典存以数字key结尾的所有递增子序列
        all_list = {}

        for i in nums:
            # 把单个数字也当做一个递增子序列，以便后续计算
            new_list = [[i]]
            # 遍历字典中所有结尾数字key小于等于当前数字的情况，将其所有递增子序列加上当前数字
            for j in all_list:
                if j <= i:
                    for k in all_list[j]:
                        new_list.append(k + [i])
            # 此时的new_list，就是遍历到i时，以数字i结尾的所有递增子序列
            all_list[i] = new_list

        # 从下标1开始的原因是，去掉由单个数字组成的递增子序列
        return [all_list[i][j] for i in all_list for j in range(1, len(all_list[i]))]

    def findLengthOfShortestSubarray(self, arr: list) -> int:
        """
        5493. 删除最短的子数组使剩余数组有序
        :see https://leetcode-cn.com/problems/shortest-subarray-to-be-removed-to-make-array-sorted/
        """
        import bisect
        # 右侧最长升序子序列
        right_begin = 0
        for i in range(len(arr) - 1, 0, -1):
            if arr[i] < arr[i - 1]:
                right_begin = i
                break

        # 删除左侧所有数字的长度
        result = right_begin
        if result == 0:
            return 0

        # 左侧最长上升子序列
        for i in range(0, len(arr)):
            if i > 0 and arr[i] < arr[i - 1]:
                break
            # 二分查找当前数字在右侧可插入的位置，当前数字的下标和插入位置之间的数字可被删除
            index = bisect.bisect_left(arr, arr[i], right_begin)
            result = min(result, index - 1 - i)
            print(arr[i], index, result)
        return result

    def numTriplets(self, nums1: list, nums2: list) -> int:
        """
        5508. 数的平方等于两数乘积的方法数
        :see https://leetcode-cn.com/problems/number-of-ways-where-square-of-number-is-equal-to-product-of-two-numbers/
        """
        num1_dict = {}
        num2_dict = {}

        for i in nums1:
            num1_dict[i] = num1_dict.get(i, 0) + 1
        for i in nums2:
            num2_dict[i] = num2_dict.get(i, 0) + 1

        result = 0

        for i in num1_dict:
            product = i * i
            for j in num2_dict:
                if product % j == 0:
                    other = product // j
                    if i > other:
                        continue
                    if j == other:
                        result += num1_dict[i] * num2_dict[j] * (num2_dict[j] - 1) // 2
                    elif other in num2_dict:
                        result += num1_dict[i] * num2_dict[j] * num2_dict[other]
                    # print(i, j, other, num1_dict[i], num2_dict[j], num2_dict.get(other, 0))

        for j in num2_dict:
            product = j * j
            for i in num1_dict:
                if product % i == 0:
                    other = product // i
                    if j > other:
                        continue
                    if i == other:
                        result += num2_dict[j] * num1_dict[i] * (num1_dict[i] - 1) // 2
                    elif other in num1_dict:
                        result += num2_dict[j] * num1_dict[i] * num1_dict[other]
                    # print(j, i, other, num2_dict[j], num1_dict[i], num1_dict.get(other, 0))

        return result

    def minCost(self, s: str, cost: list) -> int:
        """
        5509. 避免重复字母的最小删除成本
        :see https://leetcode-cn.com/problems/minimum-deletion-cost-to-avoid-repeating-letters/
        """
        # 加入最后一个字符，减少边界条件的处理
        s += ' '
        cost.append(0)

        result = 0
        max_cost = cost[0]
        total_cost = cost[0]

        for i in range(1, len(s)):
            if s[i] != s[i - 1]:
                # 当前字符将发生改变
                result += total_cost - max_cost

                # 重置计数
                max_cost = cost[i]
                total_cost = cost[i]
            else:
                max_cost = max(max_cost, cost[i])
                total_cost += cost[i]

        return result

    def numWays(self, s: str) -> int:
        """
        1573. 分割字符串的方案数
        :see https://leetcode-cn.com/problems/number-of-ways-to-split-a-string/
        """
        one_index = []
        for i in range(len(s)):
            if s[i] == '1':
                one_index.append(i)

        if len(one_index) % 3 != 0:
            return 0
        elif not one_index:
            # 左侧至少 1 个0，右侧至少 1 个0
            return (len(s) - 1) * (len(s) - 2) // 2 % (10 ** 9 + 7)

        length = len(one_index) // 3
        return (one_index[length] - one_index[length - 1]) * (one_index[2 * length] - one_index[2 * length - 1]) % (10 ** 9 + 7)

    def breakfastNumber(self, staple: list, drinks: list, x: int) -> int:
        """
        LCP 18. 早餐组合
        :see
        """
        import bisect
        result = 0
        drinks.sort()
        for i in staple:
            left = x - i
            result += bisect.bisect_right(drinks, left)
        return result

    def unhappyFriends(self, n: int, preferences: list, pairs: list) -> int:
        """
        1583. 统计不开心的朋友
        :see https://leetcode-cn.com/problems/count-unhappy-friends/
        """
        # 预处理
        preferences_dict_list = []
        for arr in preferences:
            preferences_dict_list.append({arr[i]: n - i for i in range(len(arr))})

        pairs_dict = {}
        for i in pairs:
            pairs_dict[i[0]] = i[1]
            pairs_dict[i[1]] = i[0]

        result = 0
        # 从0开始遍历
        for x in range(n):
            # 找到 x 的配对 y
            y = pairs_dict[x]
            # 找到 x 的亲密关系中，所有比 y 亲密的对象 u
            for u in preferences[x]:
                if u == y:
                    break
                # 找到 u 的配对 v
                v = pairs_dict[u]

                # 比较 u - x 的亲密关系和 u - v 的亲密关系
                if preferences_dict_list[u][x] > preferences_dict_list[u][v]:
                    result += 1
                    break

        return result

    def findRedundantConnection(self, edges: list) -> list:
        """
        684. 冗余连接
        :see https://leetcode-cn.com/problems/redundant-connection/
        """
        from LeetCode.Class.UnionFindClass import UnionFindClass
        union = UnionFindClass(len(edges) + 1)
        for u, v in edges:
            if union.find_root(u) == union.find_root(v):
                return [u, v]
            union.merge(u, v)
            # print(union.father)
        return []

    def isMagic(self, target: list) -> bool:
        """
        魔术排列
        :param target:
        :return:
        """
        l, r = 1, len(target)
        while l <= r:
            k = (l + r) // 2

            first_error = False
            arr = [i for i in range(1, len(target) + 1)]
            count = 0
            while arr:
                arr = arr[1::2] + arr[::2]

                length = min(k, len(arr))
                if target[count * k:count * k + length] == arr[:length]:
                    if not first_error:
                        l = k + 1
                        first_error = True
                    arr = arr[length:]
                else:
                    if not first_error:
                        r = k - 1
                    break
                count += 1

            if not arr:
                return True

        return False

    def minNumber(self, nums: list) -> str:
        """
        剑指 Offer 45. 把数组排成最小的数
        :see https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/
        """

        def compare(a: int, b: int) -> int:
            af = str(a) + str(b)
            bf = str(b) + str(a)
            if af < bf:
                return -1
            elif af > bf:
                return 1
            return 0

        import functools
        nums.sort(key=functools.cmp_to_key(compare))
        print(nums)
        return ''.join([str(i) for i in nums])

    def getMaxLen(self, nums: list) -> int:
        """
        1567. 乘积为正数的最长子数组长度
        :see https://leetcode-cn.com/problems/maximum-length-of-subarray-with-positive-product/
        """
        negative_length = 0
        positive_length = 0
        result = 0

        for i in range(len(nums)):
            if nums[i] == 0:
                negative_length = 0
                positive_length = 0
                continue

            if nums[i] < 0:
                positive_length, negative_length = negative_length, positive_length
                if positive_length > 0:
                    positive_length += 1
                negative_length += 1
            else:
                if negative_length > 0:
                    negative_length += 1
                positive_length += 1

            result = max(result, positive_length)

        return result

    def minTime(self, time: list, m: int) -> int:
        """
        LCP 12. 小张刷题计划
        :see https://leetcode-cn.com/problems/xiao-zhang-shua-ti-ji-hua/
        """
        l, r = 0, 10000
        while l < r:
            t = (l + r) // 2

            total_time = 0
            max_time = 0
            day = 0
            for i in time:
                total_time += i
                max_time = max(max_time, i)

                if total_time - max_time > t:
                    total_time = i
                    max_time = i
                    day += 1

            day += 1

            print(l, r, t, day)
            if day > m:
                l = t + 1
            else:
                r = t
        return l

    def countRangeSum(self, nums: list, lower: int, upper: int) -> int:
        """
        327. 区间和的个数
        :see https://leetcode-cn.com/problems/count-of-range-sum/
        """
        import bisect

        # 出现的前缀和的次数
        prefix_dict = {0: 1}
        # 出现的前缀和序列，无重复
        prefix_sum_list = [0]
        # 前缀和
        prefix_sum = 0

        result = 0

        for i in nums:
            prefix_sum += i

            # 左侧起点: prefix_sum - upper, 右侧终点: prefix_sum - lower
            index = bisect.bisect_left(prefix_sum_list, prefix_sum - upper)
            for j in range(index, len(prefix_sum_list)):
                if prefix_sum_list[j] <= prefix_sum - lower:
                    result += prefix_dict[prefix_sum_list[j]]
                else:
                    break

            print(prefix_dict, prefix_sum_list, index)

            # 添加当前前缀和
            if prefix_sum in prefix_dict:
                prefix_dict[prefix_sum] += 1
            else:
                prefix_dict[prefix_sum] = 1
                bisect.insort(prefix_sum_list, prefix_sum)

        return result

    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        """
        321. 拼接最大数
        :see https://leetcode-cn.com/problems/create-maximum-number/
        """
        '''
        从nums1中按顺序选出i个数字组成的最大数组
        再从nums2中按顺序选出k-i个数字组成的最大数组
        然后将选出的k个数字进行归并得到结果
        将最大的结果返回
        '''

        def select_max_number_list(arr: List[int]) -> List[List[int]]:
            """ 从nums1中逐个删除0个到删除len(nums1)个数字，使剩下数字组成的数组最大 """
            res = [arr[:]]
            for _ in range(len(arr)):
                # 找到第一个i+1比i大的数字，删除第i个数字即可
                i = 0
                while i < len(arr):
                    if i == len(arr) - 1:
                        arr.pop()
                    elif arr[i] < arr[i + 1]:
                        arr.pop(i)
                        break
                    else:
                        i += 1
                res.append(arr[:])
            return res

        def merge(arr1: List[int], arr2: List[int]) -> List[int]:
            """ 将两个数组合并，使合并后数组组成的数字最大 """
            res = []
            i = 0
            j = 0
            while i < len(arr1) or j < len(arr2):
                if i == len(arr1):
                    res.append(arr2[j])
                    j += 1
                elif j == len(arr2) or arr1[i] > arr2[j]:
                    res.append(arr1[i])
                    i += 1
                elif arr1[i] < arr2[j]:
                    res.append(arr2[j])
                    j += 1
                else:
                    # 当两个数字相同的时候，需要比较这两个数字后面的数字哪边大
                    comp_res = True
                    ti, tj = i, j
                    while ti < len(arr1) or tj < len(arr2):
                        if ti == len(arr1):
                            comp_res = False
                            break
                        elif tj == len(arr2):
                            break
                        elif arr1[ti] != arr2[tj]:
                            comp_res = arr1[ti] > arr2[tj]
                            break
                        ti += 1
                        tj += 1

                    if comp_res:
                        res.append(arr1[i])
                        i += 1
                    else:
                        res.append(arr2[j])
                        j += 1
            return res

        def compare_list(arr1: List[int], arr2: List[int]) -> bool:
            """ 比较两个数组的大小 """
            if len(arr1) != len(arr2):
                return len(arr1) > len(arr2)
            for i in range(len(arr1)):
                if arr1[i] > arr2[i]:
                    return True
                elif arr1[i] < arr2[i]:
                    return False
            return True

        length1 = len(nums1)
        length2 = len(nums2)

        max_nums1_list = select_max_number_list(nums1)
        max_nums2_list = select_max_number_list(nums2)

        result = []
        for i in range(max(0, k - length2), min(length1, k) + 1):
            tmp = merge(max_nums1_list[-i - 1], max_nums2_list[-k + i - 1])
            if compare_list(tmp, result):
                result = tmp
            # print(tmp, result)

        return result

    def isPossible(self, nums: List[int]) -> bool:
        """
        659. 分割数组为连续子序列
        :see https://leetcode-cn.com/problems/split-array-into-consecutive-subsequences/
        """
        import heapq
        from collections import defaultdict

        nums_dict = defaultdict(list)

        for i in nums:
            # 获得以 i - 1 结尾的最短序列，加上i，成为以 i 结尾序列
            heapq.heappush(nums_dict[i], heapq.heappop(nums_dict.get(i - 1, [0])) + 1)

            # 如果以 i - 1 结尾的序列，因为 pop 后被清空，则从 nums_dict 内删除
            if not nums_dict[i - 1]:
                del nums_dict[i - 1]

        # 如果有任意一个序列的最短长度小于3，则说明不能完成分割
        for i in nums_dict:
            if nums_dict[i][0] < 3:
                return False
        return True

    def leastInterval(self, tasks: List[str], n: int) -> int:
        """
        621. 任务调度器
        :see https://leetcode-cn.com/problems/task-scheduler/
        """
        if n == 0:
            return len(tasks)

        import heapq
        count_dict = {}
        for i in tasks:
            count_dict[i] = count_dict.get(i, 0) + 1

        arr = []
        for i in count_dict:
            heapq.heappush(arr, [-count_dict[i], i])

        cold = {}

        result = 0
        while arr or cold:
            result += 1
            if arr:
                # 取出出现次数最多的，且不在冷却期的字母
                item = heapq.heappop(arr)

                item[0] += 1
                if item[0] == 0:
                    pass
                elif n > 0:
                    cold[item[1]] = [item[0], n + 1]
                else:
                    heapq.heappush(arr, [item[0], item[1]])

            tmp = []
            for i in cold:
                cold[i][1] -= 1
                if cold[i][1] == 0:
                    heapq.heappush(arr, [cold[i][0], i])
                    tmp.append(i)

            for i in tmp:
                del cold[i]

            print(result, arr, cold)

        return result

    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        """
        373. 查找和最小的K对数字
        :see https://leetcode-cn.com/problems/find-k-pairs-with-smallest-sums/
        """
        if not nums1 or not nums2:
            return []

        import heapq
        # 最小堆初始化
        heap_queue = [(nums1[i] + nums2[0], i, 0) for i in range(len(nums1))]
        heapq.heapify(heap_queue)

        result = []
        while heap_queue:
            _, i, j = heapq.heappop(heap_queue)
            result.append([nums1[i], nums2[j]])

            if len(result) == k:
                break

            if j < len(nums2) - 1:
                heapq.heappush(heap_queue, (nums1[i] + nums2[j + 1], i, j + 1))

        return result

    def decode(self, encoded: List[int], first: int) -> List[int]:
        """
        5649. 解码异或后的数组
        :see https://leetcode-cn.com/problems/decode-xored-array/
        """
        result = [first]
        for i in encoded:
            result.append(result[-1] ^ i)
        return result

    def minimumHammingDistance(self, source: List[int], target: List[int], allowedSwaps: List[List[int]]) -> int:
        """
        5650. 执行交换操作后的最小汉明距离
        :see https://leetcode-cn.com/problems/minimize-hamming-distance-after-swap-operations/
        """
        from Class.UnionFindClass import UnionFindClass
        union = UnionFindClass(len(source))
        for l in allowedSwaps:
            union.merge(l[0], l[1])

        from collections import defaultdict
        group_dict = defaultdict(dict)
        for i in range(len(source)):
            father = union.find_root(i)
            group_dict[father][source[i]] = group_dict[father].get(source[i], 0) + 1

        result = 0
        for i in range(len(source)):
            father = union.find_root(i)
            if target[i] in group_dict[father]:
                group_dict[father][target[i]] -= 1
                result += 1

                if group_dict[father][target[i]] == 0:
                    del group_dict[father][target[i]]

        return len(source) - result

    def summaryRanges(self, nums: List[int]) -> List[str]:
        """
        228. 汇总区间
        :see https://leetcode-cn.com/problems/summary-ranges/
        """
        if not nums:
            return []
        nums.append(nums[0])

        result = []

        low_index = 0
        low = nums[0]
        for i in range(1, len(nums)):
            if nums[i] != low + i - low_index:
                if i - 1 > low_index:
                    result.append(f'{low}->{nums[i - 1]}')
                else:
                    result.append(f'{low}')
                low_index = i
                low = nums[i]

        return result


if __name__ == '__main__':
    s = Solution()
    print(s.summaryRanges([]))
