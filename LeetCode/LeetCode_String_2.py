class Solution:
    def patternMatching(self, pattern: str, value: str) -> bool:
        """
        面试题 16.18. 模式匹配
        :see https://leetcode-cn.com/problems/pattern-matching-lcci/
        """
        if not pattern and not value:
            return True
        elif not pattern:
            return False

        # 若首字符为a，则将a替换为1；否则将b替换为1
        start_ch = pattern[0]
        other_ch = 'a' if start_ch == 'b' else 'b'
        pattern = pattern.replace(start_ch, '1').replace(other_ch, '2')

        # 统计1和2出现的次数
        one_count = pattern.count('1')
        two_count = len(pattern) - one_count

        value_length = len(value)

        if value_length == 0:
            if one_count > 0 and two_count > 0:
                return False
            else:
                return True

        # 当第二个字符串的数量为0时，只需要判断第一个字符串替换后是否满足条件即可
        if two_count == 0:
            return pattern.replace('1', value[:value_length // one_count]) == value

        for i in range(value_length + 1):
            # i 表示第一个字符串的长度，通过第一个字符串的长度计算出另一个字符串的长度
            other_str_length = (value_length - i * one_count) // two_count
            if i * one_count + other_str_length * two_count != value_length:
                continue

            # 当另一个字符串的长度为0时，只需要判断第一个字符串即可
            str_one = value[:i]
            if i > 0 and other_str_length == 0:
                return pattern.replace('1', str_one).replace('2', '') == value

            if i == 0:
                if pattern.replace('1', '').replace('2', value[:other_str_length]) == value:
                    return True
            else:
                # 遍历每个长度为other_str_length的字符串，替换后进行对比
                for j in range(i, value_length - other_str_length + 1, i):
                    str_two = value[j:j + other_str_length]
                    if str_one != str_two and pattern.replace('1', str_one).replace('2', str_two) == value:
                        return True

        return False

    def isPathCrossing(self, path: str) -> bool:
        """
        5448. 判断路径是否相交
        :see https://leetcode-cn.com/problems/path-crossing/
        """
        node_set = {(0, 0)}
        x, y = 0, 0
        for i in path:
            if i == 'N':
                y += 1
            elif i == 'E':
                x += 1
            elif i == 'S':
                y -= 1
            else:
                x -= 1

            node = (x, y)
            if node in node_set:
                return True
            else:
                node_set.add(node)

        return False

    def shortestPalindrome(self, s: str) -> str:
        """
        214. 最短回文串
        :see https://leetcode-cn.com/problems/shortest-palindrome/
        """
        # 计算以左侧开始的最长回文串
        index = 1
        for i in range(1, len(s) + 1):
            string = s[:i]
            if string == string[::-1]:
                index = i

        return s[index::][::-1] + s

    def reformatDate(self, date: str) -> str:
        """
        5177. 转变日期格式
        :see
        """
        month_list = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        date_list = date.split(' ')
        month = str(month_list.index(date_list[1]))
        day = date_list[0][:-2]
        return f'{date_list[2]}-{month.zfill(2)}-{day.zfill(2)}'

    def convertToTitle(self, n: int) -> str:
        # return 'A' * ((n - 1) // 26) + chr((n - 1) % 26 + 65)
        result = []
        while n > 0:
            a = n % 26
            n //= 26
            if a == 0:
                result.append('Z')
                n -= 1
            else:
                result.append(chr(a + 64))
        return ''.join(result[::-1])

    def compareVersion(self, version1: str, version2: str) -> int:
        version1_list = version1.split('.')
        version2_list = version2.split('.')

        v1_len = len(version1_list)
        v2_len = len(version2_list)

        result = 0
        for i in range(max(v1_len, v2_len)):
            a = int(version1_list[i]) if i < v1_len else 0
            b = int(version2_list[i]) if i < v2_len else 0
            if a > b:
                result = 1
                break
            elif a < b:
                result = -1
                break

        return result

    def numSplits(self, s: str) -> int:
        """
        5458. 字符串的好分割数目
        :param s:
        :return:
        """
        left = {}
        right = {}
        for i in s:
            right[i] = right.get(i, 0) + 1

        result = 0
        for i in s:
            left[i] = left.get(i, 0) + 1
            right[i] -= 1
            if right[i] == 0:
                del right[i]
            if len(left) == len(right):
                result += 1

        return result

    def restoreString(self, s: str, indices: list) -> str:
        """
        5472. 重新排列字符串
        :see https://leetcode-cn.com/problems/shuffle-string/
        """
        result = [''] * len(indices)
        for i in range(len(indices)):
            result[indices[i]] = s[i]
        return ''.join(result)

    def minFlips(self, target: str) -> int:
        """
        5473. 灯泡开关 IV
        :see https://leetcode-cn.com/problems/bulb-switcher-iv/
        """
        last = '0'
        result = 0
        for i in target:
            if i != last:
                result += 1
                last = i
        return result

    def romanToInt(self, s: str) -> int:
        """
        13. 罗马数字转整数
        :see https://leetcode-cn.com/problems/roman-to-integer/
        """
        s += ' '
        result = 0
        index = 0
        while index < len(s):
            if s[index] == 'I':
                if s[index + 1] == 'V':
                    result += 4
                    index += 2
                elif s[index + 1] == 'X':
                    result += 9
                    index += 2
                else:
                    result += 1
                    index += 1
            elif s[index] == 'V':
                result += 5
                index += 1
            elif s[index] == 'X':
                if s[index + 1] == 'L':
                    result += 40
                    index += 2
                elif s[index + 1] == 'C':
                    result += 90
                    index += 2
                else:
                    result += 10
                    index += 1
            elif s[index] == 'L':
                result += 50
                index += 1
            elif s[index] == 'C':
                if s[index + 1] == 'D':
                    result += 400
                    index += 2
                elif s[index + 1] == 'M':
                    result += 900
                    index += 2
                else:
                    result += 100
                    index += 1
            elif s[index] == 'D':
                result += 500
                index += 1
            elif s[index] == 'M':
                result += 1000
                index += 1
            else:
                index += 1
        return result

    def intToRoman(self, num: int) -> str:
        """
        12. 整数转罗马数字
        :see https://leetcode-cn.com/problems/integer-to-roman/
        """
        num2ch_list = ((1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
                       (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'))
        result = []
        for n, ch in num2ch_list:
            while num >= n:
                result.append(ch)
                num -= n
        return ''.join(result)

    def multiply(self, num1: str, num2: str) -> str:
        """
        43. 字符串相乘
        :see https://leetcode-cn.com/problems/multiply-strings/
        """
        num1 = num1[::-1]
        num2 = num2[::-1]

        result = [0] * (len(num1) + len(num2) + 1)
        for i in range(len(num1)):
            tmp = [0] * (len(num2) + 1)
            carry = 0
            for j in range(len(num2)):
                r = int(num2[j]) * int(num1[i]) + carry
                if r >= 10:
                    carry = r // 10
                    r %= 10
                else:
                    carry = 0
                tmp[j] = r
            tmp[len(num2)] = carry

            for j in range(len(tmp)):
                result[i + j] += tmp[j]
                if result[i + j] >= 10:
                    result[i + j + 1] += result[i + j] // 10
                    result[i + j] %= 10

        while result and result[-1] == 0:
            result.pop()

        return ''.join([str(i) for i in result[::-1]]) if result else '0'

    def countBinarySubstrings(self, s: str) -> int:
        """
        696. 计数二进制子串
        :see https://leetcode-cn.com/problems/count-binary-substrings/
        """
        nums = []
        old, times = s[0], 0
        for i in s:
            if i == old:
                times += 1
            else:
                nums.append(times)
                old, times = i, 1
        nums.append(times)

        result = 0
        for i in range(len(nums) - 1):
            result += min(nums[i], nums[i + 1])

        return result

    def canConvertString(self, s: str, t: str, k: int) -> bool:
        """
        1540. K 次操作转变字符串
        :see https://leetcode-cn.com/problems/can-convert-string-in-k-moves/
        """
        if len(s) != len(t):
            return False

        times_list = [0] * 26
        for i in range(len(s)):
            diff = ord(t[i]) - ord(s[i])
            if diff < 0:
                diff += 26
            times_list[diff] += 1

        max_times = 0
        for i in range(1, len(times_list)):
            if times_list[i] > 0:
                max_times = max(max_times, (times_list[i] - 1) * 26 + i)

        return max_times <= k

    def minInsertions(self, s: str) -> int:
        """
        1541. 平衡括号字符串的最少插入次数
        :see https://leetcode-cn.com/problems/minimum-insertions-to-balance-a-parentheses-string/
        """
        # 遇到左括号+2，遇到右括号-1，
        # 分情况讨论，balance可能的值为0、奇数、非0偶数
        # 1. balance=0
        # (1) 遇到左括号，balance=2
        # (2) 遇到右括号，需要插入一个左括号，并使balance+=1
        # 2. balance=奇数
        # (1) 遇到左括号，需要插入一个右括号，并使balance+=1
        # (2) 遇到右括号，balance-=1
        # 3. balance=非0偶数
        # (1) 遇到左括号，balance+=2
        # (2) 遇到右括号，balance-=1
        balance = 0
        result = 0
        for i in s:
            if balance == 0:
                if i == '(':
                    balance += 2
                else:
                    result += 1
                    balance += 1
            elif balance % 2 == 1:
                if i == '(':
                    result += 1
                    balance += 1
                else:
                    balance -= 1
            else:
                if i == '(':
                    balance += 2
                else:
                    balance -= 1
        return result + balance

    def longestAwesome(self, s: str) -> int:
        """
        1542. 找出最长的超赞子字符串
        :see https://leetcode-cn.com/problems/find-longest-awesome-substring/submissions/
        """
        # 用二进制来表示当前数字数量的奇偶个数
        num = 0
        num_dict = {0: -1}
        result = 0
        for i in range(len(s)):
            num ^= 1 << int(s[i])
            if num in num_dict:
                result = max(result, i - num_dict[num])

            for j in range(10):
                tmp = num ^ (1 << j)
                if tmp in num_dict:
                    result = max(result, i - num_dict[tmp])

            if num not in num_dict:
                num_dict[num] = i

        return result

    def makeGood(self, s: str) -> str:
        """
        1544. 整理字符串
        :see https://leetcode-cn.com/problems/make-the-string-great/
        """
        string_list = list(s)
        while True:
            times = 0
            index = 0
            while index < len(string_list) - 1:
                if string_list[index] != string_list[index + 1] and (
                        string_list[index].upper() == string_list[index + 1] or string_list[index].lower() == string_list[index + 1]):
                    a = string_list.pop(index + 1)
                    b = string_list.pop(index)
                    times += 1
                    print(a, b)
                else:
                    index += 1
            if times == 0:
                break
            print(string_list)
        return ''.join(string_list)

    def findKthBit(self, n: int, k: int) -> str:
        """
        1545. 找出第 N 个二进制字符串中的第 K 位
        :see https://leetcode-cn.com/problems/find-kth-bit-in-nth-binary-string/
        """
        s = [0]
        for i in range(1, n):
            s = s + [1] + [1 - j for j in s[::-1]]
            # print(s)
        return str(s[k - 1])

    def isNumber(self, s: str) -> bool:
        """
        剑指 Offer 20. 表示数值的字符串
        :see https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/
        """
        import re
        return re.match(r'^\s*[+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?\s*$', s) is not None

    def calculate(self, s: str) -> int:
        """
        LCP 17. 速算机器人
        :see
        """
        x, y = 1, 0
        for i in s:
            if i == 'A':
                x = 2 * x + y
            else:
                y = 2 * y + x
        return x + y

    def minimumOperations(self, leaves: str) -> int:
        """
        LCP 19. 秋叶收藏集
        :see
        """
        '''
        1. 对两侧的字符进行特殊处理。这样做的好处可以减少边界情况的处理
            1) 如果一侧为y，则需要变为r，调整次数至少为1；如果两侧都是y，那么至少为2
        2. 从最简单的思路去想，要变成ryr的形式，需要先选中一个区间，这个区间内所有字母都为y，
           那么要把区间左侧和右侧的y变成r，把区间中间的r都变成y
        3. 接下来就是化简。
            1) 最佳的情况下，选中的区间的左侧必然是一组连续y的开始，右侧恰好是一组连续y的结束。
            2) 此时，需要调整的次数为: 区间左侧y的数量 + 区间内r的数量 + 区间右侧y的数量
            3) 假设 count[i] 为下标i(包括下标i)左侧y的数量，区间为 start ~ end (区间包括start, 不包括end)，那么
                区间左侧y的数量: count[start] - 1
                区间内部r的数量: (end - start - 1) - (count[end] - count[start])
                区间右侧y的数量: count[-1] - count[end]
                需要调整的数量为: count[-1] - 2 + (2 * count[start] - start) - (2 * count[end] - end)
            4) 上面那个式子，可以将 2 * count[i] - i 实为一个变量 f[i]，那么这个问题就转化为
                求最小调整的数量为 count[-1] - 2 + min(f[start] - f[end])，
                其中，start 为一组连续的y开始的坐标，end为一组连续的y结束的下一个坐标
        '''
        # 首尾特殊处理
        result = 0
        if leaves[0] == 'y':
            result += 1
        if leaves[-1] == 'y':
            result += 1

        # 每一组连续的y的下标
        yellow_index = []
        start = -1
        # 左侧y的数量 * 2 - 下标
        arr = [float('inf')]
        y_count = 0

        # 因为首尾已经处理，所以从下标1开始算
        for i in range(1, len(leaves) - 1):
            if leaves[i] == 'y':
                if i == 1 or leaves[i - 1] == 'r':
                    start = i
                y_count += 1
            elif leaves[i] == 'r' and start != -1 and leaves[i - 1] == 'y':
                yellow_index.append((start, i))
            arr.append(2 * y_count - i)

        # 对最后一组y进行处理
        if leaves[len(leaves) - 2] == 'y':
            yellow_index.append((start, len(leaves) - 1))
        arr.append(2 * y_count - len(leaves) + 1)

        # 当有一组连续的y时
        if len(yellow_index) == 1:
            return result
        # 当一个y都没有时
        if len(yellow_index) == 0:
            return result + 1

        # 保持 f[start] 最小，遍历 f[end]，求 f[start] - f[end] 最小的差值
        min_func = float('inf')
        rr = float('inf')
        for i in range(len(yellow_index)):
            start = yellow_index[i][0]
            end = yellow_index[i][1]

            min_func = min(min_func, arr[start])
            rr = min(rr, min_func - arr[end])
            # print(i, start, end, min_func, rr)

        return result + y_count - 2 + rr

    def partitionLabels(self, S: str) -> list:
        """
        763. 划分字母区间
        :see https://leetcode-cn.com/problems/partition-labels/
        """
        end_dict = {}
        for i in range(len(S)):
            end_dict[S[i]] = i

        result = []

        start_index = -1
        ch_set = set()

        for i in range(len(S)):
            ch_set.add(S[i])

            # 当这个字母是最后一次出现时，移除
            if end_dict[S[i]] == i:
                ch_set.remove(S[i])

            if not ch_set:
                result.append(i - start_index)
                start_index = i

        return result

    def reorganizeString(self, S: str) -> str:
        """
        767. 重构字符串
        :see https://leetcode-cn.com/problems/reorganize-string/
        """
        ch_list = [[0, chr(i + 97)] for i in range(26)]
        for i in S:
            # heapq模块只有最小堆，因此用负数实现最大堆
            ch_list[ord(i) - 97][0] -= 1

        # 初始化最大堆
        import heapq
        heapq.heapify(ch_list)

        result = []

        while ch_list[0][0] < 0:
            # 最多数量的两个字符组成一对，将这两个字符重新放入最大堆。
            # 因为最大堆的特性，当能凑对时，本次的second!=下次的first。
            first = heapq.heappop(ch_list)
            second = heapq.heappop(ch_list)

            # 如果最大的字符和上一个字符相同，说明这个字符的数量过多，已经无法凑对。
            if result and first[1] == result[-1]:
                return ''

            result.append(first[1])
            first[0] += 1

            if second[0] < 0:
                result.append(second[1])
                second[0] += 1

            heapq.heappush(ch_list, first)
            heapq.heappush(ch_list, second)

        return ''.join(result)

    def splitIntoFibonacci(self, S: str) -> list:
        """
        842. 将数组拆分成斐波那契序列
        :see https://leetcode-cn.com/problems/split-array-into-fibonacci-sequence/
        """

        def check(l: int, r: int, string: str) -> bool:
            """ 确认以l和r为前缀，s能否分割成斐波那契序列 """
            # 整形限制
            if r >= 1 << 31:
                return False
            result.append(r)
            next = str(l + r)
            return not string or string.startswith(next) and check(r, l + r, string[len(next):])

        if len(S) < 3:
            return []

        for i in range(1, len(S) // 2 + 1):
            # 开头为0时，这个数字只能为0
            if S[0] == '0' and i > 1:
                break

            # 第一个数字为S[:i]
            first = int(S[:i])
            # 整形限制
            if first >= 1 << 31:
                break

            for j in range(i + 1, len(S)):
                # 开头为0时，这个数字只能为0
                if S[i] == '0' and j > i + 1:
                    break

                # 第二个数字为S[i:j]
                second = int(S[i:j])

                result = [first]
                if check(first, second, S[j:]) and len(result) > 2:
                    return result

        return []

    def predictPartyVictory(self, senate: str) -> str:
        """
        649. Dota2 参议院
        :see https://leetcode-cn.com/problems/dota2-senate/
        """
        ch_list = list(senate)

        # 把排在自己后面的最近的非同阵营的投出去
        # 上一轮多余的投票可以留到下一轮
        r_right_count, d_right_count = 0, 0

        while True:
            # 本轮可投票的人数
            r_count, d_count, = 0, 0

            for i in range(len(ch_list)):
                if ch_list[i] == 'R':
                    if d_right_count > 0:
                        d_right_count -= 1
                        # 被投出去就变成空字符串
                        ch_list[i] = ''
                    else:
                        r_count += 1
                        r_right_count += 1
                elif ch_list[i] == 'D':
                    if r_right_count > 0:
                        r_right_count -= 1
                        ch_list[i] = ''
                    else:
                        d_count += 1
                        d_right_count += 1

            if r_count == 0:
                return 'Dire'
            elif d_count == 0:
                return 'Radiant'

    def removeDuplicateLetters(self, s: str) -> str:
        """
        316. 去除重复字母
        :see https://leetcode-cn.com/problems/remove-duplicate-letters/
        """
        ch_count_dict = {}
        for i in s:
            ch_count_dict[i] = ch_count_dict.get(i, 0) + 1

        used_set = set()
        stack = []

        for i in s:
            if i not in used_set:
                while stack and i <= stack[-1] and ch_count_dict[stack[-1]] > 1:
                    ch_count_dict[stack[-1]] -= 1
                    used_set.remove(stack[-1])
                    stack.pop()

                used_set.add(i)
                stack.append(i)
            else:
                ch_count_dict[i] -= 1

        return ''.join(stack)


if __name__ == '__main__':
    s = Solution()
    print(s.removeDuplicateLetters("bbcaac"))
