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


if __name__ == '__main__':
    s = Solution()
    print(s.findKthBit(4, 11))
