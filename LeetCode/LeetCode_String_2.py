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


if __name__ == '__main__':
    s = Solution()
    print(s.minFlips('001011101'))
