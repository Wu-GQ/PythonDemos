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


if __name__ == '__main__':
    s = Solution()
    print(s.patternMatching("aa", ""))
