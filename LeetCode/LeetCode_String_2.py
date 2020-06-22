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
        for i in range(value_length + 1):
            # i 表示 a 字符串的长度，通过 a 字符串的长度计算出 b 字符串的长度
            str_b_length = (value_length - i * one_count) // two_count if two_count > 0 else 0
            if i * one_count + str_b_length * two_count != value_length:
                continue

            str_one = value[:i]
            str_two = value[i:i + str_b_length] if str_b_length > 0 else '0'

            if str_one == str_two:
                continue
            elif pattern.replace('1', str_one).replace('2', str_two) == value:
                return True

        return False


if __name__ == '__main__':
    s = Solution()
    print(s.patternMatching("bbbaa", "cccatat"))
