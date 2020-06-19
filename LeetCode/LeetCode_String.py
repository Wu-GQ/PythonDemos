from io import StringIO
from queue import Queue


class Solution:

    def strstr(self, haystack: str, needle: str) -> int:
        """
        实现strStr()
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/5/strings/38/
        """
        # if len(needle) == 0:
        #     return 0
        return haystack.find(needle)

    def word_break(self, s: str, wordDict: list) -> bool:
        """
        单词拆分
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/275/string/1138/
        """
        match_list = [1] + [0] * len(s)
        for i in range(1, len(s) + 1):
            for word in wordDict:
                if s[:i].endswith(word) and match_list[i - len(word)] == 1:
                    match_list[i] = 1
                    break

        return bool(match_list[len(s)])

    def word_break_2(self, s: str, wordDict: list) -> list:
        """
        单词拆分 II
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/275/string/1139/
        """
        point_list = [[] for i in range(len(s))]
        point_list.insert(0, [0])

        # 通过遍历获得所有可能的路径点
        for i in range(1, len(s) + 1):
            for word in wordDict:
                if s[:i].endswith(word) and len(point_list[i - len(word)]) > 0:
                    point_list[i].append(i - len(word))

        if len(point_list[len(s)]) == 0:
            return []

        # 遍历输出所有路径
        path_list = [[] for i in range(len(s) + 1)]

        for i in range(len(s), 0, -1):
            for point_index in point_list[i]:
                now_path = s[point_index:i]

                if i == len(s):
                    path_list[point_index].append(now_path)
                else:
                    for path in path_list[i]:
                        path_list[point_index].append("%s %s" % (now_path, path))

        return path_list[0]

    def reverse_num(self, x: int) -> int:
        """
        整数反转
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/5/strings/33/
        """
        num_string = str(abs(x))[::-1]
        if (x >= 0 and float(num_string) > pow(2, 31) - 1) or (x < 0 and float(num_string) > pow(2, 31)):
            return 0
        elif x >= 0:
            return int(num_string)
        else:
            return -int(num_string)

    def longestCommonPrefix(self, strs: list) -> str:
        """
        最长公共前缀
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/5/strings/40/
        """
        if len(strs) == 0:
            return ""

        i = 0
        while i < len(strs[0]):
            ch = strs[0][i]
            for string in strs:
                if i >= len(string) or string[i] != ch:
                    return strs[0][:i]
            i += 1
        return strs[0]

    def is_palindrome(self, s: str) -> bool:
        """
        125. 验证回文串
        :see https://leetcode-cn.com/explore/featured/card/top-interview-quesitons-in-2018/275/string/1136/
        """
        i, j = 0, len(s) - 1
        while i <= j:
            if not s[i].isalnum():
                i += 1
            elif not s[j].isalnum():
                j -= 1
            elif s[i].upper() != s[j].upper():
                return False
            else:
                i += 1
                j -= 1
        return True

    def partition(self, s: str):
        """
        验证回文串
        :see https://leetcode-cn.com/explore/featured/card/top-interview-quesitons-in-2018/275/string/1137/
        """
        length = len(s)
        # 将问题转换为输出从0节点 - (length - 1)节点的所有路径
        point_array = [[] for i in range(length)]

        for j in range(length):
            for i in range(length - j):
                if self.is_palindrome_in_string(s, i, j):
                    point_array[j].append(length - i)

        # 从(length - 1)节点开始遍历所有路径
        path_array = [[] for i in range(length)]

        for i in range(length - 1, -1, -1):
            for point_index in point_array[i]:
                now_path = s[i:point_index]
                if point_index == length:
                    path_array[i].append([now_path])
                else:
                    for path_list in path_array[point_index]:
                        a = path_list.copy()
                        a.append(now_path)
                        path_array[i].append(a)

        for result_list in path_array[0]:
            result_list.reverse()

        return path_array[0]

    def is_palindrome_in_string(self, s, start_x, start_y) -> bool:
        """ 判断是否为回文串 """
        y = start_y
        for i in range(len(s) - start_x - start_y):
            if s[y + i] != s[len(s) - start_x - i - 1]:
                return False
            elif y + i >= len(s) - start_x - i - 1:
                return True
        return True

    def min_cut(self, s: str) -> int:
        """
        分割回文串 II
        :see https://leetcode-cn.com/problems/palindrome-partitioning-ii/
        """
        length = len(s)
        # 将问题转换为输出从0节点 - (length - 1)节点的所有路径
        point_array = [[] for i in range(length + 1)]

        for j in range(length):
            for i in range(length - j):
                if self.is_palindrome_in_string(s, i, j):
                    point_array[j].append(length - i)

        # 用广度优先搜索0节点到(length-1)节点的最短路径
        search_queue = Queue()
        search_queue.put(0)

        step_dict = {0: 0}

        while not search_queue.empty():
            index = search_queue.get()
            for point in point_array[index]:
                if point == length:
                    return step_dict[index] + 1

                if point not in step_dict:
                    step_dict[point] = step_dict[index] + 1
                    search_queue.put(point)

        return step_dict[length]

    def reverse_string(self, s: list) -> None:
        """
        反转字符串
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/275/string/1144/
        """
        # 应该用C或者C++来写，因为python没有char类型
        length = len(s)
        for i in range(int(length / 2)):
            s[i] ^= s[length - i - 1]
            s[length - i - 1] ^= s[i]
            s[i] ^= s[length - i - 1]
        print(s)

    def first_uniq_char(self, s: str) -> int:
        """
        字符串中的第一个唯一字符
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/275/string/1143/
        """
        # 获得字母出现的次数和第一次出现的位置
        char_dict = {}
        first_index_dict = {}

        for i in range(len(s)):
            if s[i] in char_dict:
                char_dict[s[i]] += 1
            else:
                char_dict[s[i]] = 1
                first_index_dict[s[i]] = i

        # 获得最小的第一次出现的位置
        min_index = float("inf")
        for i in char_dict:
            if char_dict[i] == 1 and i in first_index_dict and first_index_dict[i] < min_index:
                min_index = first_index_dict[i]

        return min_index if min_index != float('inf') else -1

    def is_anagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        char_dict_a = {}
        char_dict_b = {}

        for i in range(len(s)):
            if s[i] in char_dict_a:
                char_dict_a[s[i]] += 1
            else:
                char_dict_a[s[i]] = 1

            if t[i] in char_dict_b:
                char_dict_b[t[i]] += 1
            else:
                char_dict_b[t[i]] = 1

        return char_dict_a == char_dict_b

    count_and_say_array = ["1"]

    def countAndSay(self, n: int) -> str:
        """
        报数
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/5/strings/39/
        """
        if n > len(self.count_and_say_array):
            for i in range(len(self.count_and_say_array), n):
                last_word = self.count_and_say_array[i - 1]

                word = ""
                char = ""
                char_count = 0

                for j in last_word:
                    if char_count == 0:
                        char = j
                        char_count = 1
                    elif j == char:
                        char_count += 1
                    else:
                        word = "{0}{1}{2}".format(word, char_count, char)
                        char_count = 1
                        char = j

                if char_count > 0:
                    word = "{0}{1}{2}".format(word, char_count, char)
                self.count_and_say_array.append(word)

        return self.count_and_say_array[n - 1]

    def isMatch(self, s: str, p: str) -> bool:
        """
        正则表达式匹配（回溯算法， 可优化)
        :see https://leetcode-cn.com/problems/regular-expression-matching/
        """
        if len(s) == 0 and len(p) == 0:
            return True
        elif len(s) > 0 and len(p) == 0:
            return False

        if len(p) > 1 and p[1] == '*':
            # 当第二个字符为'*'时，可匹配多个字符
            if len(s) > 0 and (s[0] == p[0] or p[0] == '.'):
                # 若s和p的第一字符相同， *分别匹配多次和0次
                return self.isMatch(s[1:], p) or self.isMatch(s, p[2:])
            else:
                # 若s和p的第一个字符不相同，*匹配0次
                return self.isMatch(s, p[2:])
        elif len(s) > 0 and (s[0] == p[0] or p[0] == '.'):
            return self.isMatch(s[1:], p[1:])
        else:
            return False

    def isMatch_2(self, s: str, p: str) -> bool:
        """
        正则表达式匹配（动态规划）
        :see https://leetcode-cn.com/problems/regular-expression-matching/
        """
        if len(s) == 0 and len(p) == 0:
            return True
        elif len(s) > 0 and len(p) == 0:
            return False

        # 此处在两个字符串前加空格，是为了减少边界条件的处理
        s = f' {s}'
        p = f' {p}'

        s_length = len(s)
        p_length = len(p)

        # 存储比较结果
        compare_list = [[False] * p_length for j in range(s_length)]

        for i in range(s_length):
            for j in range(p_length):
                if i == 0 and j == 0:
                    compare_list[i][j] = True
                elif i > 0 and j == 0:
                    compare_list[i][j] = False
                elif p[j] != '*' and (s[i] == p[j] or (p[j] == '.' and i > 0)):
                    compare_list[i][j] = compare_list[i - 1][j - 1]
                elif p[j] == '*' and (s[i] == p[j - 1] or (p[j - 1] == '.' and i > 0)):
                    compare_list[i][j] = compare_list[i - 1][j] or compare_list[i][j - 2]
                elif p[j] == '*' and s[i] != p[j - 1]:
                    compare_list[i][j] = compare_list[i][j - 2]

        return compare_list[s_length - 1][p_length - 1]

    def isMatch2(self, s: str, p: str) -> bool:
        """
        通配符匹配
        :see https://leetcode-cn.com/problems/wildcard-matching/
        """
        # 此处在两个字符串前加空格，是为了减少边界条件的处理
        s = f' {s}'
        p = f' {p}'

        s_length = len(s)
        p_length = len(p)

        compare_list = [[False for i in range(p_length)] for j in range(s_length)]

        for i in range(s_length):
            for j in range(p_length):
                if i == 0 and j == 0:
                    compare_list[i][j] = True
                elif i > 0 and j == 0:
                    compare_list[i][j] = False
                elif p[j] == '*':
                    compare_list[i][j] = compare_list[i - 1][j] or compare_list[i][j - 1]
                elif p[j] != '*' and (s[i] == p[j] or i > 0 and p[j] == '?'):
                    compare_list[i][j] = compare_list[i - 1][j - 1]

        return compare_list[s_length - 1][p_length - 1]

    def minWindow(self, s: str, t: str) -> str:
        """
        最小覆盖子串
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-hard/55/array-and-strings/133/
        """
        t_length = len(t)
        s_length = len(s)
        if t_length > s_length:
            return ""

        # 被覆盖的字符序列
        selected_ch_stack = []
        # 被覆盖的字符下标
        selected_ch_index_stack = []
        # 已被覆盖的字符总数量
        selected_ch_total_count = 0
        # 未被覆盖的字符数量
        unselected_ch_count_dict = {}
        for i in t:
            if i in unselected_ch_count_dict:
                unselected_ch_count_dict[i] += 1
            else:
                unselected_ch_count_dict[i] = 1

        min_length = float("inf")
        start_index = -1
        end_index = -1

        for i in range(0, s_length):
            if s[i] in unselected_ch_count_dict:
                # 将新字符加入已覆盖字符队列
                unselected_ch_count_dict[s[i]] -= 1
                selected_ch_stack.append(s[i])
                selected_ch_index_stack.append(i)

                if unselected_ch_count_dict[s[i]] >= 0:
                    selected_ch_total_count += 1
                elif s[i] == selected_ch_stack[0]:
                    # 从头开始删除重复覆盖的字符
                    while len(selected_ch_stack) > 0 > unselected_ch_count_dict[selected_ch_stack[0]]:
                        ch = selected_ch_stack.pop(0)
                        selected_ch_index_stack.pop(0)
                        unselected_ch_count_dict[ch] += 1

                # 更新最短距离
                if selected_ch_total_count == t_length:
                    length = selected_ch_index_stack[-1] - selected_ch_index_stack[0]
                    if length < min_length:
                        start_index, end_index, min_length = selected_ch_index_stack[0], selected_ch_index_stack[
                            -1], length

        return s[start_index:end_index + 1] if start_index != -1 and end_index != -1 else ""

    def gcdOfStrings(self, str1: str, str2: str) -> str:
        """
        1071. 字符串的最大公因子
        :see https://leetcode-cn.com/problems/greatest-common-divisor-of-strings/
        """

        def is_same_prefix_string(a: str, b: str) -> bool:
            """ a字符串和b字符串是否具有相同前缀 """
            for i in range(len(b)):
                if a[i] != b[i]:
                    return False
            return True

        while len(str1) > 0 and len(str2) > 0:
            if len(str1) < len(str2):
                str1, str2 = str2, str1
            if is_same_prefix_string(str1, str2):
                str1 = str1[len(str2):]
            else:
                return ""

        return str1 + str2

    def compressString(self, S: str) -> str:
        """
        面试题 01.06. 字符串压缩
        :see https://leetcode-cn.com/problems/compress-string-lcci/
        """
        # 统计字母数量
        new_string_list = []
        for i in S:
            if len(new_string_list) > 1 and i == new_string_list[-2]:
                new_string_list[-1] += 1
            else:
                new_string_list.append(i)
                new_string_list.append(1)

        # 输出
        if len(new_string_list) >= len(S):
            return S

        string = StringIO()
        for i in new_string_list:
            if i is str:
                string.write(i)
            else:
                string.write(str(i))
        return string.getvalue()

    def countCharacters(self, words: list, chars: str) -> int:
        """
        1160. 拼写单词
        :see https://leetcode-cn.com/problems/find-words-that-can-be-formed-by-characters/
        """
        # 先统计字母表的字符数量
        char_count_dict = {}
        for i in chars:
            if i in char_count_dict:
                char_count_dict[i] += 1
            else:
                char_count_dict[i] = 1

        # 逐个分析每个词汇的字符数量
        count = 0
        for word in words:
            word_count_dict = {}
            is_break = False

            for i in word:
                if i not in char_count_dict:
                    is_break = True
                    break
                elif i in word_count_dict:
                    word_count_dict[i] += 1
                    if word_count_dict[i] > char_count_dict[i]:
                        is_break = True
                        break
                else:
                    word_count_dict[i] = 1

            if not is_break:
                count += len(word)

        return count

    def longestPalindrome(self, s: str) -> int:
        """
        409. 最长回文串
        :see https://leetcode-cn.com/problems/longest-palindrome/
        """
        # 找有多少成对的字符，如果还有单个的字符，可以排在中间
        """
        if len(s) < 1:
            return 0

        char_set = set()
        max_length = 0
        for i in s:
            if i in char_set:
                char_set.remove(i)
                max_length += 2
            else:
                char_set.add(i)
        
        return max_length + (1 if len(char_set) > 0 else 0)
        """
        if len(s) < 1:
            return 0

        char_list = [0] * 52
        max_length = 0
        single_char = 0

        for i in s:
            index = ord(i) - 64 if ord(i) < 91 else ord(i) - 71
            if char_list[index] > 0:
                max_length += 2
                char_list[index] = 0
                single_char -= 1
            else:
                char_list[index] = 1
                single_char += 1

        return max_length + (1 if single_char > 0 else 0)

    def myAtoi(self, str: str) -> int:
        """
        8. 字符串转换整数 (atoi)
        :see https://leetcode-cn.com/problems/string-to-integer-atoi/
        """
        min_int = -2 << 30
        max_int = (2 << 30) - 1

        ord_zero = ord('0')
        ord_nine = ord('9')

        # 0-正负可选，1代表正号，2表示负号
        symbol_type = 0
        # 是否允许空格
        is_spaces_allowed = True

        result = 0

        for i in str:
            if i == ' ':
                if is_spaces_allowed:
                    continue
                else:
                    break
            elif i == '-' or i == '+':
                if symbol_type > 0:
                    break
                else:
                    symbol_type = 1 if i == '+' else 2
                is_spaces_allowed = False
            elif ord_zero <= ord(i) <= ord_nine:
                result = result * 10 - ord_zero + ord(i)
                if (symbol_type != 2) and result >= max_int:
                    return max_int
                elif symbol_type == 2 and -result <= min_int:
                    return min_int

                is_spaces_allowed = False
                if symbol_type == 0:
                    symbol_type = 1
            else:
                break

        return -result if symbol_type == 2 else result

    def canConstruct(self, s: str, k: int) -> bool:
        """
        构造 K 个回文字符串
        :param s:
        :param k:
        :return:
        """
        char_dict = {}
        for i in s:
            char_dict[i] = char_dict.get(i, 0) + 1

        pair_count = 0
        not_pair_count = 0
        for i in char_dict:
            if char_dict[i] % 2 == 0:
                pair_count += 1
            else:
                not_pair_count += 1

        # min_count = not_pair_count if not_pair_count >= pair_count else not_pair_count
        return not_pair_count <= k <= len(s)

    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        """
        最长快乐字符串
        :param a:
        :param b:
        :param c:
        :return:
        """
        nums = [[a, 'a'], [b, 'b'], [c, 'c']]
        nums.sort(key=lambda num: num[0], reverse=True)

        result = []
        last_char = ''
        while nums[0][0] > 0:
            if last_char == nums[0][1]:
                if nums[1][0] == 0:
                    break
                nums[1][0] -= 1
                result.append(nums[1][1])
                last_char = nums[1][1]
            else:
                if nums[0][0] == 1:
                    result.append(nums[0][1])
                    nums[0][0] = 0
                else:
                    result.append(nums[0][1])
                    result.append(nums[0][1])
                    nums[0][0] -= 2
                last_char = nums[0][1]

            # if nums[1][0] == 0:
            #     break
            # nums[1][0] -= 1
            # result.append(nums[1][1])
            # last_char = nums[1][1]

            nums.sort(key=lambda num: num[0], reverse=True)

        return ''.join(result)

    def entityParser(self, text: str) -> str:
        """
        5382. HTML 实体解析器
        :param text:
        :return:
        """
        return text.replace('&quot;', '\\"').replace('&apos;', "\\'").replace('&amp;', '&').replace('&gt;',
                                                                                                    '>').replace('&lt;',
                                                                                                                 '<').replace(
            '&frasl;', '/')

    def reverseWords(self, s: str) -> str:
        """
        151. 翻转字符串里的单词
        :see https://leetcode-cn.com/problems/reverse-words-in-a-string/
        """
        string_list = s.split()
        # print(string_list)
        string_list.reverse()
        return ' '.join(string_list)

    def getHappyString(self, n: int, k: int) -> str:
        """长度为 n 的开心字符串中字典序第 k 小的字符串"""

        def add_one() -> bool:
            string_list[-1] += 1
            carry = 0
            for i in range(len(string_list) - 1, -1, -1):
                a = string_list[i] + carry
                if a > 2:
                    carry = a // 3
                    string_list[i] = a % 3
                else:
                    string_list[i] = a
                    carry = 0
                    break

            if carry > 0:
                return False

            repeat = False
            for i in range(len(string_list) - 1):
                if string_list[i] == string_list[i + 1]:
                    repeat = True
                    break

            if repeat:
                return add_one()
            else:
                return True

        string_list = []
        for i in range(n):
            string_list.append(i % 2)

        for i in range(k - 1):
            if not add_one():
                return ""

        for i in range(len(string_list)):
            if string_list[i] == 0:
                string_list[i] = 'a'
            elif string_list[i] == 1:
                string_list[i] = 'b'
            elif string_list[i] == 2:
                string_list[i] = 'c'
            else:
                print('Error')

        return ''.join(string_list)

    def breakPalindrome(self, palindrome: str) -> str:
        """
        1328. 破坏回文串
        :see https://leetcode-cn.com/problems/break-a-palindrome/
        """
        if len(palindrome) < 2:
            return ''

        index = -1
        for i in range(len(palindrome) // 2):
            if palindrome[i] != 'a':
                index = i
                break

        return f'{palindrome[:index]}a{palindrome[index + 1:]}' if index != -1 else f'{palindrome[:-1]}b'

    def reformat(self, s: str) -> str:
        """重新格式化字符串"""
        ch_list = []
        num_list = []

        for i in s:
            if i.isdigit():
                num_list.append(i)
            else:
                ch_list.append(i)

        if abs(len(num_list) - len(ch_list)) > 1:
            return ''

        string_list = []
        length = len(s)
        if len(num_list) > len(ch_list):
            string_list.append(num_list.pop(0))
            length -= 1
        for i in range(length):
            if i % 2 == 0:
                string_list.append(ch_list[i // 2])
            else:
                string_list.append(num_list[i // 2])

        return ''.join(string_list)

    def maxScore(self, s: str) -> int:
        """
        5392. 分割字符串的最大得分
        :param s:
        :return:
        """
        one_total_count = 0
        for i in s:
            if i == '1':
                one_total_count += 1

        max_result = one_total_count
        result = one_total_count

        if s[0] == '1':
            max_result -= 1
            result -= 1
        else:
            max_result += 1
            result += 1

        for i in range(1, len(s) - 1):
            if s[i] == '0':
                result += 1
            else:
                result -= 1
            max_result = max(max_result, result)
            # print(s[i], result, max_result)

        return max_result

    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        3. 无重复字符的最长子串
        :see https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/
        """
        # 滑动窗口
        left = 0
        right = 0

        char_set = set()

        max_length = 0

        while right < len(s):
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1

            char_set.add(s[right])
            right += 1

            max_length = max(max_length, len(char_set))

        return max_length

    def checkIfCanBreak(self, s1: str, s2: str) -> bool:
        """
        5386. 检查一个字符串是否可以打破另一个字符串
        :param s1:
        :param s2:
        :return:
        """
        s1_list = sorted(list(s1))
        s2_list = sorted(list(s2))

        a = ord(s1_list[0]) - ord(s2_list[0])
        for i in range(1, len(s1)):
            if a == 0 and s1_list[i] != s2_list[i]:
                a = ord(s1_list[i]) - ord(s2_list[i])
            elif a != 0 and (ord(s1_list[i]) - ord(s2_list[i])) * a < 0:
                return False

        return True

    def maxPower(self, s: str) -> int:
        """
        5396. 连续字符
        :param s:
        :return:
        """
        result = 0
        length = 0
        ch = ''
        for i in s:
            if i != ch:
                result = max(result, length)
                ch = i
                length = 1
            else:
                length += 1

        return max(result, length)

    def arrangeWords(self, text: str) -> str:
        """
        5413. 重新排列句子中的单词
        :param text:
        :return:
        """
        word_list = text.split(' ')
        word_list.sort(key=lambda x: len(x))
        for i in range(len(word_list)):
            if i == 0:
                word_list[0] = word_list[0][0].upper() + word_list[0][1:]
            else:
                word_list[i] = word_list[i].lower()
        return ' '.join(word_list)

    def validPalindromeII(self, s: str) -> bool:
        """
        680. 验证回文字符串 Ⅱ
        :see https://leetcode-cn.com/problems/valid-palindrome-ii/
        """

        def validPalindrome(string: str) -> bool:
            i, j = 0, len(string) - 1
            while i < j:
                if string[i] == string[j]:
                    i += 1
                    j -= 1
                else:
                    return False
            return True

        left, right = 0, len(s) - 1
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                # 判断回文串的方法：s == s[::-1]
                return validPalindrome(s[left:right]) or validPalindrome(s[left + 1:right + 1])
        return True

    def isPrefixOfWord(self, sentence: str, searchWord: str) -> int:
        """
        5416. 检查单词是否为句中其他单词的前缀
        :param sentence:
        :param searchWord:
        :return:
        """
        word_list = sentence.split(' ')
        for i, v in enumerate(word_list):
            if v.startswith(searchWord):
                return i + 1
        return -1

    def maxVowels(self, s: str, k: int) -> int:
        """
        5417. 定长子串中元音的最大数目
        :param s:
        :param k:
        :return:
        """
        ch_set = {'a', 'e', 'i', 'o', 'u'}

        count = 0
        for ch in s[:k]:
            if ch in ch_set:
                count += 1

        max_count = count
        for i in range(1, len(s) - k + 1):
            if s[i - 1] in ch_set:
                count -= 1
            if s[i + k - 1] in ch_set:
                count += 1
            max_count = max(max_count, count)

        return max_count

    def findTheLongestSubstring(self, s: str) -> int:
        """
        1371. 每个元音包含偶数次的最长子字符串
        :see https://leetcode-cn.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/
        """
        # 使用字典优化 if...else
        char_dict = {'a': 1 << 4, 'e': 1 << 3, 'i': 1 << 2, 'o': 1 << 1, 'u': 1 << 0}
        # 前缀和
        char_count = 0
        # 最早出现过前缀和的下标
        index_list = [-1] * 32  # 32 = 1 << 5
        # 最长长度
        max_length = 0

        for i, v in enumerate(s):
            char_count ^= char_dict.get(v, 0)

            if char_count > 0 > index_list[char_count]:
                index_list[char_count] = i
            else:
                max_length = max(max_length, i - index_list[char_count])

        return max_length

    def decodeString(self, s: str) -> str:
        """
        394. 字符串解码
        :see https://leetcode-cn.com/problems/decode-string/
        """
        num_stack = []
        str_stack = []

        result = ''

        for i, v in enumerate(s):
            if v.isdigit():
                if i > 0 and s[i - 1].isdigit():
                    num_stack[-1] = num_stack[-1] * 10 + int(v)
                else:
                    num_stack.append(int(v))
            elif v == '[':
                str_stack.append('')
            elif v == ']':
                times = num_stack.pop()
                string = str_stack.pop()
                if num_stack:
                    if str_stack:
                        str_stack[-1] += times * string
                    else:
                        str_stack.append(times * string)
                else:
                    result += times * string
            elif num_stack:
                if str_stack:
                    str_stack[-1] += v
                else:
                    str_stack.append(v)
            else:
                result += v

        return result

    def addStrings(self, num1: str, num2: str) -> str:
        """
        415. 字符串相加
        :see https://leetcode-cn.com/problems/add-strings/
        """
        num1_list = list(num1)[::-1]
        num2_list = list(num2)[::-1]
        result_list = []
        carry = 0

        for i in range(max(len(num1_list), len(num2_list))):
            a = num1_list[i] if i < len(num1_list) else 0
            b = num2_list[i] if i < len(num2_list) else 0
            result = int(a) + int(b) + carry
            carry = result // 10
            result_list.append(result % 10)

        if carry > 0:
            result_list.append(carry)

        return ''.join([str(i) for i in result_list[::-1]])

    def convert(self, s: str, numRows: int) -> str:
        """
        6. Z 字形变换
        :see https://leetcode-cn.com/problems/zigzag-conversion/
        """
        if numRows < 2:
            return s

        result = ['' for _ in range(numRows)]
        i = 0
        is_add = True

        for ch in s:
            result[i] += ch

            if i == 0:
                is_add = True
            elif i == numRows - 1:
                is_add = False

            if is_add:
                i += 1
            else:
                i -= 1

        return ''.join(result)

    def hasAllCodes(self, s: str, k: int) -> bool:
        """
        5409. 检查一个字符串是否包含所有长度为 K 的二进制子串
        :param s:
        :param k:
        :return:
        """
        num_set = set()
        for i in range(len(s) - k + 1):
            num_set.add(s[i:i + k])
        return len(num_set) == 1 << k

    def lengthOfLastWord(self, s: str) -> int:
        """
        58. 最后一个单词的长度
        :see https://leetcode-cn.com/problems/length-of-last-word/
        """
        s = s.rstrip()
        i = 0
        length = len(s)
        while i < length:
            if s[length - i - 1] == ' ':
                break
            i += 1
        return i

    def addBinary(self, a: str, b: str) -> str:
        """
        67. 二进制求和
        :see https://leetcode-cn.com/problems/add-binary/
        """
        carry = 0
        a = a[::-1]
        b = b[::-1]
        result = []
        for i in range(max(len(a), len(b))):
            s = carry + (int(a[i]) if i < len(a) else 0) + (int(b[i]) if i < len(b) else 0)
            carry = s // 2
            result.append(str(s % 2))
        if carry > 0:
            result.append(str(carry))
        return ''.join(result[::-1])

    def simplifyPath(self, path: str) -> str:
        """
        71. 简化路径
        :see https://leetcode-cn.com/problems/simplify-path/
        """
        path_stack = []
        for i in path.split('/'):
            if i not in {'', '.', '..'}:
                path_stack.append(i)
            elif i == '..' and path_stack:
                path_stack.pop()
        return f"/{'/'.join(path_stack)}"


if __name__ == "__main__":
    print(Solution().simplifyPath('//'))
