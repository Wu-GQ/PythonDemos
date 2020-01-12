from queue import Queue


def is_palindrome(s: str) -> bool:
    """
    验证回文串
    :see https://leetcode-cn.com/explore/featured/card/top-interview-quesitons-in-2018/275/string/1136/
    """
    i = 0
    j = len(s) - 1

    while i <= j:
        if not s[i].isalnum():
            i += 1
            continue
        if not s[j].isalnum():
            j -= 1
            continue

        if s[i].upper() != s[j].upper():
            return False

        i += 1
        j -= 1

    return True


def partition(s: str):
    """
    验证回文串
    :see https://leetcode-cn.com/explore/featured/card/top-interview-quesitons-in-2018/275/string/1137/
    """
    length = len(s)
    # 将问题转换为输出从0节点 - (length - 1)节点的所有路径
    point_array = [[] for i in range(length)]

    for j in range(length):
        for i in range(length - j):
            if is_palindrome_in_string(s, i, j):
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


def is_palindrome_in_string(s, start_x, start_y) -> bool:
    """ 判断是否为回文串 """
    y = start_y
    for i in range(len(s) - start_x - start_y):
        if s[y + i] != s[len(s) - start_x - i - 1]:
            return False
        elif y + i >= len(s) - start_x - i - 1:
            return True
    return True


def min_cut(s: str) -> int:
    """
    分割回文串 II
    :see https://leetcode-cn.com/problems/palindrome-partitioning-ii/
    """
    length = len(s)
    # 将问题转换为输出从0节点 - (length - 1)节点的所有路径
    point_array = [[] for i in range(length + 1)]

    for j in range(length):
        for i in range(length - j):
            if is_palindrome_in_string(s, i, j):
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


def reverse_string(s: list) -> None:
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


def first_uniq_char(s: str) -> int:
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


def is_anagram(s: str, t: str) -> bool:
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


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.__word_set = set()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        self.__word_set.add(word)

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        return word in self.__word_set

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        for i in self.__word_set:
            if i.startswith(prefix):
                return True
        return False


def word_break(s: str, wordDict: list) -> bool:
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


def word_break_2(s: str, wordDict: list) -> list:
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


def reverse_num(x: int) -> int:
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


def longestCommonPrefix(strs: list) -> str:
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


def strstr(haystack: str, needle: str) -> int:
    """
    实现strStr()
    :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/5/strings/38/
    """
    # if len(needle) == 0:
    #     return 0
    return haystack.find(needle)


class Solution:
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
        compare_list = [[False for i in range(p_length)] for j in range(s_length)]

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


if __name__ == "__main__":
    print(Solution().isMatch2("aa", "a"))
