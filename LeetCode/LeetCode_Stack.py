import bisect
import heapq


class MedianFinder(object):
    """
    数据流的中位数
    :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/266/heap-stack-queue/1155/
    """

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.num_list = []

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        bisect.insort_left(self.num_list, num)

    def findMedian(self):
        """
        :rtype: float
        """
        length = len(self.num_list)
        if (length & 1) == 0:
            # 长度为偶数
            return (self.num_list[length // 2 - 1] + self.num_list[length // 2]) / 2.0
        else:
            # 长度为奇数
            return self.num_list[length // 2]


class Solution(object):
    def findKthLargest(self, nums, k):
        """
        215. 数组中的第K个最大元素
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/266/heap-stack-queue/1154/
        """
        # 使用最小堆获得第k大的元素
        queue = []

        for i in nums:
            if len(queue) < k:
                heapq.heappush(queue, i)
            elif i > queue[0]:
                heapq.heapreplace(queue, i)

        return queue[0]

    def calculate(self, s) -> int:
        """
        基本计算器II
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/266/heap-stack-queue/1159/
        """
        # 将中序表达式转为后缀表达式的过程中，计算结果
        symbol_value_dict = {'+': 0, '-': 0, '*': 1, '/': 1}
        calculate = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b
        }

        char_stack = []
        symbol_stack = []

        index = 0
        while index < len(s):
            if s[index] == ' ':
                index += 1
            elif s[index].isdigit():
                length = 1
                while index + length < len(s) and s[index + length].isdigit():
                    length += 1
                char_stack.append(int(s[index:index + length]))

                index += length
            else:
                while len(symbol_stack) > 0 and symbol_value_dict[symbol_stack[-1]] >= symbol_value_dict[s[index]]:
                    a = char_stack.pop()
                    b = char_stack.pop()
                    char_stack.append(calculate[symbol_stack.pop()](b, a))
                symbol_stack.append(s[index])

                index += 1

        while len(symbol_stack) > 0:
            a = char_stack.pop()
            b = char_stack.pop()
            char_stack.append(calculate[symbol_stack.pop()](b, a))

        return char_stack[0]

    def evalRPN(self, tokens: list) -> int:
        """
        逆波兰表达式求值
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/266/heap-stack-queue/1161/
        """
        # 遇到算符，就从栈中取出前两个数字，将运算结果存入栈中
        calculate = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: int(a / b)
        }

        num_stack = []

        for i in tokens:
            if i in calculate:
                # 计算符号
                a = num_stack.pop()
                b = num_stack.pop()
                num_stack.append(calculate[i](b, a))
            else:
                # 纯数字
                num_stack.append(float(i))

        return int(num_stack[0])

    def topKFrequent(self, nums: list, k: int) -> list:
        """
        前 K 个高频元素
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/266/heap-stack-queue/1157/
        """
        # 先使用字典储存出现次数，再使用最小堆获得前K个高频元素
        count_dict = {}
        for i in nums:
            if i in count_dict:
                count_dict[i] += 1
            else:
                count_dict[i] = 1

        return heapq.nlargest(k, count_dict.keys(), key=count_dict.get)


if __name__ == '__main__':
    # print(Solution().findKthLargest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))
    # print(Solution().calculate("1*2-3/4+5*6-7*8+9/10"))
    # print(Solution().calculate("100 * 120 - 130 / 10"))
    print(Solution().topKFrequent([6, 0, 1, 4, 9, 7, -3, 1, -4, -8, 4, -7, -3, 3, 2, -3, 9, 5, -4, 0], 6))
