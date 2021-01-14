from typing import List


class Solution:

    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        """
        1202. 交换字符串中的元素
        :see https://leetcode-cn.com/problems/smallest-string-with-swaps/
        """
        from LeetCode.Class.UnionFindClass import UnionFindClass
        union = UnionFindClass(len(s))
        for item in pairs:
            union.union(item[0], item[1])

        from collections import defaultdict
        dic = defaultdict(list)
        for i in range(len(s)):
            ch_code = ord(s[i]) - 97
            father = union.find_parent(i)
            if not dic[father]:
                dic[father] = [0] * 26
            dic[father][ch_code] += 1

        arr = []
        for i in range(len(s)):
            father = union.find_parent(i)
            for j in range(26):
                if dic[father][j] > 0:
                    dic[father][j] -= 1
                    arr.append(chr(j + 97))
                    break

        return ''.join(arr)

    def findLongestChain(self, pairs: List[List[int]]) -> int:
        """
        646. 最长数对链
        :see https://leetcode-cn.com/problems/maximum-length-of-pair-chain/
        """
        pairs.sort(key=lambda x: x[1])
        last = -float('inf')
        result = 0
        for i in pairs:
            if i[0] > last:
                last = i[1]
                result += 1
        return result


if __name__ == '__main__':
    s = Solution()
    print(s.findLongestChain([[0, 3], [1, 4], [2, 5], [1, 2]]))
