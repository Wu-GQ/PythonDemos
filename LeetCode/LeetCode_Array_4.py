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
            union.merge(item[0], item[1])

        from collections import defaultdict
        dic = defaultdict(list)
        for i in range(len(s)):
            ch_code = ord(s[i]) - 97
            father = union.find_root(i)
            if not dic[father]:
                dic[father] = [0] * 26
            dic[father][ch_code] += 1

        arr = []
        for i in range(len(s)):
            father = union.find_root(i)
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

    def hitBricks(self, grid: List[List[int]], hits: List[List[int]]) -> List[int]:
        """
        803. 打砖块
        :see https://leetcode-cn.com/problems/bricks-falling-when-hit/
        """
        # PS: 这波啊，逆向思考才是关键
        # 所有被敲打的位置
        hit_dict = {}
        for i in hits:
            hit_dict[(i[0], i[1])] = hit_dict.get((i[0], i[1]), 0) + 1

        # 并查集，每个节点的根节点
        father = {}
        # 每个集合中的节点数量
        count_dict = {}

        def find_root(node):
            """ 并查集，查找根节点 """
            if node not in father:
                return (-1, -1)

            if node != father[node]:
                # 路径压缩，将元素的父节点设为所在树的根节点
                father[node] = find_root(father[node])

            return father[node]

        def merge(node1, node2):
            """ 并查集，合并两个集合 """
            a = find_root(node1)
            b = find_root(node2)
            if a != b:
                # 合并时，优先将x==0的作为根节点
                if a[0] == 0:
                    father[b] = a
                    count_dict[a] += count_dict[b]
                    del count_dict[b]
                    return a
                else:
                    father[a] = b
                    count_dict[b] += count_dict[a]
                    del count_dict[a]
                    return b
            return a

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1 and (i, j) not in hit_dict:
                    # 添加新节点
                    father[(i, j)] = (i, j)
                    count_dict[(i, j)] = 1

                    # 检查上下左右是否存在方块
                    for (x, y) in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                        if 0 <= x < len(grid) and 0 <= y < len(grid[x]) and grid[x][y] == 1 and (x, y) not in hit_dict and (x, y) in father:
                            merge((i, j), (x, y))

        result = [0] * len(hits)
        for i in range(len(hits) - 1, -1, -1):
            x, y = hits[i][0], hits[i][1]

            # 如果这个地方存在砖块，或者这个地方已经被敲打过
            if grid[x][y] == 0 or hit_dict[(x, y)] > 1:
                hit_dict[(x, y)] -= 1
                continue

            # 获取上下左右的节点所在的集合的根节点
            up = find_root((x - 1, y)) if x >= 0 else (-1, -1)
            down = find_root((x + 1, y)) if x < len(grid) else (-1, -1)
            left = find_root((x, y - 1)) if y >= 0 else (-1, -1)
            right = find_root((x, y + 1)) if y < len(grid[0]) else (-1, -1)

            tmp = 0
            # 当其中有一个集合的x==0时，说明其它集合的根节点x!=0的集合全会掉落
            is_find = x == 0
            for (a, b) in {up, down, left, right}:
                if a == 0:
                    is_find = True
                elif a > 0:
                    tmp += count_dict[(a, b)]

            if is_find:
                result[i] = tmp

            # 将敲打的节点还原
            father[(x, y)] = (x, y)
            count_dict[(x, y)] = 1

            # 串联上下左右节点
            for (a, b) in [up, down, left, right]:
                if a >= 0:
                    merge((x, y), (a, b))

        return result

    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        """
        1584. 连接所有点的最小费用
        :see https://leetcode-cn.com/problems/min-cost-to-connect-all-points/
        """
        # 最小生成树 - Kruskal 算法
        if len(points) < 2:
            return 0

        from Class.UnionFindClass import UnionFindClass
        union = UnionFindClass(len(points))

        import heapq
        queue = []

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                heapq.heappush(queue, (abs(points[j][0] - points[i][0]) + abs(points[j][1] - points[i][1]), i, j))

        result = 0
        while queue:
            dis, a, b = heapq.heappop(queue)
            if not union.merge(a, b):
                result += dis

        return result

    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        """
        1128. 等价多米诺骨牌对的数量
        :see https://leetcode-cn.com/problems/number-of-equivalent-domino-pairs/
        """
        dic = {}
        for i in dominoes:
            dic[i[0], i[1]] = dic.get((i[0], i[1]), 0) + 1
        result = 0
        for (k1, k2) in dic:
            result += dic[k1, k2] * (dic[k1, k2] - 1) // 2 + (dic[k1, k2] * dic[k2, k1] if k1 < k2 and (k2, k1) in dic else 0)
        return result

    def countBalls(self, lowLimit: int, highLimit: int) -> int:
        """
        5654. 盒子中小球的最大数量
        :see https://leetcode-cn.com/problems/maximum-number-of-balls-in-a-box/
        """
        box = {}
        for i in range(lowLimit, highLimit + 1):
            res = 0
            n = i
            while n > 0:
                res += n % 10
                n //= 10
            box[res] = box.get(res, 0) + 1

        minI = 0
        for i in box:
            if box[i] > box.get(minI, 0):
                minI = i
        return box[minI]

    def restoreArray(self, adjacentPairs: List[List[int]]) -> List[int]:
        """
        5665. 从相邻元素对还原数组
        :see https://leetcode-cn.com/problems/restore-the-array-from-adjacent-pairs/
        """
        from collections import defaultdict
        pair = defaultdict(list)
        for i in adjacentPairs:
            pair[i[0]].append(i[1])
            pair[i[1]].append(i[0])

        start = 0
        for i in pair:
            if len(pair[i]) == 1:
                start = i
                break

        res = [start]

        def dfs(left: int):
            if not pair[left]:
                return
            nextI = pair[left][0]
            pair[nextI].remove(left)
            res.append(nextI)
            dfs(nextI)

        dfs(start)
        return res

    def canEat(self, candiesCount: List[int], queries: List[List[int]]) -> List[bool]:
        """
        5667. 你能在你最喜欢的那天吃到你最喜欢的糖果吗？
        :see https://leetcode-cn.com/problems/can-you-eat-your-favorite-candy-on-your-favorite-day/
        """
        prefix = [0]
        for i in candiesCount:
            prefix.append(prefix[-1] + i)

        res = []
        for i in range(len(queries)):
            c_type, day, limit = queries[i][0], queries[i][1], queries[i][2]

            # 至少一天吃一颗，最多一天吃limit颗
            res.append(day < prefix[c_type + 1] and prefix[c_type] < (day + 1) * limit)

        return res

    def numSimilarGroups(self, strs: List[str]) -> int:
        """
        839. 相似字符串组
        :see https://leetcode-cn.com/problems/similar-string-groups/
        """

        def check(a: str, b: str) -> bool:
            indexes = []
            for i in range(len(a)):
                if a[i] != b[i]:
                    indexes.append(i)
            return len(indexes) == 0 or len(indexes) == 2 and a[indexes[0]] == b[indexes[1]] and a[indexes[1]] == b[indexes[0]]

        pair = []
        for i in range(len(strs)):
            for j in range(i + 1, len(strs)):
                if check(strs[i], strs[j]):
                    pair.append((i, j))

        from Class.UnionFindClass import UnionFindClass
        union = UnionFindClass(len(strs))

        for i in pair:
            union.merge(i[0], i[1])

        return union.get_root_count()


if __name__ == '__main__':
    s = Solution()
    # print(s.canEat([7, 4, 5, 3, 8], [[0, 2, 2], [4, 2, 4], [2, 13, 1000000000]]))
    # print(s.canEat([5, 2, 6, 4, 1], [[3, 1, 2], [4, 10, 3], [3, 10, 100], [4, 100, 30], [1, 3, 1]]))
    # print(s.canEat([1, 8], [[1, 0, 1]]))
    print(s.numSimilarGroups(["omv", "ovm"]))
