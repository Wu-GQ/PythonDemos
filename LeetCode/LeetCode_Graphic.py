import heapq


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: list) -> int:
        """
        单词接龙
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/273/graph-theory/1182/
        """
        # 使用广度优先遍历的思想，判断两个单词的最短路径
        # PS1: 直接使用广度优先遍历时，遇到较长的wordList非常耗时。因此，当wordList大于26时，使用直接替换字母然后搜索的方式加快速度
        # PS2: 使用双向的广度优先遍历，哪边的stack比较少，就遍历哪边，只有两边的stack拥有相同数据时，计算步数
        if endWord not in wordList or beginWord == endWord or len(beginWord) != len(endWord):
            return 0

        word_set: set = set(wordList)

        # 用来存储广度优先遍历时的数据
        stack: list = [beginWord]
        # 用来记录步数
        step_stack: list = [1]

        while len(stack) > 0:
            last_word = stack.pop(0)
            last_word_step = step_stack.pop(0)

            # 遍历搜索具有相同前缀和后缀的单词，加入的队列
            for i in range(0, len(last_word)):
                for char in "abcdefghijklmnopqrstuvwxyz":
                    word = "{0}{1}{2}".format(last_word[:i], char, last_word[i + 1:])
                    if word == endWord:
                        return last_word_step + 1
                    elif word in word_set:
                        stack.append(word)
                        step_stack.append(last_word_step + 1)
                        word_set.remove(word)

        return 0

    def canFinish(self, numCourses: int, prerequisites: list) -> bool:
        """
        课程表
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/273/graph-theory/1184/
        """
        # 拓扑排序：统计每个节点的入边数量和出边数量，逐个删除入边为0的节点及其出边，若入边为0的节点全部删除后，仍然存在节点，则存在环
        # 统计所有节点的入节点和出节点
        in_side_list = [set() for i in range(numCourses)]
        out_side_list = [[] for i in range(numCourses)]

        for prerequisite in prerequisites:
            in_side_list[prerequisite[1]].add(prerequisite[0])
            out_side_list[prerequisite[0]].append(prerequisite[1])

        delete_node_set = set()
        is_to_continue = True
        while is_to_continue:
            is_to_continue = False

            # 遍历所有入节点，删除没有入节点的节点
            for node in range(numCourses):
                if len(in_side_list[node]) == 0 and node not in delete_node_set:
                    is_to_continue = True

                    delete_node_set.add(node)

                    for i in out_side_list[node]:
                        in_side_list[i].remove(node)

        return len(delete_node_set) == numCourses

    def canFinish2(self, numCourses: int, prerequisites: list) -> bool:
        """
        课程表（kahn算法，效率高）
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/273/graph-theory/1184/
        """
        # 拓扑排序：删除所有入度为0的节点及其出边，然后逐个其入度为0的子节点
        # 统计所有节点的入节点和出节点
        in_side_count_list = [0 for i in range(numCourses)]
        out_side_list = [[] for i in range(numCourses)]

        for prerequisite in prerequisites:
            in_side_count_list[prerequisite[1]] += 1
            out_side_list[prerequisite[0]].append(prerequisite[1])

        # 存储无入边的节点
        zero_in_side_node_stack = []
        for node in range(numCourses):
            if in_side_count_list[node] == 0:
                zero_in_side_node_stack.append(node)

        # 拓扑排序的数组
        check_node_list = []

        # 逐个删除入度为0的节点，并将其入度为0的子节点加入栈中，待所有节点遍历之后，若发现仍有节点未被入栈，则存在环
        while len(zero_in_side_node_stack) != 0:
            check_node = zero_in_side_node_stack.pop()
            check_node_list.append(check_node)

            for sub_node in out_side_list[check_node]:
                in_side_count_list[sub_node] -= 1
                if in_side_count_list[sub_node] == 0:
                    zero_in_side_node_stack.append(sub_node)

        return len(check_node_list) == numCourses

    def findOrder(self, numCourses: int, prerequisites: list) -> list:
        """
        课程表II
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/273/graph-theory/1185/
        """
        in_side_count_list = [0 for i in range(numCourses)]
        out_side_list = [[] for i in range(numCourses)]

        for prerequisite in prerequisites:
            in_side_count_list[prerequisite[1]] += 1
            out_side_list[prerequisite[0]].append(prerequisite[1])

        # 存储无入边的节点
        zero_in_side_node_stack = []
        for node in range(numCourses):
            if in_side_count_list[node] == 0:
                zero_in_side_node_stack.append(node)

        # 拓扑排序的数组
        check_node_list = []

        # 逐个删除入度为0的节点，并将其入度为0的子节点加入栈中，待所有节点遍历之后，若发现仍有节点未被入栈，则存在环
        while len(zero_in_side_node_stack) != 0:
            check_node = zero_in_side_node_stack.pop()
            check_node_list.insert(0, check_node)

            for sub_node in out_side_list[check_node]:
                in_side_count_list[sub_node] -= 1
                if in_side_count_list[sub_node] == 0:
                    zero_in_side_node_stack.append(sub_node)

        return check_node_list if len(check_node_list) == numCourses else []

    def scheduleCourse(self, courses: list) -> int:
        """
        课程表III
        :see https://leetcode-cn.com/problems/course-schedule-iii/
        """
        # 贪心算法：将课程按照持续时长的从短到长和结束时间的先后顺序排列
        courses.sort(key=lambda x: (x[1], x[0]))

        # 已花费的时间
        cost_time = 0
        # 已经上过的课程
        cost_courses = []

        # 取时间最短且最早结束的课程
        for course in courses:
            if len(cost_courses) > 0:
                top = -cost_courses[0]
                if top > course[0] and cost_time + course[0] > course[1]:
                    heapq.heapreplace(cost_courses, -course[0])
                    cost_time = cost_time - top + course[0]
                elif cost_time + course[0] <= course[1]:
                    heapq.heappush(cost_courses, -course[0])
                    cost_time += course[0]
            else:
                heapq.heappush(cost_courses, -course[0])
                cost_time = course[0]

        return len(cost_courses)

    def sumOfDistancesInTree(self, N: int, edges: list) -> list:
        """
        树中距离之和
        :see https://leetcode-cn.com/problems/sum-of-distances-in-tree/
        """
        # 参考最小生成树算法——floyd算法（超时）
        distance_list = [[float('inf')] * N for i in range(0, N)]

        # 无向图
        for i in edges:
            distance_list[i[0]][i[1]] = 1
            distance_list[i[1]][i[0]] = 1

        # floyd算法核心，判断点i到点k+点k到点j的距离和点i到点j的距离
        for k in range(0, N):
            for i in range(0, N):
                for j in range(0, N):
                    if i == j:
                        distance_list[i][i] = 0
                    else:
                        distance_list[i][j] = min(distance_list[i][j], distance_list[i][k] + distance_list[k][j])

        return [sum(i) for i in distance_list]

    def orangesRotting(self, grid: list) -> int:
        """
        994. 腐烂的橘子
        :see https://leetcode-cn.com/problems/rotting-oranges/
        """

        def check_contiguous_oranges(row: int, column: int, times: int) -> list:
            contiguous_oranges_list = []

            if row - 1 >= 0 and grid[row - 1][column] == 1:
                contiguous_oranges_list.append((row - 1, column, times))
                grid[row - 1][column] = 3
            if row + 1 < len(grid) and grid[row + 1][column] == 1:
                contiguous_oranges_list.append((row + 1, column, times))
                grid[row + 1][column] = 3
            if column - 1 >= 0 and grid[row][column - 1] == 1:
                contiguous_oranges_list.append((row, column - 1, times))
                grid[row][column - 1] = 3
            if column + 1 < len(grid[row]) and grid[row][column + 1] == 1:
                contiguous_oranges_list.append((row, column + 1, times))
                grid[row][column + 1] = 3

            return contiguous_oranges_list

        if len(grid) < 1 or len(grid[0]) < 1:
            return 0

        # 最小分钟数
        times = 0
        # 每次广度遍历的位置
        orange_queue = []
        # 腐烂的总数量
        total_orange_count = 0

        for i in range(0, len(grid)):
            for j in range(0, len(grid[i])):
                if grid[i][j] == 2:
                    orange_queue.append((i, j, 0))
                    grid[i][j] = 3
                    total_orange_count += 1
                elif grid[i][j] == 1:
                    total_orange_count += 1

        while len(orange_queue) > 0:
            top_orange = orange_queue.pop(0)
            orange_queue += check_contiguous_oranges(top_orange[0], top_orange[1], top_orange[2] + 1)
            times = max(times, top_orange[2])
            total_orange_count -= 1

        return times if total_orange_count == 0 else -1

    def maxAreaOfIsland(self, grid: list) -> int:
        """
        695. 岛屿的最大面积
        :see https://leetcode-cn.com/problems/max-area-of-island/
        """

        def seek_next_island(x: int, y: int) -> list:
            """ 寻找该岛屿周围的岛屿坐标 """
            result = []
            if x - 1 >= 0 and grid[x - 1][y] == 1:
                result.append((x - 1, y))
            if x + 1 < len(grid) and grid[x + 1][y] == 1:
                result.append((x + 1, y))
            if y - 1 >= 0 and grid[x][y - 1] == 1:
                result.append((x, y - 1))
            if y + 1 < len(grid[x]) and grid[x][y + 1] == 1:
                result.append((x, y + 1))
            return result

        if len(grid) < 1 or len(grid[0]) < 1:
            return 0

        max_island_count = 0

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    # print('----------------')
                    island_queue = [(i, j)]
                    island_count = 0
                    while len(island_queue) > 0:
                        next_island = island_queue.pop(0)
                        if grid[next_island[0]][next_island[1]] == 1:
                            island_count += 1
                            grid[next_island[0]][next_island[1]] = 2
                            # print(next_island, island_count)
                            island_queue += seek_next_island(next_island[0], next_island[1])
                    max_island_count = max(max_island_count, island_count)

        return max_island_count

    def solve(self, board: list) -> None:
        """
        130. 被围绕的区域
        :see https://leetcode-cn.com/problems/surrounded-regions/
        """

        def seek_next_area(x: int, y: int) -> list:
            """ 寻找该岛屿周围的岛屿坐标 """
            result = []
            if x - 1 >= 0 and board[x - 1][y] == 'O':
                result.append((x - 1, y))
            if x + 1 < len(board) and board[x + 1][y] == 'O':
                result.append((x + 1, y))
            if y - 1 >= 0 and board[x][y - 1] == 'O':
                result.append((x, y - 1))
            if y + 1 < len(board[x]) and board[x][y + 1] == 'O':
                result.append((x, y + 1))
            return result

        if len(board) < 1 or len(board[0]) < 1:
            return

        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 'O':
                    checked_area = []
                    need_not_change = False
                    area_queue = [(i, j)]

                    while len(area_queue) > 0:
                        next_area = area_queue.pop(0)
                        if board[next_area[0]][next_area[1]] == 'O':
                            need_not_change = need_not_change or next_area[0] == 0 or next_area[0] == len(board) - 1 or next_area[1] == 0 or \
                                              next_area[1] == len(board[0]) - 1
                            checked_area.append(next_area)
                            board[next_area[0]][next_area[1]] = 'o'
                            area_queue += seek_next_area(next_area[0], next_area[1])

                    if not need_not_change:
                        for area in checked_area:
                            board[area[0]][area[1]] = 'X'

        for i in board:
            print(i)

        for i in range(len(board)):
            for j in range(len(board[i])):
                board[i][j] = board[i][j].upper()

        # print(board)


if __name__ == '__main__':
    # print(Solution().ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
    # print(Solution().scheduleCourse([[5, 15], [3, 19], [6, 7], [2, 10], [5, 16], [8, 14], [10, 11], [2, 19]]))
    # print(Solution().scheduleCourse([[5, 5], [4, 6], [2, 6]]))
    s = Solution()
    a = [["X", "X", "X", "O", "X", "O", "X"],
         ["X", "O", "X", "O", "X", "O", "O"],
         ["X", "X", "X", "X", "X", "X", "O"],
         ["X", "X", "X", "X", "O", "X", "O"],
         ["X", "X", "X", "X", "X", "X", "O"],
         ["X", "X", "X", "X", "X", "X", "X"],
         ["O", "X", "X", "O", "O", "O", "X"]]
    print(s.solve(a))
    print(a)
