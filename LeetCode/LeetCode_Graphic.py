import heapq
import string
from queue import PriorityQueue


class Node:
    def __init__(self, val=0, neighbors=[]):
        self.val = val
        self.neighbors = neighbors


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
                            need_not_change = need_not_change or next_area[0] == 0 or next_area[0] == len(board) - 1 or \
                                              next_area[1] == 0 or \
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

    def maxDistance(self, grid: list) -> int:
        """
        1162. 地图分析
        :see https://leetcode-cn.com/problems/as-far-from-land-as-possible/
        """
        """ 超时
        def max_distance_for_ocean(x: int, y: int) -> int:
            queue = [(x, y)]
            checked_ocean = set()
            _min_distance = float('inf')
            # print(f'----------- {x, y} -----------')
            while queue:
                next_ocean = queue.pop(0)
                distance = abs(next_ocean[0] - x) + abs(next_ocean[1] - y)

                # print(next_ocean)

                if grid[next_ocean[0]][next_ocean[1]] == 1:
                    return distance

                if distance >= _min_distance:
                    return _min_distance

                if next_ocean not in checked_ocean:
                    checked_ocean.add(next_ocean)

                    if next_ocean[0] - 1 >= 0:
                        queue.append((next_ocean[0] - 1, next_ocean[1]))
                        if (next_ocean[0] - 1, next_ocean[1]) in search_ocean_max_distance:
                            _min_distance = min(_min_distance, search_ocean_max_distance[(next_ocean[0] - 1, next_ocean[1])] + abs(next_ocean[0] - 1 -x) + abs(next_ocean[1] - y))
                    if next_ocean[1] - 1 >= 0:
                        queue.append((next_ocean[0], next_ocean[1] - 1))
                        if (next_ocean[0], next_ocean[1] - 1) in search_ocean_max_distance:
                            _min_distance = min(_min_distance, search_ocean_max_distance[(next_ocean[0], next_ocean[1] - 1)] + abs(next_ocean[0] - x) + abs(next_ocean[1] - 1 -y))
                    if next_ocean[0] + 1 < len(grid):
                        queue.append((next_ocean[0] + 1, next_ocean[1]))
                    if next_ocean[1] + 1 < len(grid[x]):
                        queue.append((next_ocean[0], next_ocean[1] + 1))

            return _min_distance

        search_ocean_max_distance = {}

        max_distance = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 0:
                    distance = max_distance_for_ocean(i, j)
                    max_distance = max(max_distance, distance)
                    search_ocean_max_distance[(i, j)] = distance
        return -1 if max_distance == 0 or max_distance == float('inf') else max_distance
        """
        # 多源广度遍历
        ocean_queue = []
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    ocean_queue.append((i, j, 0))

        search_ocean_set = set()
        min_distance = 0
        while ocean_queue:
            next_ocean = ocean_queue.pop(0)
            x, y, distance = next_ocean

            if (x, y) in search_ocean_set:
                continue

            search_ocean_set.add((x, y))
            min_distance = max(distance, min_distance)

            if x > 0:
                ocean_queue.append((x - 1, y, distance + 1))
            if x + 1 < len(grid):
                ocean_queue.append((x + 1, y, distance + 1))
            if y > 0:
                ocean_queue.append((x, y - 1, distance + 1))
            if y + 1 < len(grid[0]):
                ocean_queue.append((x, y + 1, distance + 1))

        # print(grid)

        return min_distance if min_distance != 0 else -1

    def gameOfLife(self, board: list) -> None:
        """
        289. 生命游戏
        :see https://leetcode-cn.com/problems/game-of-life/
        """

        def is_alive_for_cell(x: int, y: int, is_alive: int) -> bool:
            """ 确认某一位置的细胞是否存活 """
            alive_cell_count = 0
            for i, j in [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1), (x, y - 1), (x, y + 1), (x + 1, y - 1), (x + 1, y),
                         (x + 1, y + 1)]:
                if 0 <= i < len(another_board) and 0 <= j < len(another_board[x]):
                    alive_cell_count += another_board[i][j]

            if is_alive == 0:
                return alive_cell_count == 3
            elif is_alive == 1:
                return alive_cell_count == 2 or alive_cell_count == 3
            return False

        # 此处使用额外数组来进行计算。当然也可以通过拓展状态的方式，来解决这道题以降低空间复杂度，比如说2代表之前死的后面还是死的，3代表前死后活，4代表前活后死，5代表前活后活
        # ！！！大佬的解法：使用位运算，末位代表当前状态，前一位代表下一状态
        another_board = []
        for i in board:
            another_board.append(i.copy())

        for i in range(len(another_board)):
            for j in range(len(another_board[i])):
                board[i][j] = int(is_alive_for_cell(i, j, board[i][j]))

    def rotate(self, matrix: list) -> None:
        """
        面试题 01.07. 旋转矩阵
        :see https://leetcode-cn.com/problems/rotate-matrix-lcci/
        """
        if not matrix or not matrix[0]:
            return

        length = len(matrix)

        for i in range(0, length // 2 + 1):
            for j in range(0, length - 1 - 2 * i):
                a = i
                b = i + j
                c = length - 1 - i
                d = length - 1 - i - j
                matrix[a][b], matrix[b][c], matrix[c][d], matrix[d][a] = matrix[d][a], matrix[a][b], matrix[b][c], \
                                                                         matrix[c][d]

    def updateMatrix(self, matrix: list) -> list:
        """
        542. 01 矩阵
        :see https://leetcode-cn.com/problems/01-matrix/
        """
        # 多源广度优先遍历
        checked_index_set = set()
        zero_index_set = set()

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 0:
                    zero_index_set.add((i, j))
                    checked_index_set.add((i, j))

        distance = 0
        while zero_index_set:
            next_index_set = set()
            for i, j in zero_index_set:
                checked_index_set.add((i, j))
                matrix[i][j] = distance
                for x, y in [(i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j)]:
                    if 0 <= x < len(matrix) and 0 <= y < len(matrix[x]) and (x, y) not in checked_index_set and (
                            x, y) not in zero_index_set:
                        next_index_set.add((x, y))
            zero_index_set = next_index_set
            distance += 1
            # print(zero_index_set)

        return matrix

    def numWays(self, n: int, relation: list, k: int) -> int:
        """
        传递信息
        :param n:
        :param relation:
        :param k:
        :return:
        """
        next_person_list = [[] for _ in range(n)]
        for i in relation:
            next_person_list[i[0]].append(i[1])

        for i in next_person_list:
            print(i)

        result = [0] * n
        current_list = [0]
        while k > 0:
            next_list = []
            result = [0] * n
            for i in current_list:
                for person in next_person_list[i]:
                    next_list.append(person)
                    result[person] += 1
            current_list = next_list
            k -= 1

            # print(result)

        return result[-1]

    def numIslands(self, grid: list) -> int:
        """
        200. 岛屿数量
        :see https://leetcode-cn.com/problems/number-of-islands/
        """

        # 使用深度遍历标记小岛
        def island(x: int, y: int, current_count: int):
            grid[x][y] = current_count
            for i, j in [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]:
                if 0 <= i < len(grid) and 0 <= j < len(grid[i]) and grid[i][j] == '1':
                    island(i, j, current_count)

        # 初始化为2，以免与1和0冲突
        islands_count = 2

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == '1':
                    island(i, j, islands_count)
                    islands_count += 1
        # print(grid)
        return islands_count - 2

    def findDiagonalOrder(self, nums: list) -> list:
        """
        5394. 对角线遍历 II
        :param nums:
        :return:
        """
        num_list = []
        for i in range(len(nums)):
            for j in range(len(nums[i])):
                num_list.append((i + j, j, nums[i][j]))

        num_list.sort()
        return [i[2] for i in num_list]

    def destCity(self, paths: list) -> str:
        """
        5400. 旅行终点站
        :param paths:
        :return:
        """
        to_city = set()
        from_city = set()
        for path in paths:
            from_city.add(path[0])
            to_city.add(path[1])
        return to_city.difference(from_city).pop()

    def maximalSquare(self, matrix: list) -> int:
        """
        221. 最大正方形
        :see https://leetcode-cn.com/problems/maximal-square/
        """

        # 暴力法，也可以用动态规划解决

        def find_max_square_from_point(x: int, y: int) -> int:
            """ 返回从左上角开始查找，可以找到的最大的正方形 """
            index = 1
            while x + index < len(matrix) and y + index < len(matrix[x + index]):
                is_all_one = True
                # 检查 (x + index, y) 到 (x + index, y + index) 是否全不为零
                for i in range(y, y + index):
                    if matrix[x + index][i] == '0':
                        is_all_one = False
                        break

                # 检查 (x, y + index) 到 (x + index, y + index) 是否全不为零
                for i in range(x, x + index + 1):
                    if matrix[i][y + index] == '0':
                        is_all_one = False
                        break

                if is_all_one:
                    index += 1
                else:
                    break

            return index

        max_length = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] != '0':
                    max_length = max(max_length, find_max_square_from_point(i, j))

        for i in matrix:
            print(i)

        return max_length ** 2

    def minTime(self, n: int, edges: list, hasApple: list) -> int:
        """
        5406. 收集树上所有苹果的最少时间
        :param n:
        :param edges:
        :param hasApple:
        :return:
        """

        def sub_tree(root: int) -> int:
            """ 从节点 root 出发，到子节点上所有未检查过的苹果的最短距离 """
            # 从 root 出发的所有未检查节点
            next_node = set()
            if root in edge_dict:
                next_node = edge_dict[root].difference(checked_node_set)
            checked_node_set.update(next_node)

            # 递归判断子树里有没有苹果
            sub_distance = 0
            for i in next_node:
                x = sub_tree(i)
                if x > 0:
                    sub_distance += 2 + x
                elif hasApple[i]:
                    sub_distance += 2

            return sub_distance

        # 检查过的节点
        checked_node_set = set()

        # 邻接矩阵
        edge_dict = {}
        for i in edges:
            if i[0] not in edge_dict:
                edge_dict[i[0]] = set()
            edge_dict[i[0]].add(i[1])

        return sub_tree(0)

    def findOrder(self, numCourses: int, prerequisites: list) -> list:
        """
        210. 课程表 II
        :see https://leetcode-cn.com/problems/course-schedule-ii/
        """
        # 节点的出边字典
        out_dict = {key: [] for key in range(numCourses)}
        # 节点的入边字典
        in_dict = {key: [] for key in range(numCourses)}

        result = []

        # 转换邻接矩阵
        for i in prerequisites:
            # 要学习i[0], 要先学习i[1], 即i[1] -> i[0]
            in_dict[i[0]].append(i[1])
            out_dict[i[1]].append(i[0])

        # 拓扑排序
        while in_dict:
            # 找到一个入边数量为0的节点
            node = -1
            for i in in_dict:
                if len(in_dict[i]) == 0:
                    node = i
                    break

            if node == -1:
                result = []
                break

            # 从字典中去掉该节点
            for out_node in out_dict[node]:
                in_dict[out_node].remove(node)

            result.append(node)
            del in_dict[node]

        return result if len(result) == numCourses else []

    def checkIfPrerequisite(self, n: int, prerequisites: list, queries: list) -> list:
        """
        5410. 课程安排 IV
        :param n:
        :param prerequisites:
        :param queries:
        :return:
        """
        distance = [[False] * n for _ in range(n)]

        for p in prerequisites:
            distance[p[0]][p[1]] = True

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distance[i][j]:
                        continue
                    else:
                        distance[i][j] = distance[i][k] and distance[k][j]

        result = []
        for i in queries:
            result.append(distance[i[0]][i[1]])

        return result

    def minReorder(self, n: int, connections: list) -> int:
        """
        5426. 重新规划路线
        :param n:
        :param connections:
        :return:
        """
        in_dict = {}
        out_dict = {}
        for i in connections:
            if i[0] in out_dict:
                out_dict[i[0]].append(i[1])
            else:
                out_dict[i[0]] = [i[1]]

            if i[1] in in_dict:
                in_dict[i[1]].append(i[0])
            else:
                in_dict[i[1]] = [i[0]]

        node_stack = [0]
        result = 0
        checked_node = {0}
        while node_stack:
            node = node_stack.pop(0)
            out_list = out_dict.get(node, [])
            in_list = in_dict.get(node, [])

            for i in out_list:
                if i not in checked_node:
                    node_stack.append(i)
                    checked_node.add(i)
                    result += 1

            for i in in_list:
                if i not in checked_node:
                    node_stack.append(i)
                    checked_node.add(i)

        return result

    def findLadders(self, beginWord: str, endWord: str, wordList: list) -> list:
        """
        126. 单词接龙 II
        :see https://leetcode-cn.com/problems/word-ladder-ii/
        """
        # 1. 建立邻接表
        words_set = set(wordList)
        words_set.add(beginWord)

        word_length = len(beginWord)

        words_dict = {}
        for word in wordList:
            for i in range(word_length):
                for ch in string.ascii_lowercase:
                    new_word = f'{word[:i]}{ch}{word[i + 1:]}'
                    if new_word != word and new_word in words_set:
                        if word not in words_dict:
                            words_dict[word] = {new_word}
                        else:
                            words_dict[word].add(new_word)

                        if new_word not in words_dict:
                            words_dict[new_word] = {word}
                        else:
                            words_dict[new_word].add(word)

        if endWord not in words_dict:
            return []

        # 2. 广度优先遍历查找最短路径
        word_stack = [beginWord]
        word_time_dict = {}
        length = 1
        time = 0

        while word_stack:
            word = word_stack.pop(0)

            if word not in word_time_dict:
                word_time_dict[word] = time

                if word == endWord:
                    break

                for i in words_dict.get(word, {}):
                    word_stack.append(i)

            length -= 1
            if length == 0:
                length = len(word_stack)
                time += 1

        # 3. 回溯输出路径
        def backtrace(words: list):
            if words[0] == beginWord:
                result.append(words.copy())
                return

            if words[0] in words_dict:
                time = word_time_dict[words[0]]
                for i in words_dict[words[0]]:
                    if i in word_time_dict and word_time_dict[i] == time - 1:
                        words.insert(0, i)
                        backtrace(words)
                        words.pop(0)

        result = []
        backtrace([endWord])

        return result

    def minNumberOfSemesters(self, n: int, dependencies: list, k: int) -> int:
        """
        5435. 并行课程 II
        :param n:
        :param dependencies:
        :param k:
        :return:
        """
        in_dict = {}
        out_dict = {}
        for i in dependencies:
            out_node, in_node = i[0], i[1]
            if in_node in in_dict:
                in_dict[in_node].add(out_node)
            else:
                in_dict[in_node] = {out_node}

            if out_node in out_dict:
                out_dict[out_node].add(in_node)
            else:
                out_dict[out_node] = {in_node}

        out_node_list = PriorityQueue()
        for i in range(1, n + 1):
            if i in out_dict:
                out_node_list.put((n - len(out_dict[i]), i))
            else:
                out_node_list.put((n, i))

        # print(out_dict, in_dict)

        step = 0
        undeleted_nodes = []
        deleted_nodes = []
        while not out_node_list.empty():
            node = out_node_list.get()

            if node[1] in in_dict:
                undeleted_nodes.append(node)
            else:
                deleted_nodes.append(node[1])

            if len(deleted_nodes) == k or out_node_list.empty():
                index = 0
                step += 1
                # print(f'step: {step}, delete: {deleted_nodes}')

                for i in deleted_nodes:
                    if i in out_dict:
                        for j in out_dict[i]:
                            in_dict[j].remove(i)
                            if len(in_dict[j]) == 0:
                                del in_dict[j]
                        del out_dict[i]
                        index += 1
                deleted_nodes = []

                for i in undeleted_nodes:
                    out_node_list.put((i[0], i[1]))
                undeleted_nodes = []

        return step

    def maxProbability(self, n: int, edges: list, succProb: list, start: int, end: int) -> float:
        """
        5211. 概率最大的路径
        :see https://leetcode-cn.com/problems/path-with-maximum-probability/
        """

        from queue import PriorityQueue

        def dij() -> int:
            prob = [0] * n
            prob[start] = 1
            check = [0] * n

            q = PriorityQueue()
            q.put((-1, start))

            while not q.empty():
                p, node = q.get()
                p = -p

                if node == end:
                    return p
                if p < prob[end]:
                    continue
                check[node] = 1

                if node not in graphic:
                    continue

                for nn, pp in graphic[node]:
                    if check[nn] == 0 and pp * prob[node] > prob[nn]:
                        prob[nn] = pp * prob[node]
                        q.put((-prob[nn], nn))

            return prob[end]

        graphic = {}
        for i in range(len(edges)):
            x, y, p = edges[i][0], edges[i][1], succProb[i]
            if x in graphic:
                graphic[x].append((y, p))
            else:
                graphic[x] = [(y, p)]

            if y in graphic:
                graphic[y].append((x, p))
            else:
                graphic[y] = [(x, p)]

        return dij()

    def isBipartite(self, graph: list) -> bool:
        """
        785. 判断二分图
        :see https://leetcode-cn.com/problems/is-graph-bipartite/
        """
        a_set = set()
        b_set = set()

        for i in range(len(graph)):
            if not i:
                continue
            queue = [i]
            a_set.add(i)
            while queue:
                node = queue.pop(0)
                for j in graph[node]:
                    queue.append(j)
                    if node in a_set:
                        b_set.add(j)
                    else:
                        a_set.add(j)

                    if j in a_set and j in b_set:
                        return False

                graph[node] = []

        return True

    def countSubTrees(self, n: int, edges: list, labels: str) -> list:
        """
        5465. 子树中标签相同的节点数
        :param n:
        :param edges:
        :param labels:
        :return:
        """

        def dfs(index: int) -> dict:
            ch_dict = {labels[index]: 1}
            checked[index] = 1
            for i in out_dict.get(index, []):
                # ch_dict.update(dfs(i))
                if checked[i] == 1:
                    continue
                tmp = dfs(i)
                for j in tmp:
                    if j in ch_dict:
                        ch_dict[j] += tmp[j]
                    else:
                        ch_dict[j] = tmp[j]
            result[index] = ch_dict[labels[index]]
            print(index, ch_dict)
            return ch_dict

        out_dict = {}
        for a, b in edges:
            if a in out_dict:
                out_dict[a].append(b)
            else:
                out_dict[a] = [b]

            if b in out_dict:
                out_dict[b].append(a)
            else:
                out_dict[b] = [a]

        result = [0] * n
        checked = [0] * n
        dfs(0)
        return result

    def cloneGraph(self, node: Node) -> Node:
        """
        133. 克隆图
        :see https://leetcode-cn.com/problems/clone-graph/
        """

        def create_node(n: Node) -> Node:
            if not n:
                return n
            if n.val in node_dict:
                return node_dict[n.val]

            new_node = Node(n.val, [])
            node_dict[n.val] = new_node

            node_list = []
            for i in n.neighbors:
                node_list.append(create_node(i))
            new_node.neighbors = node_list

            return new_node

        node_dict = {}
        return create_node(node)

    def updateBoard(self, board: list, click: list) -> list:
        """
        529. 扫雷游戏
        :see https://leetcode-cn.com/problems/minesweeper/
        """
        # 确认点击的地方是否为地雷，是则直接结束
        click_x, click_y = click
        if board[click_x][click_y] == 'M':
            board[click_x][click_y] = 'X'
            return board

        # 用来存储某坐标附近的地雷数量
        num_board = [[0 for _ in i] for i in board]

        # 找到所有的地雷'M'，更新地雷周围一圈的地板数字
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 'M':
                    num_board[i][j] = -1
                    for x, y in [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]:
                        if 0 <= x < len(board) and 0 <= y < len(board[i]) and board[x][y] != 'M':
                            num_board[x][y] += 1

        # 若点击的地方为大于0的数字，则只改变这个点击位置的字母即可
        if num_board[click_x][click_y] > 0:
            board[click_x][click_y] = str(num_board[click_x][click_y])
            return board

        # 若点击位置的数字为0，则需要从这个坐标开始dfs/bfs，翻出附近所有的白块和第一次遍历遇见的数字
        checked = {(click_x, click_y)}
        queue = [(click_x, click_y)]
        while queue:
            node_x, node_y = queue.pop(0)
            if num_board[node_x][node_y] == 0:
                board[node_x][node_y] = 'B'
            elif num_board[node_x][node_y] > 0:
                board[node_x][node_y] = str(num_board[node_x][node_y])
                continue

            for i, j in [(node_x - 1, node_y - 1), (node_x - 1, node_y), (node_x - 1, node_y + 1), (node_x, node_y - 1), (node_x, node_y + 1),
                         (node_x + 1, node_y - 1), (node_x + 1, node_y), (node_x + 1, node_y + 1)]:
                if 0 <= i < len(board) and 0 <= j < len(board[i]) and (i, j) not in checked:
                    queue.append((i, j))
                    checked.add((i, j))

        return board

    def findSmallestSetOfVertices(self, n: int, edges: list) -> list:
        """
        5480. 可以到达所有点的最少点数目
        :param n:
        :param edges:
        :return:
        """
        nums = [0] * n
        for o, i in edges:
            nums[i] += 1
        result = []
        for i in range(n):
            if nums[i] == 0:
                result.append(i)
        return result

    def containsCycle(self, grid: list) -> bool:
        """
        5482. 二维网格图中探测环
        :param grid:
        :return:
        """

        def dfs_by_stack(a, b, ch: str) -> bool:
            stack = [(a, b, 0)]
            while stack:
                x, y, index = stack.pop()
                grid[x][y] = f'{grid[x][y]}{index}'
                for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= i < len(grid) and 0 <= j < len(grid[i]):
                        if grid[i][j] == ch:
                            # 如果只是字母且字母相同，则视为未走过的格子，并在该格子后加上编号
                            stack.append((i, j, index + 1))
                        elif grid[i][j][0] == ch:
                            # 如果格子不是纯字母且首字母相同，则说明这个格子之前走过，那么计算两个格子之间的步差
                            step = int(grid[i][j][1:])
                            if index - step >= 3:
                                return True
            return False

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j].isalpha() and dfs_by_stack(i, j, grid[i][j]):
                    return True

        for i in grid:
            print(i)

        return False


if __name__ == '__main__':
    s = Solution()
    print(s.containsCycle([["a", "a", "a", "a"], ["a", "b", "b", "a"], ["a", "b", "b", "a"], ["a", "a", "a", "a"]]))
