# 图结构如下：
# a → b
# ↓     ↘
# c → d → e → f
#   ↘   ↗
#     g

import queue


def deep_first_search_by_recursion(graph, start_point):
    """
    递归实现的深度优先遍历
    :param graph:       被遍历的图
    :param start_point: 路径开始点
    :return:
    """

    # MARK: 需要有一个dict来存储已经遍历过的节点，以免再次遍历
    print(start_point)
    for point in graph[start_point]:
        deep_first_search_by_recursion(graph, point)


def deep_first_search_by_nonrecursion(graph, start_point):
    """
    非递归实现的深度优先遍历
    :param graph:       被遍历的图
    :param start_point: 路径开始点
    :return:
    """

    # 待遍历的节点，用list代替栈
    point_stack = [start_point]

    # 当前遍历的路线
    path_stack = queue.LifoQueue()
    path_stack.put(start_point)

    # 已遍历的节点
    searched_point_set = set()

    while len(point_stack) > 0:
        # 当待遍历的节点的所有子节点已被遍历后，才从栈内pop
        next_point = point_stack[len(point_stack) - 1]

        if next_point not in graph:
            continue

        if next_point not in searched_point_set:
            print(next_point, end=' ')
            searched_point_set.add(next_point)

        if len(graph[next_point]) > 0:
            point_stack.append(graph[next_point][0])
            graph[next_point].pop(0)
        else:
            point_stack.pop()


def breadth_fist_search(graph, start_point, end_point) -> list:
    """
    广度优先遍历，以查找最短路径，并输出最短路径
    :param graph:       被遍历的图
    :param start_point: 路径开始点
    :param end_point:   路径结束点
    :return:            第一条找到的最短路径
    """

    # 待遍历的点
    point_queue = queue.Queue().put(start_point)

    # 已遍历的点和从出发点到达该点需要的步数
    searched_point_dict = {start_point: 0}

    # 从哪些点可达该节点的字典
    last_point_dict = {}

    # 查找最短路径
    while not point_queue.empty():
        next_point = point_queue.get()
        if next_point == end_point:
            break

        for point in graph[next_point]:
            # 记录next_point节点可达point节点
            if point in last_point_dict:
                last_point_dict[point].append(next_point)
            else:
                last_point_dict[point] = [next_point]

            # 判断该点是否已经遍历
            if point in searched_point_dict:
                continue
            else:
                point_queue.put(point)
                # 保存到达该点需要的步数
                searched_point_dict[point] = searched_point_dict[next_point] + 1

    # 输出最短路径
    if end_point not in searched_point_dict:
        return []
    else:
        shortest_path = [end_point]

        step = searched_point_dict[end_point]
        point = end_point
        while step > 0:
            for last_point in last_point_dict[point]:
                if last_point in searched_point_dict \
                        and searched_point_dict[last_point] == step - 1:
                    shortest_path.append(last_point)
                    step -= 1
                    point = last_point
                    break

        return shortest_path


if __name__ == '__main__':
    search_gragh = {"a": ["b", "c"],
                    "b": ["e"],
                    "c": ["d", "g"],
                    "d": ["e"],
                    "e": ["f"],
                    "f": [],
                    "g": ["e"]}

    # print(breadth_fist_search(search_map, "a", "f"))
    # deep_first_search_by_recursion(search_gragh, "a")
    deep_first_search_by_nonrecursion(search_gragh, "a")
