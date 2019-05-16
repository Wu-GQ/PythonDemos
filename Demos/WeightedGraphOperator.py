# a - (10) → b - (20) → c - (30) → d
#            ↑          |
#           (1) - e  ← (1)
# 1. 求a到所有节点的最短路径
# 2. 再添加 b - (50) → e，求问题1
# 3. 在问题2的基础上，再添加 e - (-100) → d，求问题1
#
# MARK 1: 狄克斯特拉算法只适用于在有向无环的加权图中查找最短路径，而广度优先遍历适用于非加权图
# MARK 2: 狄克斯特拉算法要求加权图中所有边的权重都是正数，若含负权边，则需要使用贝尔曼-福德算法
# MARK 3: 狄克斯特拉算法的时间复杂度为O(顶点数 * 顶点数)
# MARK 4: 相较于狄克斯特拉算法，贝尔曼-福德算法支持存在负权重的情况，而且代码实现相对简单，但是时间复杂度较高，为O(顶点数 * 边数)


def find_lowest_cost_node(costs, processed):
    """
    查找最小开销的节点
    :param costs:       节点的开销字典
    :param processed:   已遍历的节点
    :return:            返回最小的开销节点
    """
    lowest_cost = float('inf')
    lowest_cost_node = None

    for node in costs:
        cost = costs[node]
        if cost < lowest_cost and node not in processed:
            lowest_cost = cost
            lowest_cost_node = node

    return lowest_cost_node


def dijkstra_algorithm_from_book(graph, start_point):
    costs = {start_point: 0}
    parents = {start_point: None}
    processed = []

    node = find_lowest_cost_node(costs, processed)
    while node is not None:
        cost = costs[node]
        for point in graph[node]:
            new_cost = cost + graph[node][point]
            if point not in costs or new_cost < costs[point]:
                costs[point] = new_cost
                parents[point] = node
        processed.append(node)
        node = find_lowest_cost_node(costs, processed)

    print(costs)
    print(parents)


def dijkstra_algorithm(graph, start_point, end_point) -> int:
    """
    使用狄克斯特拉算法求最短路径
    :param graph:           带正权图
    :param start_point:     开始点
    :param end_point:       结束点
    :return:                开始点到结束点的最小开销
    """

    # 从出发点达到每个点的最小开销
    lowest_cost_dict = {start_point: 0}

    # 每个节点的最小开销的父节点
    lowest_cost_parent_dict = {start_point: None}

    # 未遍历过的点
    unsearched_point_list = set(graph.keys())

    while len(unsearched_point_list) > 0:
        # 查找最小开销的且未遍历过的点
        # MARK: 此处可使用最小堆优化遍历速度？？？ - 因为节点的cost可能会改变，改变节点cost时，并不会自动调整最小堆，而且需要从最小堆中查找节点的开销等同于遍历的开销
        lowest_cost = float('inf')
        lowest_cost_point = None
        for point in unsearched_point_list:
            if point in lowest_cost_dict and lowest_cost_dict[point] < lowest_cost:
                lowest_cost = lowest_cost_dict[point]
                lowest_cost_point = point

        # 移除已遍历的节点
        unsearched_point_list.remove(lowest_cost_point)

        # 遍历该节点的出边
        for next_point in graph[lowest_cost_point]:
            cost = lowest_cost + graph[lowest_cost_point][next_point]
            if next_point not in lowest_cost_dict:
                lowest_cost_dict[next_point] = cost
                lowest_cost_parent_dict[next_point] = lowest_cost_point
            elif cost < lowest_cost_dict[next_point]:
                lowest_cost_dict[next_point] = cost
                lowest_cost_parent_dict[next_point] = lowest_cost_point

    print(lowest_cost_dict)
    print(lowest_cost_parent_dict)

    return lowest_cost_dict[end_point]


if __name__ == "__main__":
    path_weighted_graph_1 = {"a": {"b": 10},
                             "b": {"c": 20},
                             "c": {"d": 30, "e": 1},
                             "d": {},
                             "e": {"b": 1}}

    path_weighted_graph_2 = {"a": {"b": 10},
                             "b": {"c": 20, "e": 50},
                             "c": {"d": 30, "e": 1},
                             "d": {},
                             "e": {"b": 1}}

    path_weighted_graph_3 = {"a": {"b": 10},
                             "b": {"c": 20, "e": 50},
                             "c": {"d": 30, "e": 1},
                             "d": {},
                             "e": {"b": 1, "d": -100}}

    dijkstra_algorithm_from_book(path_weighted_graph_3, "a")
