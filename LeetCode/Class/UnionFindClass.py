class UnionFindClass:

    def __init__(self, n):
        # 用来表示节点的父节点
        self.father = [i for i in range(n)]
        # 用来表示所在树的深度
        self.depth = [1] * n

    def find_parent(self, node: int):
        """
        找到元素的父节点
        :param node: 被查找的元素
        :return: 被查找元素的父节点
        """
        if node == self.father[node]:
            # 元素所在树的根节点就是元素本身
            return node

        # 路径压缩，将元素的父节点设为所在树的根节点
        self.father[node] = self.find_parent(self.father[node])
        return self.father[node]

    def union(self, node1: int, node2: int):
        """
        将两个元素所在的集合合并为一个集合
        :param node1: 第一个元素
        :param node2: 第二个元素
        """
        father1 = self.find_parent(node1)
        father2 = self.find_parent(node2)

        # 将深度较浅的树合并到深度较深的树
        if self.depth[father1] <= self.depth[father2]:
            self.father[father1] = father2
        else:
            self.father[father2] = father1

        # 如果两棵树的深度相同且根节点不同时，新的根节点的深度+1
        if self.depth[father1] == self.depth[father2] and father1 != father2:
            self.depth[father2] += 1
