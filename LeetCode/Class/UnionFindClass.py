class UnionFindClass:

    def __init__(self, n):
        # 用来表示节点的父节点
        self.father = [i for i in range(n)]

    def get_root_count(self) -> int:
        """
        获得集合的数量
        :return:
        """
        father_set = set()
        for i in range(len(self.father)):
            father_set.add(self.find_root(i))
        return len(father_set)

    def find_root(self, node: int) -> int:
        """
        找到元素的根节点
        :param node: 被查找的元素
        :return: 被查找元素的根节点
        """
        if node == self.father[node]:
            # 元素所在树的根节点就是元素本身
            return node

        # 路径压缩，将元素的父节点设为所在树的根节点
        self.father[node] = self.find_root(self.father[node])
        return self.father[node]

    def merge(self, node1: int, node2: int) -> bool:
        """
        将两个元素所在的集合合并为一个集合
        :param node1: 第一个元素
        :param node2: 第二个元素
        :return: 合并前的两个元素是否在同一个集合
        """
        father1 = self.find_root(node1)
        father2 = self.find_root(node2)

        if father1 != father2:
            self.father[father1] = father2

        return father1 == father2

    def reset(self):
        """
        重置并查集
        :return:
        """
        for i in range(len(self.father)):
            self.father[i] = i
