class UnionFindSet:
    # 用来表示节点的父节点
    father = {}

    def find_root(self, node):
        """
        找到元素的父节点
        :param node: 被查找的元素
        :return: 被查找元素的父节点
        """
        if node not in self.father:
            self.father[node] = node
            return node

        if node == self.father[node]:
            # 元素所在树的根节点就是元素本身
            return node

        # 路径压缩，将元素的父节点设为所在树的根节点
        self.father[node] = self.find_root(self.father[node])
        return self.father[node]

    def merge(self, node1, node2):
        """
        将两个元素所在的集合合并为一个集合
        :param node1: 第一个元素
        :param node2: 第二个元素
        """
        a = self.find_root(node1)
        b = self.find_root(node2)
        if a != b:
            self.father[b] = a
        return a
