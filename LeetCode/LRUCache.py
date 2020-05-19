from io import StringIO


class TwoWayLinkedListNode:

    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None


class TwoWayLinkedList:

    def __init__(self):
        self.head: TwoWayLinkedListNode = None
        self.tail: TwoWayLinkedListNode = None
        self.node_count: int = 0

    def description(self) -> str:
        """ 将链表转换为字符串输出 """
        string = StringIO()

        p = self.head
        while p is not None:
            string.write(f'({p.key}, {p.val}) ')
            p = p.next
            if p == self.head:
                break

        return string.getvalue()

    def add_node_at_the_end(self, key, value) -> TwoWayLinkedListNode:
        """ 在链表末尾添加节点 """
        node = TwoWayLinkedListNode(key, value)

        if self.head is None and self.tail is None:
            self.head = node
            self.tail = node

            node.next = node
            node.prev = node
        else:
            node.prev = self.tail
            node.next = self.head
            self.tail.next = node
            self.head.prev = node
            self.tail = node

        self.node_count += 1

        return node

    def add_node_at_first(self, key, value) -> TwoWayLinkedListNode:
        """ 在链表首部添加节点 """
        if self.head is None and self.tail is None:
            return self.add_node_at_the_end(key, value)
        else:
            node: TwoWayLinkedListNode = TwoWayLinkedListNode(key, value)
            self.head.prev = node
            self.tail.next = node
            node.next = self.head
            node.prev = self.tail
            self.head = node
            self.node_count += 1
            return node

    def add_node_at_first_with_node(self, node: TwoWayLinkedListNode):
        """ 在链表首部添加节点 """
        if self.node_count == 0:
            self.head = node
            self.tail = node
            node.prev = node
            node.next = node
        else:
            self.tail.next = node
            self.head.prev = node
            node.next = self.head
            node.prev = self.tail
            self.head = node

        self.node_count += 1

    def delete_end_node(self):
        """ 删除末尾节点 """
        if self.tail is None:
            return

        if self.tail != self.head:
            node = self.tail
            node.prev.next = self.head
            self.tail = node.prev
        else:
            self.head = None
            self.tail = None

        self.node_count -= 1

    def delete_node(self, node: TwoWayLinkedListNode):
        if self.node_count <= 1:
            self.head = None
            self.tail = None

            self.node_count = 0

            return

        node.prev.next = node.next
        node.next.prev = node.prev

        if node == self.head:
            self.head = node.next
        if node == self.tail:
            self.tail = node.prev

        self.node_count -= 1


class LRUCache:

    def __init__(self, capacity: int):
        # 最大容量
        self.max_num_count: int = capacity
        # 当前容量
        self.num_count: int = 0
        # 数字和节点字典
        self.num_node_dict: dict = {}
        # 使用先后顺序的双向链表
        self.two_way_linked_list: TwoWayLinkedList = TwoWayLinkedList()

    def get(self, key: int) -> int:
        if key in self.num_node_dict:
            if self.two_way_linked_list.node_count == 1:
                return self.num_node_dict[key].val

            node: TwoWayLinkedListNode = self.num_node_dict[key]
            self.two_way_linked_list.delete_node(node)
            self.two_way_linked_list.add_node_at_first_with_node(node)

            return self.num_node_dict[key].val
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.num_node_dict:
            self.two_way_linked_list.delete_node(self.num_node_dict[key])
            self.num_node_dict[key] = self.two_way_linked_list.add_node_at_first(key, value)
        elif self.num_count < self.max_num_count:
            self.num_node_dict[key] = self.two_way_linked_list.add_node_at_first(key, value)
            self.num_count += 1
        else:
            if self.two_way_linked_list.tail is not None:
                del self.num_node_dict[self.two_way_linked_list.tail.key]
            self.two_way_linked_list.delete_end_node()
            self.num_node_dict[key] = self.two_way_linked_list.add_node_at_first(key, value)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

if __name__ == '__main__':
    cache: LRUCache = LRUCache(capacity=2)
    cache.put(2, 1)
    cache.put(3, 2)
    print(cache.get(3))
    print(cache.get(2))
    cache.put(4, 3)
    print(cache.get(2))
    print(cache.get(3))
    print(cache.get(4))
