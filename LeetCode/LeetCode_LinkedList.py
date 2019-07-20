from io import StringIO


class ListNode(object):
    """ 链表节点 """

    def __init__(self, x):
        self.val = x
        self.next = None

    def __eq__(self, other):
        return self.val == other.val


class LinkedList(object):
    """ 链表类 """

    # 头节点
    head: ListNode = None

    def __init__(self, data_list: list):
        """ 由数组初始化为链表 """
        self.head = ListNode(data_list[0])
        p = self.head
        for i in range(1, len(data_list)):
            p.next = ListNode(data_list[i])
            p = p.next

    def __str__(self) -> str:
        """ 自定义输出格式 """
        string = StringIO()

        p = self.head
        while p is not None:
            string.write(str(p.val) + " ")
            p = p.next

        return string.getvalue()

    def description(self, head) -> str:
        string = StringIO()

        p = head
        while p is not None:
            string.write(str(p.val) + " ")
            p = p.next

        return string.getvalue()

    def add_node_at_the_end(self, head: ListNode, value: int) -> ListNode:
        """ 在链表末尾添加节点 """
        if head is None:
            return LinkedList([value]).head

        p = head
        while p.next is not None:
            p = p.next
        p.next = ListNode(value)

        return head

    def deleteNode(self, head: ListNode, n: int) -> ListNode:
        """
        删除链表的倒数第N个节点
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/6/linked-list/42/
        """
        p = head
        q = head
        while p is not None:
            p = p.next
            if n >= 0:
                n -= 1
            else:
                q = q.next

        if n == 0:
            if q.next is None:
                return
            else:
                q.val = q.next.val
                q.next = q.next.next
        else:
            q.next = q.next.next if q.next is not None else None

        return head

    def reverseList(self, head: ListNode) -> ListNode:
        """
        翻转链表
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/6/linked-list/43/
        """
        if head is None or head.next is None:
            return head

        p = head
        q = head.next

        p.next = None
        while q.next is not None:
            t = q.next

            q.next = p

            p = q
            q = t

        q.next = p

        self.head = q

        return q

    def isPalindrome(self, head) -> bool:
        """
        回文链表
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/6/linked-list/45/
        """
        if head is None or head.next is None:
            return True

        # 通过快慢指针确定链表的中间位置和末尾位置
        slow_pointer = head
        fast_pointer = head

        while fast_pointer.next is not None and fast_pointer.next.next is not None:
            slow_pointer = slow_pointer.next
            fast_pointer = fast_pointer.next.next

        # 反转后半部分的链表节点
        p = slow_pointer
        q = slow_pointer.next

        while q.next is not None:
            t = q.next

            q.next = p
            p = q
            q = t

        q.next = p

        slow_pointer.next = None

        # 比较前半部分链表和后半部分链表的值
        p = head
        while p is not None and q is not None:
            if p.val == q.val:
                p = p.next
                q = q.next
            else:
                return False

        return True

    def quick_sort(self, head) -> ListNode:
        """ 链表的快速排序 """
        if head is None or head.next is None:
            return head

        middle_value_node = head

        # 初始化小于中值的左链表和大于中值的右链表，其中多创一个节点作为头结点
        left_linked_list = LinkedList([0])
        right_linked_list = LinkedList([0])

        # 遍历链表，将小于中值的值放入左链表，大于中值的值放入右链表
        p = head
        while p.next is not None:
            p = p.next
            if p.val <= middle_value_node.val:
                left_linked_list.add_node_at_the_end(left_linked_list.head, p.val)
            else:
                right_linked_list.add_node_at_the_end(right_linked_list.head, p.val)

        # 递归快速排序，要排除左链表和右链表头结点的值
        left_linked_list_header_node = self.quick_sort(left_linked_list.head.next)
        right_linked_list_header_node = self.quick_sort(right_linked_list.head.next)

        # 获得左链表的尾结点
        left_linked_list_tail_node = left_linked_list.head
        while left_linked_list_tail_node.next is not None:
            left_linked_list_tail_node = left_linked_list_tail_node.next

        # 拼接链表
        if left_linked_list_header_node is None:
            middle_value_node.next = right_linked_list_header_node
            return middle_value_node
        else:
            left_linked_list_tail_node.next = middle_value_node
            middle_value_node.next = right_linked_list_header_node
            return left_linked_list_header_node

    def hasCycle(self, head: ListNode) -> bool:
        """
        环形链表
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/6/linked-list/46/
        """
        if head is None:
            return False

        slow_point = head
        fast_point = head

        while fast_point.next is not None and fast_point.next.next is not None:
            fast_point = fast_point.next.next
            slow_point = slow_point.next
            if fast_point == slow_point:
                return True

        if fast_point.next is None or fast_point.next.next is None:
            return False


def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    """
    合并两个有序链表
    :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/6/linked-list/44/
    """
    p = l1
    q = l2

    if p is None:
        return q
    elif q is None:
        return p

    head = ListNode(p.val) if p.val <= q.val else ListNode(p.val)
    if p.val <= q.val:
        head = ListNode(p.val)
        p = p.next
    else:
        head = ListNode(q.val)
        q = q.next

    t = head
    while p is not None or q is not None:
        if p is None:
            t.next = q
            break
        elif q is None:
            t.next = p
            break
        elif p.val <= q.val:
            t.next = ListNode(p.val)
            p = p.next
        else:
            t.next = ListNode(q.val)
            q = q.next
        t = t.next

    return head


if __name__ == '__main__':
    data = [12, 1, 9, 4, 6, 8, 2, 3, 5]
    linked_list = LinkedList(data)

    p: ListNode = linked_list.quick_sort(linked_list.head)
    print(linked_list.description(p))
