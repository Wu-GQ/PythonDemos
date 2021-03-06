from io import StringIO


class ListNode(object):
    """ 链表节点 """

    def __init__(self, x):
        self.val: int = x
        self.next: ListNode = None


class LinkedList(object):
    """ 链表类 """

    # 头节点
    head: ListNode = None

    def __init__(self, data_list: list):
        """ 由数组初始化为链表 """
        self.head = ListNode(data_list[0] if len(data_list) > 0 else 0)
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
        234. 回文链表
        :see https://leetcode-cn.com/problems/palindrome-linked-list/
        """

        def compare_linked_list(a: ListNode, b: ListNode) -> bool:
            """ 比较两个链表的值是否相同 """
            while a and b:
                if a.val != b.val:
                    return False
                a = a.next
                b = b.next
            return a is None and b is None

        if not head or not head.next:
            return True

        # 找到中间节点，同时反转前半段链表
        slow_before, slow, fast = None, head, head
        while fast and fast.next:
            fast = fast.next.next

            # 反转前半段链表
            slow_next = slow.next
            slow.next = slow_before
            slow_before, slow = slow, slow_next

        print(fast is not None, slow_before.val, slow.val)

        # 当fast非空，说明链表为奇数长度，否则为偶数长度
        if fast:
            return compare_linked_list(slow_before, slow.next)
        else:
            return compare_linked_list(slow_before, slow)

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

    def merge_sort_for_linked_list(self, head: ListNode) -> ListNode:
        """
        排序链表（归并排序）
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/265/linked-list/1147/
        """

        def merge_for_part_linked_list(head1: ListNode, head2: ListNode) -> ListNode:
            """ 归并排序 - 并 """
            sorted_linked_list_head = ListNode(0)
            sorted_linked_list_end = sorted_linked_list_head

            p = head1
            q = head2

            while p is not None and q is not None:
                if p.val < q.val:
                    sorted_linked_list_end.next = p
                    p = p.next
                    sorted_linked_list_end = sorted_linked_list_end.next
                else:
                    sorted_linked_list_end.next = q
                    q = q.next
                    sorted_linked_list_end = sorted_linked_list_end.next

            if p is not None:
                sorted_linked_list_end.next = p
            if q is not None:
                sorted_linked_list_end.next = q

            return sorted_linked_list_head.next

        if head is None or head.next is None:
            return head

        # 通过快慢指针，获得链表中间的位置
        slow_ptr = head
        fast_ptr = head

        while fast_ptr.next is not None and fast_ptr.next.next is not None:
            slow_ptr = slow_ptr.next
            fast_ptr = fast_ptr.next.next

        # 分割前后链表
        p = slow_ptr.next
        slow_ptr.next = None

        # 递归执行归并排序
        return merge_for_part_linked_list(self.merge_sort_for_linked_list(head), self.merge_sort_for_linked_list(p))

    def sortList(self, head: ListNode) -> ListNode:
        """
        排序链表
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/265/linked-list/1147/
        """

        def quick_sort_for_linked_list(head_node: ListNode) -> list:
            # 使用快速排序，对链表进行排序（时间超限，是因为第一个元素是1，后面元素随机1、2、3，导致第一次的快排无效）
            if head_node is None or head_node.next is None:
                return [head_node, head_node]

            # 中值
            middle_value_node = head_node

            # 比中值小的链表
            smaller_numbers_linked_list_head = ListNode(-1)
            smaller_numbers_linked_list_end = smaller_numbers_linked_list_head
            smaller_numbers_linked_list_end.next = None
            # 比中值大的链表
            bigger_numbers_linked_list_head = ListNode(1)
            bigger_numbers_linked_list_end = bigger_numbers_linked_list_head
            bigger_numbers_linked_list_end.next = None

            # 遍历链表，将小于中值的值放入小链表，大于中值的值放入大链表
            p = head_node.next
            while p is not None:
                q = p.next
                if p.val < middle_value_node.val:
                    smaller_numbers_linked_list_end.next = p
                    smaller_numbers_linked_list_end = p
                    if smaller_numbers_linked_list_end is not None:
                        smaller_numbers_linked_list_end.next = None
                else:
                    bigger_numbers_linked_list_end.next = p
                    bigger_numbers_linked_list_end = p
                    if bigger_numbers_linked_list_end is not None:
                        bigger_numbers_linked_list_end.next = None
                p = q

            # 对小链表和大链表执行快速排序
            recursive_smaller_linked_list = quick_sort_for_linked_list(smaller_numbers_linked_list_head.next)
            recursive_bigger_linked_list = quick_sort_for_linked_list(bigger_numbers_linked_list_head.next)

            # 拼接小链表和大链表
            if recursive_bigger_linked_list[1] is None:
                recursive_smaller_linked_list[1].next = middle_value_node
                middle_value_node.next = None
                return [recursive_smaller_linked_list[0], middle_value_node]
            elif recursive_smaller_linked_list[1] is None:
                middle_value_node.next = recursive_bigger_linked_list[0]
                return [middle_value_node, recursive_bigger_linked_list[1]]
            else:
                recursive_smaller_linked_list[1].next = middle_value_node
                middle_value_node.next = recursive_bigger_linked_list[0]
                return [recursive_smaller_linked_list[0], recursive_bigger_linked_list[1]]

        return quick_sort_for_linked_list(head)[0]

    def oddEvenList(self, head: ListNode) -> ListNode:
        """
        奇偶链表
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/265/linked-list/1152/
        """
        if head is None:
            return head

        # 奇数链表的头结点和尾结点
        odd_numbers_linked_list_head = head
        odd_numbers_linked_list_end = odd_numbers_linked_list_head
        # 偶数链表的头结点和尾结点
        even_numbers_linked_list_head = head.next
        even_numbers_linked_list_end = even_numbers_linked_list_head

        p = head.next

        index = 1
        while p is not None:
            q = p.next

            if index & 1 == 0:
                odd_numbers_linked_list_end.next = p
                odd_numbers_linked_list_end = odd_numbers_linked_list_end.next
                odd_numbers_linked_list_end.next = None
            else:
                even_numbers_linked_list_end.next = p
                even_numbers_linked_list_end = even_numbers_linked_list_end.next
                even_numbers_linked_list_end.next = None

            index += 1
            p = q

        # 拼接奇偶链表
        odd_numbers_linked_list_end.next = even_numbers_linked_list_head

        return odd_numbers_linked_list_head

    def whole_linked_list_description(self, head: ListNode) -> str:
        p = head
        s: str = ''
        while p is not None:
            s = f'{s} {p.val} '
            p = p.next

        return s

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        """
        21. 合并两个有序链表
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

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        """
        两数相加
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-medium/31/linked-list/82/
        """
        if l1 is None and l2 is None:
            return ListNode(0)

        result_linked_list_head = ListNode(0)
        r = result_linked_list_head

        last_result = 0
        p = l1
        q = l2
        while p is not None or q is not None:
            result = last_result + (p.val if p is not None else 0) + (q.val if q is not None else 0)
            last_result = result // 10
            r.next = ListNode(result % 10)

            if p is not None:
                p = p.next
            if q is not None:
                q = q.next
            r = r.next

        if last_result > 0:
            r.next = ListNode(last_result % 10)

        return result_linked_list_head.next

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        """
        相交链表
        :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-medium/31/linked-list/84/
        """
        if headA is None or headB is None:
            return None

        # 两个链表的长度差值
        length_difference = 0

        p = headA
        q = headB

        # 先让一个链表走到末尾
        while p is not None and q is not None:
            p = p.next
            q = q.next

        # 统计两个链表的长度差值
        if p is None:
            while q is not None:
                q = q.next
                length_difference += 1
        elif q is None:
            while p is not None:
                p = p.next
                length_difference -= 1

        # 差值大于0表示B链表比A链表长，否则A链表比B链表长
        p = headA
        q = headB

        # 将两个指针移到距离链表结尾相同长度的位置
        if length_difference > 0:
            while length_difference > 0:
                q = q.next
                length_difference -= 1
        else:
            while length_difference < 0:
                p = p.next
                length_difference += 1

        # 同时移动两个指针，若两个指针地址相同，则是相交的起点
        while p is not None and q is not None and p != q:
            p = p.next
            q = q.next

        return p

    def reverseList(self, head: ListNode) -> ListNode:
        """
        反转链表
        :see https://leetcode-cn.com/problems/reverse-linked-list/
        """
        """ 
        # 迭代，利用栈的特性，先进后出实现链表反转 
        if head is None:
            return head

        node_stack = []

        p = head
        while p is not None:
            node_stack.append(p)
            p = p.next

        head = node_stack[-1]
        while len(node_stack) > 0:
            p = node_stack.pop()
            p.next = node_stack[-1] if len(node_stack) > 0 else None

        return head
        """
        # 递归
        if head is None or head.next is None:
            return head

        def aaa(head: ListNode) -> ListNode:
            if head is None or head.next is None:
                return head

            q = head.next
            p = aaa(q)
            q.next = head

            return p

        p = aaa(head)
        head.next = None

        return p

    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        """
        25. K 个一组翻转链表
        :see https://leetcode-cn.com/problems/reverse-nodes-in-k-group/
        """

        def reverse_linked_list(head: ListNode) -> (ListNode, ListNode, ListNode):
            """
            反转链表
            :param head: 链表头结点
            :return: 反转后的链表头结点，反转后的链表的尾结点，下一次开始翻转的头结点
            """
            if head is None:
                return head

            node_stack = []

            # 将全部节点入栈
            p = head
            node_index = k
            while p is not None and node_index > 0:
                node_stack.append(p)
                p = p.next
                node_index -= 1

            # 当node_index大于0时，说明剩下的链表不足以翻转
            if node_index > 0:
                return head, None, None

            # 保存下一次开始翻转的头结点
            next_head_node = p

            # 翻转链表
            new_head = node_stack[-1]
            while len(node_stack) > 0:
                p = node_stack.pop()
                p.next = node_stack[-1] if len(node_stack) > 0 else None

            return new_head, head, next_head_node

        if head is None:
            return head

        # 对第一组翻转链表特殊处理
        head_node, end_node, next_node = reverse_linked_list(head)

        # 连接每一组翻转后的链表
        while next_node is not None:
            new_head_node, new_end_node, next_node = reverse_linked_list(next_node)
            end_node.next = new_head_node
            end_node = new_end_node

        return head_node

    def swapPairs(self, head: ListNode) -> ListNode:
        """
        24. 两两交换链表中的节点
        :see https://leetcode-cn.com/problems/swap-nodes-in-pairs/
        """
        if head is None or head.next is None:
            return head

        q = head.next
        t = q.next

        q.next = head
        head.next = self.swapPairs(t)

        return q

    def middleNode(self, head: ListNode) -> ListNode:
        """
        876. 链表的中间结点
        :see https://leetcode-cn.com/problems/middle-of-the-linked-list/
        """
        if head is None or head.next is None:
            return head

        slow_node, quick_node = head, head

        while quick_node and quick_node.next:
            slow_node, quick_node = slow_node.next, quick_node.next.next

        return slow_node

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        """
        445. 两数相加 II
        :see https://leetcode-cn.com/problems/add-two-numbers-ii/
        """
        # 使用栈来存储所有节点
        l1_stack = []
        l2_stack = []

        p = l1
        while p is not None:
            l1_stack.append(p)
            p = p.next
        p = l2
        while p is not None:
            l2_stack.append(p)
            p = p.next

        # 进位
        carry = 0
        # 入栈后，从末位开始取即可
        while l1_stack or l2_stack or carry > 0:
            # 当 l1 的节点被取完时，需要构建新的0节点，并把 l1 的头指针往前移
            if l1_stack:
                p = l1_stack.pop()
            else:
                p = ListNode(0)
                p.next = l1
                l1 = p

            q_val = l2_stack.pop().val if l2_stack else 0
            result = p.val + q_val + carry

            # 计算当前位
            p.val = result % 10
            # 进位
            carry = result // 10

        return l1

    def mergeKLists(self, lists: list) -> ListNode:
        """
        23. 合并K个排序链表
        :see https://leetcode-cn.com/problems/merge-k-sorted-lists/
        """
        if not lists:
            return None

        head_node_list = [(node.val, node) for node in lists if node]
        head_node_list.sort(key=lambda x: x[0])

        head = ListNode(0)
        p = head
        while head_node_list:
            value, node = head_node_list.pop(0)
            if node.next:
                index = 0
                while index < len(head_node_list):
                    if head_node_list[index][0] < node.next.val:
                        index += 1
                    else:
                        break
                head_node_list.insert(index, (node.next.val, node.next))

            p.next = node
            p = node

        return head.next

    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        """
        61. 旋转链表
        :see https://leetcode-cn.com/problems/rotate-list/
        """
        if not head or not head.next:
            return head

        # 确认链表长度，计算需要移动几步
        p = head
        length = 1
        while p.next:
            p = p.next
            length += 1

        k %= length
        if k == 0:
            return head

        # 双指针法，找到需要移到链表顶部的开始节点的前一节点
        p = head
        q = head
        while k > 0:
            p = p.next
            k -= 1
        while p.next:
            q = q.next
            p = p.next

        # 将开始节点及其之后的节点移到链表顶部
        p.next = head
        head = q.next
        q.next = None

        return head

    def removeDuplicateNodes(self, head: ListNode) -> ListNode:
        """
        面试题 02.01. 移除重复节点
        :see https://leetcode-cn.com/problems/remove-duplicate-node-lcci/
        """
        if not head:
            return head

        num = [False] * 20001

        p = head
        num[p.val] = True

        while p.next:
            if num[p.next.val]:
                p.next = p.next.next
            else:
                p = p.next
                num[p.val] = True

        return head

    def reorderList(self, head: ListNode) -> None:
        """
        143. 重排链表
        :see https://leetcode-cn.com/problems/reorder-list/
        """
        if not head or not head.next:
            return

        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        reverse = slow.next
        slow.next = None
        stack = []
        while reverse:
            stack.append(reverse)
            reverse = reverse.next

        slow, fast = head, head.next
        while stack and slow:
            node = stack.pop()
            slow.next = node
            node.next = fast
            slow, fast = fast, fast.next if fast else None

    def swapNodes(self, head: ListNode, k: int) -> ListNode:
        """
        5652. 交换链表中的节点
        :see https://leetcode-cn.com/problems/swapping-nodes-in-a-linked-list/
        """
        fast = head
        for _ in range(1, k):
            fast = fast.next
        left = fast

        fast = fast.next

        slow = head
        while fast:
            fast = fast.next
            slow = slow.next

        left.val, slow.val = slow.val, left.val
        return head


if __name__ == '__main__':
    data = [1, 2, 3, 4, 5]
    linked_list = LinkedList(data)

    # data1 = [9]
    # linked_list1 = LinkedList(data1)

    # p: ListNode = linked_list.addTwoNumbers(LinkedList(data1).head, LinkedList(data2).head)
    # p: ListNode = linked_list.middleNode(linked_list.head)
    # result_list: ListNode = linked_list.removeDuplicateNodes(linked_list.head)
    # print(result_list)
    print(linked_list.description(linked_list.swapNodes(linked_list.head, 2)))
