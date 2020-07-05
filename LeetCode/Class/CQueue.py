class CQueue:
    """
    剑指 Offer 09. 用两个栈实现队列
    :see https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/
    """

    def __init__(self):
        # stack1 负责入
        self.stack1 = []
        # stack2 负责出
        self.stack2 = []

    def appendTail(self, value: int) -> None:
        self.stack1.append(value)

    def deleteHead(self) -> int:
        if not self.stack1 and not self.stack2:
            return -1
        elif not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()


if __name__ == '__main__':
    q = CQueue()
    q.appendTail(3)
    q.appendTail(2)
    q.appendTail(1)
    print(q.deleteHead())
    print(q.deleteHead())
    print(q.deleteHead())
    print(q.deleteHead())
    q.appendTail(4)
    q.appendTail(5)
    print(q.deleteHead())
    print(q.deleteHead())
    print(q.deleteHead())
    print(q.deleteHead())
    print(q.deleteHead())
