class MaxQueue:
    """
    面试题59 - II. 队列的最大值
    :see https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/
    """

    def __init__(self):
        self.queue = []
        self.max_queue = []

    def max_value(self) -> int:
        return self.max_queue[0] if len(self.max_queue) > 0 else -1

    def push_back(self, value: int) -> None:
        self.queue.append(value)

        while len(self.max_queue) > 0 and self.max_queue[-1] < value:
            self.max_queue.pop(-1)
        self.max_queue.append(value)

    def pop_front(self) -> int:
        if len(self.queue) < 1:
            return -1

        result = self.queue.pop(0)

        if self.max_queue[0] == result:
            self.max_queue.pop(0)

        return result


# Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(value)
# param_3 = obj.pop_front()

if __name__ == '__main__':
    s = MaxQueue()

    s.push_back(1)
    s.push_back(2)
    print(s.max_value())
    print(s.pop_front())
    print(s.max_value())
