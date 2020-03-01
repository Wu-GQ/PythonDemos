class MyStack:
    """
    225. 用队列实现栈
    :see https://leetcode-cn.com/problems/implement-stack-using-queues/
    """

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._empty_queue = []
        self._current_queue = []

    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self._current_queue.append(x)

    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        length = len(self._current_queue)
        if length < 1:
            return 0

        while length > 1:
            self._empty_queue.append(self._current_queue.pop(0))
            length -= 1

        self._current_queue, self._empty_queue = self._empty_queue, self._current_queue

        return self._empty_queue.pop(0)

    def top(self) -> int:
        """
        Get the top element.
        """
        length = len(self._current_queue)
        if length < 1:
            return 0

        while length > 1:
            self._empty_queue.append(self._current_queue.pop(0))
            length -= 1

        end = self._current_queue.pop(0)
        self._empty_queue.append(end)

        self._current_queue, self._empty_queue = self._empty_queue, self._current_queue

        return end

    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return len(self._current_queue) == 0


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()

if __name__ == '__main__':
    obj = MyStack()
    obj.push(1)
    obj.push(2)
    obj.push(3)
    print(obj.pop())
    obj.push(4)
    print(obj.pop())
    obj.push(5)
    print(obj.empty())
