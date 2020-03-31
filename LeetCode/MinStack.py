class MinStack:
    """
    最小栈
    :see https://leetcode-cn.com/explore/interview/card/top-interview-questions-easy/24/design/59/
    """

    def __init__(self):
        """
        initialize your data structure here.
        """
        # 用来存正常的数据
        self.stack = []
        # 用来存最小的数据
        self.min_stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        # 比较前一存入的数据,如果新加入的数据比较小，则在另一栈中存入新的数据
        self.min_stack.append(self.min_stack[-1] if len(self.min_stack) > 0 and self.min_stack[-1] < x else x)

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
