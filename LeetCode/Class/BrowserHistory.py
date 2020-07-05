class BrowserHistory:
    """
    5430. 设计浏览器历史记录
    """

    def __init__(self, homepage: str):
        self.current_index = 0
        self.url_list = [homepage]
        self.length = 1

    def visit(self, url: str) -> None:
        self.current_index += 1
        if self.current_index == len(self.url_list):
            self.url_list.append(url)
        else:
            self.url_list[self.current_index] = url
        self.length = self.current_index + 1

    def back(self, steps: int) -> str:
        self.current_index = max(0, self.current_index - steps)
        return self.url_list[self.current_index]

    def forward(self, steps: int) -> str:
        self.current_index = min(self.length - 1, self.current_index + steps)
        return self.url_list[self.current_index]


# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)

if __name__ == '__main__':
    obj = BrowserHistory('leetcode')
    obj.visit('google')
    obj.visit('facebook')
    obj.visit('youtube')
    print(obj.back(1))
    print(obj.back(1))
    print(obj.forward(1))
    obj.visit('linkedin')
    print(obj.forward(2))
    print(obj.back(2))
    print(obj.back(7))
