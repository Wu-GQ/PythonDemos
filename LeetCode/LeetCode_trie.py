class TrieNode:

    def __init__(self, value):
        self.value = value
        self.childNodeDict = {}

    def add_word(self, word: str) -> None:
        """ 添加单词 """
        if not word:
            return

        first_char = word[-1]
        if first_char not in self.childNodeDict:
            self.childNodeDict[first_char] = TrieNode(first_char)
        self.childNodeDict[first_char].add_word(word[:-1])

    def total_child_node_count_per_branch(self, father_node_count: int) -> int:
        """ 计算该节点的所有分支上的子节点的总数量 """
        if not self.childNodeDict:
            return 1 + father_node_count

        count = 1 + father_node_count
        child_count = 0
        for node in self.childNodeDict:
            child_count += self.childNodeDict[node].total_child_node_count_per_branch(count)
        return child_count


class Solution:

    def minimumLengthEncoding(self, words: list) -> int:
        """
        820. 单词的压缩编码
        :see
        """
        if not words:
            return 0

        root: TrieNode = TrieNode('#')
        for i in words:
            root.add_word(i)
        return root.total_child_node_count_per_branch(0)


if __name__ == '__main__':
    s = Solution()
    print(s.minimumLengthEncoding(["time", "me", "atime"]))
