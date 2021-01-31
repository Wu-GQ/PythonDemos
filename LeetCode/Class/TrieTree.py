class TrieNode:

    def __init__(self, char: str):
        self.char = char
        self.is_word = False
        self.children: {str: TrieNode} = {}


class TrieTree:

    def __init__(self):
        self.root = TrieNode("/")

    def insert(self, word: str):
        """ 往前缀树中插入一个新的单词 """
        node = self.root
        for i in word:
            if i not in node.children:
                node.children[i] = TrieNode(i)
            node = node.children[i]
        node.is_word = True

    def is_exist(self, word: str) -> bool:
        """ 判断单词在前缀树中是否存在 """
        node = self.root
        for i in word:
            if i not in node.children:
                node = node.children[i]
            else:
                return False
        return node.is_word

    def starts_with(self, word: str) -> bool:
        """ 判断是否以word开头的前缀 """
        node = self.root
        for i in word:
            if node.children[i]:
                node = node.children[i]
            else:
                return False
        return True
