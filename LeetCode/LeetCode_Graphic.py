class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: list) -> int:
        """
        单词接龙
        :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/273/graph-theory/1182/
        """
        # 使用广度优先遍历的思想，判断两个单词的最短路径
        # PS1: 直接使用广度优先遍历时，遇到较长的wordList非常耗时。因此，当wordList大于26时，使用直接替换字母然后搜索的方式加快速度
        # PS2: 使用双向的广度优先遍历，哪边的stack比较少，就遍历哪边，只有两边的stack拥有相同数据时，计算步数
        if endWord not in wordList or beginWord == endWord or len(beginWord) != len(endWord):
            return 0

        word_set: set = set(wordList)

        # 用来存储广度优先遍历时的数据
        stack: list = [beginWord]
        # 用来记录步数
        step_stack: list = [1]

        while len(stack) > 0:
            last_word = stack.pop(0)
            last_word_step = step_stack.pop(0)

            # 遍历搜索具有相同前缀和后缀的单词，加入的队列
            for i in range(0, len(last_word)):
                for char in "abcdefghijklmnopqrstuvwxyz":
                    word = "{0}{1}{2}".format(last_word[:i], char, last_word[i + 1:])
                    if word == endWord:
                        return last_word_step + 1
                    elif word in word_set:
                        stack.append(word)
                        step_stack.append(last_word_step + 1)
                        word_set.remove(word)

        return 0


if __name__ == '__main__':
    print(Solution().ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
