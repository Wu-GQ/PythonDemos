class Solution:

    def changeTitle(self, filepath: str):
        """ 修改所有章节的名字 """
        import os, re

        def new_title(matched) -> str:
            return f'第{matched.group(1)}章 {matched.group(2)}'

        if not os.path.exists(filepath):
            print(f'Error: 文件路径不存在. File path: {filepath}')
            return

        new_file = open("第一序列(1).txt", 'a')
        for line in open(filepath):
            new_file.write(re.sub(r'^(\d+)、(.*\n)$', new_title, line))
        new_file.close()


if __name__ == '__main__':
    s = Solution()
    s.changeTitle('第一序列.txt')
