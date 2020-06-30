class LFUCache:
    """
    460. LFU缓存
    https://leetcode-cn.com/problems/lfu-cache/
    """

    def __init__(self, capacity: int):
        # 最大容量
        self.capacity = capacity
        # 键值对
        self.key_value_dict = {}
        # 键频对
        self.key_frequency_dict = {}
        # 频键列表
        self.frequency_key_list = [[]]

    def get(self, key: int) -> int:
        # print(f'---- get({key}) ----')

        if key not in self.key_value_dict:
            return -1

        # 获得频率
        frequency = self.key_frequency_dict[key]

        # 修改键频列表
        self.frequency_key_list[frequency].remove(key)
        if frequency + 1 < len(self.frequency_key_list):
            self.frequency_key_list[frequency + 1].append(key)
        else:
            self.frequency_key_list.append([key])

        # 修改键频对
        self.key_frequency_dict[key] += 1

        # print(self.key_value_dict)
        # print(self.key_frequency_dict)
        # print(self.frequency_key_list)

        return self.key_value_dict[key]

    def put(self, key: int, value: int) -> None:
        if self.capacity < 1:
            return

        if key in self.key_value_dict:
            frequency = self.key_frequency_dict[key]

            self.frequency_key_list[frequency].remove(key)
            if frequency + 1 < len(self.frequency_key_list):
                self.frequency_key_list[frequency + 1].append(key)
            else:
                self.frequency_key_list.append([key])

            self.key_frequency_dict[key] += 1
        else:
            if len(self.key_value_dict) == self.capacity:
                index = 0
                while len(self.frequency_key_list[index]) == 0:
                    index += 1
                delete_key = self.frequency_key_list[index].pop(0)
                del (self.key_frequency_dict[delete_key])
                del (self.key_value_dict[delete_key])

            self.frequency_key_list[0].append(key)
            self.key_frequency_dict[key] = 0

        self.key_value_dict[key] = value

        # print(f'---- put({key}, {value}) ----')
        # print(self.key_value_dict)
        # print(self.key_frequency_dict)
        # print(self.frequency_key_list)


# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

# ["LFUCache","put","put","put","put","put","get","put","get","get","put","get","put","put","put","get","put","get","get","get","get","put","put","get","get","get","put","put","get","put","get","put","get","get","get","put","put","put","get","put","get","get","put","put","get","put","put","put","put","get","put","put","get","put","put","get","put","put","put","put","put","get","put","put","get","put","get","get","get","put","get","get","put","put","put","put","get","put","put","put","put","get","get","get","put","put","put","get","put","put","put","get","put","put","put","get","get","get","put","put","put","put","get","put","put","put","put","put","put","put"]
# [[10],[10,13],[3,17],[6,11],[10,5],[9,10],[13],[2,19],[2],[3],[5,25],[8],[9,22],[5,5],[1,30],[11],[9,12],
# [7],[5],[8],[9],[4,30],[9,3],[9],[10],[10],[6,14],[3,1],[3],[10,11],[8],[2,14],[1],[5],[4],[11,4],[12,24],
# [5,18],[13],[7,23],[8],[12],[3,27],[2,12],[5],[2,9],[13,4],[8,18],[1,7],[6],[9,29],[8,21],[5],[6,30],[1,12],
# [10],[4,15],[7,22],[11,26],[8,17],[9,29],[5],[3,4],[11,30],[12],[4,29],[3],[9],[6],[3,4],[1],[10],[3,29],[10,28],
# [1,20],[11,13],[3],[3,12],[3,8],[10,9],[3,26],[8],[7],[5],[13,17],[2,27],[11,15],[12],[9,19],[2,15],[3,16],[1],[12,17],
# [9,1],[6,19],[4],[5],[5],[8,1],[11,7],[5,2],[9,28],[1],[2,2],[7,4],[4,22],[7,24],[9,26],[13,28],[11,26]]
# [null,null,null,null,null,null,-1,null,19,17,null,-1,null,null,null,-1,null,-1,5,-1,12,null,null,3,5,5,null,null,1,null,-1,null,30,5,30,null,null,null,-1,null,-1,24,null,null,18,null,null,null,null,-1,null,null,18,null,null,-1,null,null,null,null,null,18,null,null,24,null,4,29,30,null,12,-1,null,null,null,null,29,null,null,null,null,17,-1,18,null,null,null,24,null,null,null,-1,null,null,null,-1,18,18,null,null,null,null,-1,null,null,null,null,null,null,null]
# [null,null,null,null,null,null,-1,null,19,17,null,-1,null,null,null,-1,null,-1,5,-1,12,null,null,3,5,5,null,null,1,null,-1,null,30,5,30,null,null,null,-1,null,-1,24,null,null,18,null,null,null,null,14,null,null,18,null,null,11,null,null,null,null,null,18,null,null,-1,null,4,29,30,null,12,11,null,null,null,null,29,null,null,null,null,17,-1,18,null,null,null,-1,null,null,null,20,null,null,null,29,18,18,null,null,null,null,20,null,null,null,null,null,null,null]
if __name__ == '__main__':
    cache: LFUCache = LFUCache(10)
    cache.put(10, 13)
    cache.put(3, 17)
    cache.put(6, 11)
    cache.put(10, 5)
    cache.put(9, 10)
    print(cache.get(13))
    cache.put(2, 19)
    print(cache.get(2))
    print(cache.get(3))
    cache.put(5, 25)
    print(cache.get(8))
    cache.put(9, 22)
    cache.put(5, 5)
    cache.put(1, 30)
    print(cache.get(11))
    cache.put(9, 12)
    print(cache.get(7))
    print(cache.get(5))
    print(cache.get(8))
    print(cache.get(9))
    cache.put(4, 30)
    cache.put(9, 3)
    print(cache.get(9))
    print(cache.get(10))
    print(cache.get(10))
    cache.put(6, 14)
    cache.put(3, 1)
    print(cache.get(3))
    cache.put(10, 11)
    print(cache.get(8))
    cache.put(2, 14)
    print(cache.get(1))
    print(cache.get(5))
    print(cache.get(4))
    cache.put(11, 4)
    cache.put(12, 24)
    # [5,18],[13],[7,23],[8],[12],[3,27],[2,12],[5],[2,9],[13,4],[8,18],[1,7],[6],[9,29],[8,21],[5],[6,30],[1,12],
    cache.put(5, 18)
    print(cache.get(13))
    cache.put(7, 23)
    print(cache.get(8))
    print(cache.get(12))
    cache.put(3, 27)
    cache.put(2, 12)
    print(cache.get(5))
    cache.put(2, 9)
    cache.put(13, 4)
    cache.put(8, 18)
    cache.put(1, 7)
    print(cache.get(6))
    cache.put(9, 29)
    cache.put(8, 21)
    print(cache.get(5))
    cache.put(6, 30)
    cache.put(1, 12)
    # [10],[4,15],[7,22],[11,26],[8,17],[9,29],[5],[3,4],[11,30],[12],[4,29],[3],[9],[6],[3,4],[1],[10],[3,29],[10,28],
    print(cache.get(10))
