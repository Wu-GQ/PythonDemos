# 背包容量：6
# 物品 | 重量 | 价值
# -------------------
# 水     3      10
# 书     1      3
# 食物   2      9
# 夹克   2      5
# 相机   1      6
# 请问携带哪些物品时价值最高？最高价值是多少？


class Something:
    name = ''
    volume = float('inf')
    value = 0

    def __init__(self, n, vol, val):
        self.name = n
        self.volume = vol
        self.value = val


def function(capacity, things_list) -> list:
    """
    0-1背包问题
    :param capacity:        背包容量
    :param things_list:     物品列表
    :return:                最佳价值及其物品列表
    """
    # 最佳价值列表
    value_array = [[0 for i in range(capacity + 1)] for j in range(len(things_list))]

    # 最佳价值的物品列表
    name_array = [[[] for i in range(capacity + 1)] for j in range(len(things_list))]

    for i in range(len(things_list)):
        for j in range(capacity + 1):
            if things_list[i].volume <= j and things_list[i].value + value_array[i - 1][j - things_list[i].volume] > value_array[i - 1][j]:
                value_array[i][j] = things_list[i].value + value_array[i - 1][j - things_list[i].volume]

                if name_array[i - 1][j - things_list[i].volume] is None:
                    name_array[i][j].append(things_list[i].name)
                else:
                    a = name_array[i - 1][j - things_list[i].volume].copy()
                    a.append(things_list[i].name)
                    name_array[i][j] = a
            else:
                value_array[i][j] = value_array[i - 1][j]
                name_array[i][j] = name_array[i - 1][j].copy()

    return [value_array, name_array]


if __name__ == "__main__":
    bag_capacity = 6
    things_list = [Something("水", 3, 10),
                   Something("书", 1, 3),
                   Something("食物", 2, 9),
                   Something("夹克", 2, 5),
                   Something("相机", 1, 6)]

    print(function(bag_capacity, things_list))
