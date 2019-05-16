# 回溯法解决： 图的着色问题、八皇后问题、求子集和

# 集合覆盖问题
# 地区集合: ["wa", "mt", "or", "id", "nv", "ut", "ca", "az"]
# a: ["id", "nv", "ut"]
# b: ["wa", "id", "mt"]
# c: ["or", "nv", "ca"]
# d: ["nv", "ut"]
# e: ["ca", "az"]
# 选择尽可能少的集合组合以覆盖所有地区集合

# MARK: 贪婪算法只能求出近似解，并非最优解


def set_covering_problem(regions_set, stations_dict):
    """ 使用贪婪算法解决结合覆盖问题 """
    covered_region_list = set()

    unselected_station = set(stations_dict.keys())

    while not covered_region_list.issuperset(regions_set):
        c = len(covered_region_list)
        s = None
        for station in stations_dict:
            a_set = covered_region_list | set(stations_dict[station])
            if len(a_set) > c:
                c = len(a_set)
                s = station

        covered_region_list |= set(stations_dict[s])
        unselected_station.remove(s)

    print(covered_region_list)
    print(set(stations_dict.keys()) - unselected_station)


if __name__ == '__main__':
    regions_set = {"wa", "mt", "or", "id", "nv", "ut", "ca", "az"}

    stations_dict = {"a": ["id", "nv", "ut"],
                     "b": ["wa", "id", "mt"],
                     "c": ["or", "nv", "ca"],
                     "d": ["nv", "ut"],
                     "e": ["ca", "az"]}

    set_covering_problem(regions_set, stations_dict)
