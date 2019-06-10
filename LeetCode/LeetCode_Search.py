def largest_number(nums: list):
    """
    最大数
    :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/270/sort-search/1169/
    """
    str_list = [str(i) for i in nums]

    result_string = "".join(quick_sort(str_list))

    start_index = len(result_string) - 1
    for i in range(len(result_string)):
        if result_string[i] != "0":
            start_index = i
            break

    return result_string[start_index:]


def quick_sort(array: list) -> list:
    """ 快速排序 """
    if len(array) < 2:
        return array

    middle_value = array[0]
    left_array = [i for i in array[1:] if compare_str(i, middle_value)]
    right_array = [i for i in array[1:] if not compare_str(i, middle_value)]
    return quick_sort(left_array) + [middle_value] + quick_sort(right_array)


def compare_str(str1: str, str2: str) -> bool:
    """ 字符串比较 """
    # 834 > 8248 > 824 > 8247
    # 12 = 1212 > 121
    i = 0
    j = 0
    while i < len(str1) or j < len(str2):
        if i == len(str1):
            i = 0
        if j == len(str2):
            j = 0

        if str1[i] > str2[j]:
            return True
        elif str1[i] < str2[j]:
            return False
        else:
            i += 1
            j += 1

    return True


if __name__ == "__main__":
    nums_list = [0, 0, 0]
    print(largest_number(nums_list))
