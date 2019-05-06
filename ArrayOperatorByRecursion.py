def array_sum(array, start_index) -> int:
    """ 数组求和 """
    if start_index == len(array):
        return 0
    return array[start_index] + array_sum(array, start_index + 1)


def array_sum_2(array) -> int:
    """ 数组求和 """
    return 0 if not array else array[0] + array_sum_2(array[1:])


def array_count(array, start_index) -> int:
    """ 数组个数 """
    return 0 if start_index == len(array) else 1 + array_count(array, start_index + 1)


def array_count_2(array):
    """ 数组个数 """
    return 0 if not array else 1 + array_count_2(array[1:])


def array_max(array) -> int:
    """ 数组求最大值 """
    if len(array) == 1:
        return array[0]

    a = array_max(array[1:])
    return array[0] if array[0] > a else a


def array_quick_sort(array) -> list:
    """ 快速排序 """
    if len(array) < 2:
        return array

    middle_value = array[0]
    left_array = [i for i in array[1:] if i <= middle_value]
    right_array = [i for i in array[1:] if i > middle_value]
    return left_array + [middle_value] + right_array


# <editor-fold desc="折叠后要显示的内容">

def array_merge_sort(array) -> list:
    """ 归并排序 - 分 """
    if len(array) < 2:
        return array
    left_array = array_merge_sort(array[0:len(array) // 2])
    right_array = array_merge_sort(array[len(array) // 2:])
    return array_merge_sort_merge(left_array, right_array)


def array_merge_sort_merge(left_array, right_array) -> list:
    """ 归并排序 - 并 """
    i = 0
    j = 0
    new_list = []

    while i < len(left_array) and j < len(right_array):
        if left_array[i] < right_array[j]:
            new_list.append(left_array[i])
            i += 1
        else:
            new_list.append(right_array[j])
            j += 1

    while i < len(left_array):
        new_list.append(left_array[i])
        i += 1

    while j < len(right_array):
        new_list.append(right_array[j])
        j += 1

    return new_list

# </editor-fold>


if __name__ == '__main__':
    array_list = [0, 2, 5, 9, 7, 5, 6]
    # print(array_sum(array_list, 0))
    # print(array_sum_2(array_list))
    #
    # print(array_count(array_list, 0))
    # print(array_count_2(array_list))

    # print(array_max(array_list))
    # print(array_quick_sort(array_list))

    print(array_merge_sort(array_list))
