from functools import reduce
from queue import Queue


def single_number(nums: list) -> int:
    """
    136. 只出现一次的数字
    # see: https://leetcode-cn.com/explore/featured/card/top-interview-quesitons-in-2018/261/before-you-start/1106/
    """
    # a = 0
    # for i in nums:
    #     a ^= i
    # return a
    return reduce(lambda x, y: x ^ y, nums)


def majority_element(nums: list) -> int:
    """
    求众数
    # see: https://leetcode-cn.com/explore/featured/card/top-interview-quesitons-in-2018/261/before-you-start/1107/
    """
    count_dict = {}
    max_count = 0
    max_count_number = float('inf')

    for i in nums:
        if i in count_dict:
            count_dict[i] += 1
        else:
            count_dict[i] = 1

        if count_dict[i] > max_count:
            max_count = count_dict[i]
            max_count_number = i

    return max_count_number


def search_matrix(matrix, target) -> bool:
    """
    二维数组找数字
    # see: https://leetcode-cn.com/explore/featured/card/top-interview-quesitons-in-2018/261/before-you-start/1108/
    """
    # MARK: 这种做法的效率是比较低的。可以考虑从左下角或者右上角的数字开始搜索。
    # 用广度优先搜索
    len_x = len(matrix)
    if len_x < 1:
        return False

    len_y = len(matrix[0])
    if len_y < 1:
        return False

    if matrix[0][0] == target:
        return True
    elif matrix[0][0] > target:
        return False

    queue = Queue()
    queue.put((0, 0))
    searched_set = {(0, 0)}

    len_x = len(matrix)
    len_y = len(matrix[0])

    while not queue.empty():
        index_list = queue.get()
        x = index_list[0]
        y = index_list[1]

        if x < len_x - 1 and matrix[x + 1][y] == target:
            return True
        elif x < len_x - 1 and matrix[x + 1][y] < target and (x + 1, y) not in searched_set:
            queue.put((x + 1, y))
            searched_set.add((x + 1, y))

        if y < len_y - 1 and matrix[x][y + 1] == target:
            return True
        elif y < len_y - 1 and matrix[x][y + 1] < target and (x, y + 1) not in searched_set:
            queue.put((x, y + 1))
            searched_set.add((x, y + 1))

    return False


def combine_arrays(nums1: list, m: int, nums2: list, n: int) -> None:
    """
    合并两个有序数组
    :see: https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/261/before-you-start/1109/
    """
    # Mark: 这只是取巧的方法，正确的办法应该是，从末尾开始比较，选择nums1和nums2中最大的那个值放在末尾即可
    if n == 0:
        return

    tmp = list(nums1[:m])
    i = 0
    j = 0
    for index in range(m + n):
        if i == m:
            nums1[index] = nums2[j]
            j += 1
        elif j == n:
            nums1[index] = tmp[i]
            i += 1
        elif tmp[i] <= nums2[j]:
            nums1[index] = tmp[i]
            i += 1
        else:
            nums1[index] = nums2[j]
            j += 1


if __name__ == '__main__':
    # print(search_matrix(array, 0))

    array1 = [1, 2, 4, 10, 90, 100, 0, 0, 0]
    m = 6
    array2 = [6, 80, 120]
    n = 3
    combine_arrays(array1, m, array2, n)

    print(array1)
