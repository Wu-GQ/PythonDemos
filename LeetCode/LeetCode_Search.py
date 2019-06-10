import bisect


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


def wiggle_sort(nums: list) -> None:
    """
    摆动排序II
    :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/270/sort-search/1170/
    """
    # nums_list = [4, 5, 5, 6] # [4, 5, 6, 5]
    # nums_list = [1, 5, 1, 1, 6, 4]
    nums.sort()
    length = len(nums)

    if length < 3:
        return

    nums2 = nums.copy()

    start_index = int(length / 2 - 0.5)

    for i in range(length):
        if i & 1 == 0:
            nums[i] = nums2[start_index - int(i / 2)]
        else:
            nums[i] = nums2[length - int(i / 2) - 1]


def find_peak_element(nums: list):
    """
    寻找峰值
    :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/270/sort-search/1171/
    """
    if len(nums) < 2:
        return 0

    if nums[0] > nums[1]:
        return 0

    for i in range(1, len(nums) - 1):
        if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
            return i

    if nums[len(nums) - 1] > nums[len(nums) - 2]:
        return len(nums) - 1


def count_smaller(nums: list) -> list:
    """
    计算右侧小于当前元素的个数
    :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/270/sort-search/1173/
    """
    count_list = []

    result_list = []

    nums.reverse()

    for i in nums:
        index = bisect.bisect_left(count_list, i)  # binary_search(count_list, i)
        count_list.insert(index, i)
        result_list.append(index)

    result_list.reverse()
    return result_list


def binary_search(nums: list, value: int) -> int:
    length = len(nums)

    low = 0
    high = length
    while low <= high:
        mid = (low + high) // 2

        if mid >= length:
            return length

        if nums[mid] > value:
            high = mid - 1
        elif nums[mid] < value:
            low = mid + 1
        elif mid > 0 and nums[mid] == nums[mid - 1]:
            high = high - 1
        else:
            return mid

    return low


def find_duplicate(nums: list) -> int:
    """
    寻找重复数
    :see https://leetcode-cn.com/explore/interview/card/top-interview-quesitons-in-2018/270/sort-search/1172/
    """
    left = 0
    right = len(nums)

    while left < right:
        mid = (left + right) // 2
        count = 0

        for i in nums:
            if i <= mid:
                count += 1

        if count <= mid:
            left = mid + 1
        else:
            right = mid

    return right


if __name__ == "__main__":
    nums_list = [1, 3, 4, 2, 5, 3, 4, 6]

    print(find_duplicate(nums_list))
