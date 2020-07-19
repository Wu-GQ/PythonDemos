def sort(a: list, x):
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        print(lo, hi, mid)
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid

    return lo


if __name__ == '__main__':
    # print(sort([0, 1, 2, 3, 5, 6], 4))
    a = {1: 2}
    b = {1: 1}
    a.update(b)
    print(1 | 1)
