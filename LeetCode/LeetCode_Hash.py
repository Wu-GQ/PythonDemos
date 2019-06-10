def title_to_number(s: str) -> int:
    result = 0
    for ch in s:
        result = result * 26 + ord(ch) - 64
    return result


if __name__ == "__main__":
    string = 'AB'

    print(title_to_number(string))
