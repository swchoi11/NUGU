from common.config import CATEGORY_LIST

def is_valid(category: list) -> bool:

    for item in category:
        if item not in CATEGORY_LIST:
            print(f"Invalid category: {item}")
            print(f"Valid category: {CATEGORY_LIST}")
            return False
    return True


def valid_category(exclude_category: list, full_category: list) -> list:
    if exclude_category is None:
        return full_category

    if not is_valid(exclude_category):
        raise ValueError("Invalid category")
    else:
        for item in exclude_category:
            if item in full_category:
                full_category.remove(item)
        return full_category

