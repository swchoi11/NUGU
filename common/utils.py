from common.config import parser

def valid_category(category: list) -> bool:
    valid_category = ['모델-스튜디오', '모델-연출', '상품-연출', '누끼', '마네킹', '옷걸이(행거)이미지', '상품소재디테일이미지']
    for item in category:
        if item not in valid_category:
            print(f"Invalid category: {item}")
            print(f"Valid category: {valid_category}")
            return False

    return True

def _category(exclude_category: list, full_category: list) -> list:
    if exclude_category is None:
        return full_category

    if not valid_category(exclude_category):
        raise ValueError("Invalid category")
    else:
        for item in exclude_category:
            if item in full_category:
                full_category.remove(item)
        return full_category
