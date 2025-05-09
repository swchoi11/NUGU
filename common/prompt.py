"""
 translation, thumbnail 알고리즘 수행에 필요한 prompt 정의
"""
from common.utils import valid_category
class TRANSLATION:

    @staticmethod
    def family_instruction()->str:

        return f"""
            Generates a prompt to translate text from the source language to the target language
            using polite (non-casual) Japanese speech style (~です / ~ます form).
        """

    @staticmethod
    def to_japanese(text: str, src_lang, target_lang) -> str:
        """
        generates a prompt to translate {src_lang} text into {target_lang}.
        
        :param text: The {src_lang} text to be translated.
        :return: A string prompt for generating the {target_lang} translation(원본).
        """

        return f"""
            {text}
            Translate the following sentence from {src_lang} to {target_lang} with high accuracy.
            Preserve grammatical correctness and semantic fidelity.
            Return only the translated {target_lang} text. No explanations, quotes, or formatting.
            """

class THUMBNAIL:

    @staticmethod
    def choose_box_prompt() -> str:
        return """
            다음 이미지는 구간이 색으로 구분되어 있습니다. 
            사각형의 왼쪽 아래에 구간의 이름이 경계와 동일한 색으로 표기되어 있습니다. 

            이미지를 보고 썸네일로 사용할 수 있는 구간을 추출해주세요.

            썸네일로 사용할 수 있는 구간을 평가하는 기준은 다음과 같습니다.
            * 모델이 이미지에 포함되어 있는 경우
               * 모델이 이미지의 중앙에 위치하는 이미지
            * 모델이 이미지에 포함되어 있지 않은 경우
               * 상품이 이미지의 중앙에 위치하는 이미지
               * 텍스트나 표가 포함되지 않는 이미지
               * 배경이 흰색이나 단색 등 스튜디오에서 찍히지 않은 이미지

            썸네일로 사용할 수 있다고 판단되는 경우 두가지 카테고리 중 하나로 분류해 주세요.
            분류 기준은 다음과 같습니다.
            < 분류 기준 >
            - 연출 이미지: 모델, 배경과 연출 요소가 포함된 이미지
            - 디테일 이미지: 제품의 질감, 소재, 부분 확대 등 세부 정보를 중심으로 한 이미지

            분류는 배타적으로 진행해야 합니다. 
            즉, 연출 이미지는 디테일 이미지가 아니고, 디테일 이미지는 연출 이미지가 아닙니다.


            결과 예시는 다음과 같습니다. 
            - 결과 예시

            [
                {
                    "box": "box_1",
                    "category": "연출 이미지"
                },
                {
                    "box": "box_4",
                    "category": "디테일 이미지"
                }
            ]
        """

    @staticmethod
    def classification_studio_prompt(include_category: list) -> str:
        if not valid_category(include_category):
            raise ValueError("Invalid category")

        return f"""
            다음 이미지는 아래의 분류 기준 중 어떤 이미지에 해당하는지 1개만 선택해 주세요.

            < 분류 기준 >
            {include_category}
            """

    @staticmethod
    def classification_detail_prompt(include_category: list) -> str:
        if not valid_category(include_category):
            raise ValueError("Invalid category")

        return f"""
            다음 이미지는 아래의 분류 기준 중 어떤 이미지에 해당하는지 1개만 선택해 주세요.

            < 분류 기준 >
            {include_category}

            """
