"""
 translation, thumbnail 알고리즘 수행에 필요한 prompt 정의
"""

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
    def extract(text: str) -> str:
        """
        generates prompt for extracting usable thumbnail regions from an image.

        :param text: A descriptive prompt or text that provides context or guidance for thumbnail extraction.
        :return: A string containing guidelines to identify vertical segments of an image suitable for thumbnails.
        """
        return """
            다음 이미지를 보고 썸네일로 사용할 수 있는 구간을 추출해주세요.
            가로는 원본 이미지와 동일하게 사용할 것이므로 세로 구간만 추출해주세요.
    
            썸네일로 사용할 수 있는 구간을 평가하는 기준은 다음과 같습니다.
            * 모델이 이미지에 포함되어 있는 경우
               * 모델이 이미지의 중앙에 위치하는 이미지
            * 모델이 이미지에 포함되어 있지 않은 경우
               * 상품이 이미지의 중앙에 위치하는 이미지
               * 텍스트나 표가 포함되지 않는 이미지
               * 배경이 흰색이나 단색 등 스튜디오에서 찍히지 않은 이미지
    
            결과 예시는 다음과 같습니다. 
            - 결과 예시
            
            [
                {
                    "image_path": "image_path",
                    "segment": [y1, y2]
                }
            ]
            """

    @staticmethod
    def extract_color(text: str) -> str:
        return print("extract color map")
