class TRANSLATION:

    @staticmethod
    def to_japanese(text: str) -> str:
        """
        generates a prompt to translate Korean text into Japanese.
        
        :param text: The Korean text to be translated.
        :return: A string prompt for generating the Japanese translation.
        """

        return f"""
            {text}
            Translate the following sentence from Korean to Japanese with high accuracy.
            Preserve grammatical correctness and semantic fidelity.
            Return only the translated Japanese text. No explanations, quotes, or formatting.
            """

class THUMBNAIL:
    @staticmethod
    def segment_prompt() -> str:
        """
        이미지에서 썸네일로 사용할 수 있는 구간을 추출하는 프롬프트입니다.

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
