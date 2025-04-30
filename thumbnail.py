# 1. 이미지 받아서 분류하기
# 2. 분류된 이미지 중 썸네일로 쓸 이미지 구간 찾기
# 3. 이미지 구간 찾은걸로 썸네일 만들기
# 4. 썸네일 만든거에 대한 mood, color pallette 추출

## 분류
# 1차 분류 : 연출 이미지, 디테일 이미지
# 2차 분류 : 모델-스튜디오, 모델-연출, 상품-연출, 누끼, 마네킹, 옷걸이(행거) 이미지, 상품소재 디테일 이미지


# 1. 이미지 받아서 분류하기
 
import os
import glob
import json
import time
import cv2
import base64
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions, Part

# .env 파일에서 API 키 로드
load_dotenv()

class Image:
    def __init__(self, image_path: str, ext: str):
        self.image_path = image_path
        self.ext = ext
        self.image_list = glob.glob(image_path + f"/*.{self.ext}")
        self.code = os.path.basename(image_path).split("(")[0].strip()
        self.thumbnail_size = [268, 335]
        
        # Gemini API 클라이언트 초기화
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def process_image(self):
        rows = []
        for image in self.image_list:
            response = self.get_segment(image)
            response = json.loads(response)

            rows.append({
                "image_path": image,
                "code": self.code,
                "segment": response.get('segment')
            })  

        return pd.DataFrame(rows)

    def get_segment(self, image):
        print(f"Processing {image} ...")

        prompt = Part.from_text(self._prompt())
        image = self._image_content(image)

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "image_path": {"type": "STRING"},
                "segment": {"type": "ARRAY", "items": {"type": "NUMBER"}},
            },
            "required": ["image_path", "segment"],
        }

        safety_settings = [
            SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ]

        generate_content_config = GenerationConfig(
            temperature=0,
            top_p=0.95,
            max_output_tokens=128,
            response_mime_type="application/json",
            response_schema=response_schema
        )


        response = self.client.generate_content(
            model="gemini-2.0-flash-001",
            contents=[
                self._prompt(),
                self._image_content(image)
            ],
            response_mime_type="application/json",
            response_schema=response_schema
        )
        time.sleep(5)
        return response.text
    
    def _prompt(self):
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
    
    def _image_content(self, image):
        with open(image, "rb") as f:
            raw_obj = base64.b64encode(f.read()).decode("utf-8")
            content_obj = Part.from_data(data=base64.b64decode(raw_obj), mime_type=f"image/{self.ext}")
        return content_obj



if __name__ == "__main__":

    image = Image("./data/DAOXGZ5738(통이미지)", "png")
    df = image.process_image()
    print(df)

