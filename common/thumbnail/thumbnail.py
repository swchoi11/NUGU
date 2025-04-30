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
import google.generativeai as genai
from google.genai.types import HttpOptions, Part
from google.genai.types import SafetySetting, GenerationConfig

from common.prompt import THUMBNAIL

# .env 파일에서 API 키 로드
load_dotenv()

class Image:
    def __init__(self, image_path: str, ext: str):
        self.image_path = image_path
        self.ext = ext
        self.image_list = glob.glob(image_path + f"/*.{self.ext}")
        self.code = os.path.basename(image_path).split("(")[0].strip()
        
        # Gemini API 클라이언트 초기화
        # self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash-001')

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

        prompt = THUMBNAIL.segment_prompt()
        image = self._image_content(image)

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "image_path": {"type": "STRING"},
                "segment": {"type": "ARRAY", "items": {"type": "NUMBER"}},
            },
            "required": ["image_path", "segment"],
        }


        generate_config = {
            "temperature": 0,
            "top_p": 0.95,
            "max_output_tokens": 128,
            # "response_mime_type": "application/json",
            # "response_schema": response_schema
        }

        response = self.model.generate_content(
            contents = [
                {
                    "mime_type": f"image/{self.ext}",
                    "data": image
                },
                prompt
            ],
            generation_config = generate_config
        )

        time.sleep(5)
        
        return response.text
        

    def _image_content(self, image):
        with open(image, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        return image_data



