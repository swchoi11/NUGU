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
import random  # 랜덤 색상 생성을 위해 추가

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from common.prompt import THUMBNAIL
from common import init_logger, timefn

# .env 파일에서 API 키 로드
load_dotenv()

class Image:
    def __init__(self, image_path: str, ext: str):
        self.image_path = image_path
        self.ext = ext
        self.image_list = glob.glob(image_path + f"/*.{self.ext}")
        self.code = os.path.basename(image_path).split("(")[0].strip()
        
        # Gemini API 클라이언트 초기화
        self.client = genai.Client(api_key = os.getenv("API_KEY"))
        self.model_id = 'gemini-2.0-flash-001'
        self.logger = init_logger()
        self.target_w = 820
        self.target_h = 1024

    # def process_image(self):
    #     rows = []
    #     for image in self.image_list:
    #         response = self.get_segment(image)
    #         response = json.loads(response)

    #         rows.append({
    #             "image_path": image,
    #             "code": self.code,
    #             "segment": response.get('segment')
    #         })  

    #     return pd.DataFrame(rows)
    
    @timefn
    def get_segment(self, image, resize_ratio):
        self.logger.info(f"이미지 처리 시작: {image}")

        prompt = THUMBNAIL.segment_prompt()

        image = cv2.imread(image)
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(w/resize_ratio), int(h/resize_ratio)))
        self.logger.debug(f"이미지 크기: {image.shape}")

        cv2.imwrite('./resource/thumbnail/DOYMAW9090/resized.png', image)
        self.logger.debug("리사이즈된 이미지 저장 완료")

        image = self.client.files.upload(file = './resource/thumbnail/DOYMAW9090/resized.png')
        self.logger.debug("이미지 업로드 완료")

        response_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["image_path", "index", "segment"],
                "properties": {
                    "image_path": {"type": "string"},
                    "index": {"type": "number"},
                    "segment": {"type": "array", "items": {"type": "number"}}
                }
            }
        }

        self.logger.info("Gemini API 호출 시작")
        response = self.client.models.generate_content(
            model = self.model_id,
            contents = [
                prompt,
                image
            ],
            config = types.GenerateContentConfig(
                response_mime_type = "application/json",
                response_schema = response_schema
            )
        )
        self.logger.info("Gemini API 호출 완료")

        return json.loads(response.text)
        

    def rolling_segment(self, image, resize_ratio):
        image = cv2.imread(image)
        h, w = image.shape[:2]

        resize_ratio = w/self.target_w

        resized_image = cv2.resize(image, (self.target_w, int(h/resize_ratio)))

        cv2.imwrite(f'./resource/thumbnail/DOYMAW9090/resized.png', resized_image)
        
        self.logger.info(f"원본 이미지 크기: {image.shape}  ---> 리사이즈된 이미지 크기 : {resized_image.shape}")
        
        windows = int(h/resize_ratio) // self.target_h

        self.logger.info(f"윈도우 개수 : {windows}")

        image = cv2.imread(f'./resource/thumbnail/DOYMAW9090/resized.png')

        for i in range(windows+1):
            # 랜덤 BGR 색상 생성
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            y1 = max(0, i * self.target_h - 100*i)
            y2 = min(int(h/resize_ratio), y1 + self.target_h)
            self.logger.info(f"y1 : {y1}, y2 : {y2}, id: {i}")

            cv2.rectangle(image, (0, y1), (self.target_w, y2), color, 3)
            cv2.putText(image, f'image_{i}', (0, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            # 마지막 반복에서만 이미지 저장
            if i == windows:
                cv2.imwrite(f'./resource/thumbnail/DOYMAW9090/final_result.png', image)

    def choose_image(self):

        prompt = """다음 이미지는 구간이 색으로 구분되어 있습니다. 
                 이 중 썸네일로 사용할 수 있는 구간을 선택해 주세요. 
                 사각형의 왼쪽 아래에 구간의 이름이 경계와 동일한 색으로 표기되어 있습니다.
                 썸네일로 사용할 구간의 이름을 출력해 주세요. 가장 유력한 구간 순서대로 3개만 출력해주세요.
                 
                 예시)
                 image_0
                 image_1
                 """
        image = self.client.files.upload(file = './resource/thumbnail/DOYMAW9090/final_result.png')
        self.logger.debug("이미지 업로드 완료")

        response_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["image_path", "index", "segment"],
                "properties": {
                    "image_path": {"type": "string"},
                    "index": {"type": "number"},
                    "segment": {"type": "array", "items": {"type": "number"}}
                }
            }
        }

        self.logger.info("Gemini API 호출 시작")
        response = self.client.models.generate_content(
            model = self.model_id,
            contents = [
                prompt,
                image
            ]
        )
        
        self.logger.info("Gemini API 호출 완료")

        # return json.loads(response.text)
        return response.text


        

