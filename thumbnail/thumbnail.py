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
from common.config import parser
from common.utils import _category
# .env 파일에서 API 키 로드
load_dotenv()

class Image:
    def __init__(self, images_path: str, ext: str):
        '''
        images_path : 동일한 제품의 이미지들이 포함된 상위 경로
        ext : 이미지 확장자 - 동일한 확장자를 가진 이미지만 처리합니다. 
        '''
        self.args = parser()

        self.images_path = images_path
        self.ext = ext
        self.image_list = glob.glob(images_path + f"/*.{self.ext}")

        # 제품 코드 
        if "(" in os.path.basename(images_path):
            self.code = os.path.basename(images_path).split("(")[0].strip()
        else:
            self.code = os.path.basename(images_path)
        
        # Gemini API 클라이언트 초기화
        self.client = genai.Client(api_key = os.getenv("API_KEY"))

        # logger 초기화
        self.logger = init_logger()

        # 분류 
        self.studio_category = ['모델-스튜디오','모델-연출','상품-연출']
        self.detail_category = ['누끼','마네킹','옷걸이(행거)이미지','상품소재디테일이미지']
    
    def process_image(self):
        for image in self.image_list:
            # 윈도우 생성
            box_rows = self.rolling_window(image)
            print(box_rows)

            # 윈도우 선별 및 크롭
            box_choice_result = self.choose_box(image, box_rows)
            print(box_choice_result)

            # 분류
            self.box_classification([])

            

    

    def rolling_window(self, image_path):
        '''
        1차 분류를 위한 이미지 리사이즈 및 윈도우 생성
        '''
        # 이미지 읽기
        index = os.path.basename(image_path).split("_")[1].split(".")[0]
        image = cv2.imread(image_path)

        # 원본 이미지 크기
        h, w = image.shape[:2]

        # 리사이즈 비율 계산
        resize_ratio = w/self.args.thumbnail_size[0]

        # 리사이즈 이미지 생성
        resized_image = cv2.resize(image, (self.args.thumbnail_size[0], int(h/resize_ratio)))
        self.logger.info(f"원본 이미지 크기: {image.shape}  ---> 리사이즈된 이미지 크기 : {resized_image.shape}")

        # 리사이즈 이미지 저장
        cv2.imwrite(f'{self.images_path}/{self.code}_{index}_resized.{self.ext}', resized_image)
        self.logger.info(f"리사이즈 이미지 저장 완료 : {self.images_path}/{self.code}_{index}_resized.{self.ext}")
        
        window_height = self.args.thumbnail_size[1]
        step_size = window_height // 4  # 윈도우 높이의 1/4만큼씩 이동
        windows = int((int(h/resize_ratio) - window_height) / step_size) + 1
        self.logger.info(f"윈도우 개수 {windows}개에 대해 구간 추출 시작")

        # 윈도우가 그려질 이미지 복사
        window_image = resized_image.copy()

        rows = {}

        for i in range(windows):
            # 랜덤 BGR 색상 생성
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            y1 = i * step_size
            y2 = min(int(h/resize_ratio), y1 + window_height)
            self.logger.info(f"윈도우 {i} : ({y1}, {y2})")

            cv2.rectangle(window_image, (0, y1), (self.args.thumbnail_size[0], y2), color, 3)
            cv2.putText(window_image, f'box_{i}', (0, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            rows[f'box_{i}'] = {
                "y1": y1,
                "y2": y2,
                "color": color,
            }

            # 마지막 반복에서만 이미지 저장
            if i == windows - 1:
                cv2.imwrite(f'{self.images_path}/{self.code}_{index}_windows.{self.ext}', window_image)
                self.logger.info(f"구간 추출 이미지 저장 완료 : {self.images_path}/{self.code}_{index}_windows.{self.ext}")
        
        return rows
        
    def choose_box(self, image_path, box_rows):
        '''
        구간 추출 이미지 중 썸네일로 사용할 수 있도록 객체가 중앙에 오는 이미지 선별 및 1차 분류 진행
        '''
        prompt = THUMBNAIL.choose_box_prompt()
        index = os.path.basename(image_path).split("_")[1].split(".")[0]

        image = self.client.files.upload(file = f'{self.images_path}/{self.code}_{index}_windows.{self.ext}')
        self.logger.info(f"이미지 api에 업로드 : {self.images_path}/{self.code}_{index}_windows.{self.ext}")

        response_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["box", "category"],
                "properties": {
                    "box": {"type": "string"},
                    "category": {"type": "string"}
                }
            }
        }

        self.logger.info("Gemini API 호출 시작")

        response = self.client.models.generate_content(
            model = self.args.model,
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

        result = json.loads(response.text)
                
        # 결과 처리
        for selected_box in result:
            box_info = box_rows.get(selected_box['box'])
            if box_info:
                # 이미지 크롭
                img = cv2.imread(f'{self.images_path}/{self.code}_{index}_resized.{self.ext}')
                cropped = img[box_info['y1']:box_info['y2'], :]

                # 저장
                os.makedirs(f'{self.images_path}/{selected_box["category"]}', exist_ok=True)
                thumbnail_path = f'{self.images_path}/{selected_box["category"]}/{self.code}_{index}_{selected_box["box"]}.{self.ext}'
                cv2.imwrite(thumbnail_path, cropped)
                self.logger.info(f" 저장 완료: {thumbnail_path}")

        return result

    def box_classification(self, exclude_category: list):
        '''
        선별된 윈도우에 대한 2차 분류 진행
        '''
        # 연출 이미지에 대한 분류
        studio_images = glob.glob(f'{self.images_path}/연출 이미지/*.{self.ext}')
        include_category = _category(exclude_category, self.studio_category)
        prompt = THUMBNAIL.classification_studio_prompt(include_category)
        
        for s_img in studio_images:
            img_file = self.client.files.upload(file = s_img)
            response = self.client.models.generate_content(
                model = self.args.model,
                contents = [
                    prompt,
                    img_file
                ]
            )
            
            # 분류 결과에 따라 이미지 이동
            category = response.text.strip()
            if category in include_category:
                # 카테고리 폴더 생성
                os.makedirs(f'{self.images_path}/연출 이미지/{category}', exist_ok=True)
                
                # 이미지 이동
                new_path = f'{self.images_path}/연출 이미지/{category}/{os.path.basename(s_img)}'
                os.rename(s_img, new_path)
                self.logger.info(f"이미지 이동 완료: {s_img} -> {new_path}")

        # 디테일 이미지에 대한 분류
        detail_images = glob.glob(f'{self.images_path}/디테일 이미지/*.{self.ext}')
        include_category = _category(exclude_category, self.detail_category)
        prompt = THUMBNAIL.classification_detail_prompt(include_category)
        
        for d_img in detail_images:
            img_file = self.client.files.upload(file = d_img)
            response = self.client.models.generate_content(
                model = self.args.model,
                contents = [
                    prompt,
                    img_file
                ]
            )
            
            # 분류 결과에 따라 이미지 이동
            category = response.text.strip()
            if category in include_category:
                # 카테고리 폴더 생성
                os.makedirs(f'{self.images_path}/디테일 이미지/{category}', exist_ok=True)
                
                # 이미지 이동
                new_path = f'{self.images_path}/디테일 이미지/{category}/{os.path.basename(d_img)}'
                os.rename(d_img, new_path)
                self.logger.info(f"이미지 이동 완료: {d_img} -> {new_path}")

        


