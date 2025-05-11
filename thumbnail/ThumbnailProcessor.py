## 목차
# 1. 이미지에 포함된 객체가 가장 잘 포함되는 구간을 나누기
# 2. 구간별로 1차 분류하기
# 3. 1차 분류된 구간을 대상으로 2차 분류하기
# 4. 2차 분류된 구간에 대한 우선순위선정 및 최소 썸네일 개수를 바탕으로 한 최종 썸네일 선택
# 5. 썸네일 만든거에 대한 mood, color pallette 추출
import os
import cv2
import glob
import json
import numpy as np

from dotenv import load_dotenv
from google import genai
from google.genai import types

from common import init_logger, timefn
from common.prompt import THUMBNAIL
from common.config import parser
from common.utils import valid_category
from thumbnail.SegmentProcessor import SegmentProcessor



# .env 파일에서 API 키 로드
load_dotenv()

class ThumbnailProcessor:
    def __init__(self, images_path: str):
        '''
        images_path : 동일한 제품의 이미지들이 포함된 상위 경로
        jpg, png, jpeg 확장자를 가진 이미지만 처리합니다. 
        '''
        self.args = parser()

        self.images_path = images_path
        self.image_list = []
        for ext in ['jpg','jpeg','png']:
            self.image_list.extend(glob.glob(images_path + f"/*.{ext}"))
        print(self.image_list)

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
        self.detail_category = ['누끼','마네킹','옷걸이이미지','상품소재디테일이미지']
    
    @timefn
    def process_image(self, exclude_category: list):
        '''
        상위 경로에 포함된 모든 이미지에 대해 가장 적합한 썸네일 후보군을 선별하는 과정입니다.
        '''
        for image in self.image_list:
            
            # 1. 이미지에 포함된 객체가 가장 잘 포함되는 구간을 나누기
            self.logger.info(f"이미지 리사이즈 및 객체 선별을 통한 세그먼트 생성을 시작합니다. : {image}")
            box_rows = self.segment_image(image)
            self.logger.debug(f"이미지 리사이즈 및 객체 선별을 통한 세그먼트 생성 결과 : {box_rows}")

            # 2. 구간별로 1차 분류하기
            self.logger.info(f"구간에 대한 선별 및 1차 분류를 시작합니다. : {image}")
            box_choice_result = self.choose_box(image, box_rows)
            self.logger.debug(f"구간에 대한 선별 및 1차 분류 결과 : {box_choice_result}")

            # 3. 1차 분류된 구간을 대상으로 2차 분류하기
            self.logger.info(f"2차 분류를 시작합니다. : {image}")
            self.box_classification(exclude_category)

            # 4. 2차 분류된 구간에 대한 우선순위선정 및 최소 썸네일 개수를 바탕으로 한 최종 썸네일 선택
            self.select_thumbnail()

            # 5. 썸네일 만든거에 대한 mood, color pallette 추출
    @timefn
    def segment_image(self, image_name):
        '''
        1차 분류를 위한 이미지 리사이즈 및 윈도우 생성
        '''
        seg_processor = SegmentProcessor(self.images_path, image_name)
        rows = seg_processor.segment_image()
        return rows

    @timefn
    def choose_box(self, image_path, box_rows):
        '''
        구간 추출 이미지 중 썸네일로 사용할 수 있도록 객체가 중앙에 오는 이미지 선별 및 1차 분류 진행
        '''
        prompt = THUMBNAIL.choose_box_prompt()
        index = os.path.basename(image_path).split("_")[1].split(".")[0]
        ext = os.path.basename(image_path).split(".")[-1]

        image = self.client.files.upload(file = f'{self.images_path}/{self.code}_{index}_windows.{ext}')
        self.logger.info(f"이미지 api에 업로드 : {self.images_path}/{self.code}_{index}_windows.{ext}")

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
                print(box_info)
                # 이미지 크롭
                img = cv2.imread(f'{self.images_path}/{self.code}_{index}_resized.{ext}')
                cropped = img[box_info['y1']:box_info['y2'], :]

                # 저장
                os.makedirs(f'{self.images_path}/{selected_box["category"]}', exist_ok=True)
                thumbnail_path = f'{self.images_path}/{selected_box["category"]}/{self.code}_{index}_{selected_box["box"]}.{ext}'
                cv2.imwrite(thumbnail_path, cropped)
                self.logger.info(f" 저장 완료: {thumbnail_path}")

        os.remove(f'{self.images_path}/{self.code}_{index}_windows.{ext}')
        os.remove(f'{self.images_path}/{self.code}_{index}_resized.{ext}')

        return result

    @timefn
    def box_classification(self, exclude_category: list):
        '''
        선별된 윈도우에 대한 2차 분류 진행
        '''
        # ext = os.path.basename().split(".")[-1]

        # 연출 이미지에 대한 분류
        studio_images = glob.glob(f'{self.images_path}/연출 이미지/*.*')
        include_category = valid_category(exclude_category, self.studio_category)
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
        detail_images = glob.glob(f'{self.images_path}/디테일 이미지/*.*')
        include_category = valid_category(exclude_category, self.detail_category)
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

        

    def select_thumbnail(self):

        min_thumbnail_count = self.args.select_frame

        # 연출이미지 - 모델-스튜디오, 모델-연출, 상품-연출, 디테일이미지 - 누끼, 마네킹, 옷걸이이미지, 상품소재디테일이미지 순서대로 최소 썸네일 개수만큼 이미지 경로 반환
        thum_list = []
        for category in self.studio_category:
            for img in glob.glob(f'{self.images_path}/연출 이미지/{category}/*.*'):
                thum_list.append(img)

        for category in self.detail_category:
            for img in glob.glob(f'{self.images_path}/디테일 이미지/{category}/*.*'):
                thum_list.append(img)

        return thum_list[:min_thumbnail_count-1]