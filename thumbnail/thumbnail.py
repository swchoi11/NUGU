import os
import glob
import json

import cv2
import random
import numpy as np

from google import genai
from google.genai import types

from common import init_logger, timefn
from common.prompt import THUMBNAIL
from common.config import parser
from common.utils import valid_category
from segment import ImageSegmentor

class ImageProcessor(ImageSegmentor):

    def __init__(self):
        super().__init__()
        self.args = parser()
        self.client = genai.Client(api_key=self.args.api_key) # Gemini API 클라이언트 초기화

        # logger 초기화
        self.logger = init_logger()

        ## set variable
        self.studio_category = ['모델-스튜디오', '모델-연출', '상품-연출']
        self.detail_category = ['누끼', '마네킹', '옷걸이(행거)이미지', '상품소재디테일이미지']
        self.studio_output_dir = os.path.join(self.args.output_dir, self.args.project, '연출 이미지')
        self.detail_output_dir = os.path.join(self.args.output_dir, self.args.project, '디테일 이미지')


    def read_images(self, extensions=['jpg', 'jpeg', 'png'])->list:
        root_dir = os.path.join(self.args.input_dir, self.args.project)
        image_list = []
        for ext in extensions:
            image_list.extend(glob.glob(os.path.join(root_dir, f"**/*.{ext}"), recursive=True))

        return image_list

    def choose_box(self, image_path, box_rows):
        """
            구간 추출 이미지 중 썸네일로 사용할 수 있도록 객체가 중앙에 오는 이미지 선별 및 1차 분류 진행
        """
        prompt = THUMBNAIL.choose_box_prompt()
        index = os.path.basename(image_path).split("_")[1].split(".")[0]
        ext = os.path.basename(image_path).split(".")[-1]

        image = self.client.files.upload(file=f'{self.images_path}/{self.code}_{index}_windows.{ext}')
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
            model=self.args.model,
            contents=[
                prompt,
                image
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )

        self.logger.info("Gemini API 호출 완료")

        result = json.loads(response.text)

        # 결과 처리
        for selected_box in result:
            box_info = box_rows.get(selected_box['box'])
            if box_info:
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

    def box_classification(self, exclude_category: list):
        """
            선별된 윈도우에 대한 2차 분류 진행
        """

        # 연출 이미지에 대한 분류
        studio_images = glob.glob(f'{self.studio_output_dir}/*.*')
        include_category = valid_category(exclude_category, self.studio_category)
        prompt = THUMBNAIL.classification_studio_prompt(include_category)

        for s_img in studio_images:
            img_file = self.client.files.upload(file=s_img)
            response = self.client.models.generate_content(
                model=self.args.model,
                contents=[
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
        detail_images = glob.glob(f'{self.detail_output_dir}/*.*')
        include_category = valid_category(exclude_category, self.detail_category)
        prompt = THUMBNAIL.classification_detail_prompt(include_category)

        for d_img in detail_images:
            img_file = self.client.files.upload(file=d_img)
            response = self.client.models.generate_content(
                model=self.args.model,
                contents=[
                    prompt,
                    img_file
                ]
            )

            # 분류 결과에 따라 이미지 이동
            category = response.text.strip()
            if category in include_category:
                # 카테고리 폴더 생성
                os.makedirs(f'{self.studio_output_dir}/{category}', exist_ok=True)

                # 이미지 이동
                new_path = f'{self.detail_output_dir}/{category}/{os.path.basename(d_img)}'
                os.rename(d_img, new_path)
                self.logger.info(f"이미지 이동 완료: {d_img} -> {new_path}")

    def select_thumbnail(self):

        min_thumbnail_count = self.args.select_frame
        # 연출이미지 - 모델-스튜디오, 모델-연출, 상품-연출, 디테일이미지 - 누끼, 마네킹, 옷걸이이미지, 상품소재디테일이미지 순서대로 최소 썸네일 개수만큼 이미지 경로 반환
        thum_list = []
        for category in self.studio_category:
            for img in glob.glob(f'{self.detail_output_dir}/{category}/*.*'):
                thum_list.append(img)

        for category in self.detail_category:
            for img in glob.glob(f'{self.detail_output_dir}/{category}/*.*'):
                thum_list.append(img)

        return thum_list[:min_thumbnail_count - 1]

    def run(self):
        """
            상위 경로에 포함된 모든 이미지에 대해 가장 적합한 썸네일 후보군을 선별하는 과정입니다.
        """
        image_list = self.read_images()
        for image_path in image_list:
            # 1. 이미지에 포함된 객체가 가장 잘 포함되는 구간을 나누기
            self.logger.info(f"이미지 리사이즈 및 객체 선별을 통한 세그먼트 생성을 시작합니다. : {image_path}")
            box_rows = self.segment_image(image_path)
            self.logger.debug(f"이미지 리사이즈 및 객체 선별을 통한 세그먼트 생성 결과 : {box_rows}")

            # 2. 구간별로 1차 분류하기
            self.logger.info(f"구간에 대한 선별 및 1차 분류를 시작합니다. : {image_path}")
            box_choice_result = self.choose_box(image_path, box_rows)
            self.logger.debug(f"구간에 대한 선별 및 1차 분류 결과 : {box_choice_result}")

            # 3. 1차 분류된 구간을 대상으로 2차 분류하기
            self.logger.info(f"2차 분류를 시작합니다. : {image_path}")
            self.box_classification(self.exclude_category)

            # 4. 2차 분류된 구간에 대한 우선순위선정 및 최소 썸네일 개수를 바탕으로 한 최종 썸네일 선택
            self.select_thumbnail()


if __name__ == "__main__":

    thumbnail = ImageProcessor()
    thumbnail.run()
