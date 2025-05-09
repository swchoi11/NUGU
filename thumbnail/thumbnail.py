import os
import glob
import json

import cv2
import random
import numpy as np

from google import genai
from google.cloud.aiplatform.utils import extract_project_and_location_from_parent
from google.genai import types

from common.prompt import THUMBNAIL
from common import init_logger
from common.config import parser
from common.utils import _category

STUDIO_CATEGORY = ['모델-스튜디오', '모델-연출', '상품-연출']
DETAIL_CATEGORY = ['누끼', '마네킹', '옷걸이(행거)이미지', '상품소재디테일이미지']

class ImageProcessor:
    def __init__(self):
        self.args = parser()
        self.client = genai.Client(api_key=self.args.api_key) # Gemini API 클라이언트 초기화

        # logger 초기화
        self.logger = init_logger()

    def read_images(self, extensions=['jpg', 'jpeg', 'png'])->list:
        root_dir = os.path.join(self.args.input_dir, self.args.project)
        image_list = []
        for ext in extensions:
            image_list.extend(glob.glob(os.path.join(root_dir, f"**/*.{ext}"), recursive=True))

        return image_list

    def extract_product_code(self, image_path: list):
        """
            이미지의 상위 폴더명을 기반으로 제품 코드를 추출 합니다.
            향후 제품 코드 명 규칙이 변경될 경우, 해당 로직을 수정해야 합니다.
        """
        return os.path.dirname(image_path).split("//")[-1]

    def resize_image(self, image, target_width: int):
        """
           이미지를 self.args.thumbnail_size 비율에 맞춰 리사이즈 합니다.

        """
        height, width = image.shape[:2]

        # 리사이즈 비율 계산
        resize_ratio = width / target_width
        resized_height = int(height * resize_ratio)

        # 이미지 리사이즈
        resized_image = cv2.resize(image, (target_width, resized_height))
        self.logger.info(f" 원본 이미지 크기: {image.shape}  ---> 리사이즈 된 이미지 크기 : {resized_image.shape}")

        return resized_image, resize_ratio

    def segment_image(self, image_path:str):
        filename = os.path.basename(image_path)
        image = cv2.imread(image_path)

        thumb_width, thumb_height = self.args.thumbnail_size
        resized_image, resize_ratio = self.resize_image(image, target_width=thumb_width)

        if self.args.debug:
        # 리사이즈 이미지 저장
                temp_dir = './resize'
                os.makedirs(temp_dir, exist_ok=True)
                cv2.imwrite(os.path.join(temp_dir, filename), resized_image)
                self.logger.info(f"리사이즈 이미지 저장 완료 : {os.path.join(temp_dir, filename)}")

        display_image = resized_image.copy()
        height, width = resized_image.shape[:2]

        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # 다중 접근법 사용
        # 1. 에지(sobel filter) 강도에 따른 분류
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        sobel_8u = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # 2. 색상 변화량에 따른 분류(HSV 색 공간 사용)
        hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 채도 수직 변화량 계산
        sobel_s = cv2.Sobel(s, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_s = np.absolute(sobel_s)
        sobel_s_8u = np.uint8(255 * abs_sobel_s / np.max(abs_sobel_s) if np.max(abs_sobel_s) > 0 else abs_sobel_s)

        # 3. 두 접근법 조합
        combined_edge = cv2.addWeighted(sobel_8u, 0.7, sobel_s_8u, 0.3, 0)

        window_size = 100
        projection = np.sum(combined_edge, axis=1)
        smoothed = np.convolve(projection, np.ones(window_size) / window_size, mode='same')
        mean_value = np.mean(smoothed)

        # 로컬 최소값 찾기 (경계 후보)
        # 각 위치에서 앞뒤로 일정 범위 내에 더 작은 값이 없으면 로컬 최소값
        local_min_indices = []
        window_half = 50
        for i in range(window_half, len(smoothed) - window_half):
            local_region = smoothed[i - window_half:i + window_half + 1]
            if smoothed[i] == np.min(local_region) and smoothed[i] < mean_value * 0.5:
                local_min_indices.append(i)

        # 최소 거리 기준으로 필터링
        min_distance = 100  # 최소 분할 간격(px)
        filtered_indices = []

        if local_min_indices:
            filtered_indices.append(local_min_indices[0])
            for idx in local_min_indices[1:]:
                if idx - filtered_indices[-1] >= min_distance:
                    filtered_indices.append(idx)

        # 컨텐츠 기반 필터링 - 각 경계가 실제로 내용 변화가 있는지 확인
        valid_boundaries = []
        padding = 20  # 비교할 상/하 영역 크기

        for idx in filtered_indices:
            # 경계 주변의 위아래 영역
            top_region = max(0, idx - padding)
            bottom_region = min(height, idx + padding)

            # 경계 상/하 영역 이미지 특성 비교
            top_area = resized_image[max(0, top_region - padding * 2):top_region, :]
            bottom_area = resized_image[bottom_region:min(height, bottom_region + padding * 2), :]

            if top_area.shape[0] > 0 and bottom_area.shape[0] > 0:
                # 영역의 평균 색상 비교
                top_mean = np.mean(top_area, axis=(0, 1))
                bottom_mean = np.mean(bottom_area, axis=(0, 1))

                # 색상 차이를 유클리드 거리로 계산
                color_diff = np.sqrt(np.sum((top_mean - bottom_mean) ** 2))

                if color_diff > 20 or smoothed[idx] < mean_value * 0.3:
                    valid_boundaries.append(idx)

        valid_boundaries.sort()

        # 유효한 경계가 없으면 원본 이미지 그대로 반환
        if not valid_boundaries:
            print("유효한 경계가 감지되지 않았습니다. 원본 이미지를 그대로 반환합니다.")
            return [resized_image], [], display_image

        # 세그먼트 생성
        segments = []
        rows = {}  # ← 여기서 rows 생성
        start_y = 0
        segment_height_threshold = thumb_height / 2

        for i, boundary in enumerate(valid_boundaries):
            if boundary - start_y >= segment_height_threshold:
                segment = resized_image[start_y:boundary, :]
                segments.append(segment)

                # 시각화를 위한 색상 및 정보 저장
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                rows[f'box_{i}'] = {
                    "y1": start_y,
                    "y2": boundary,
                    "color": color,
                }

            start_y = boundary

        # 마지막 세그먼트 추가
        if height - start_y >= segment_height_threshold:
            segment = resized_image[start_y:height, :]
            segments.append(segment)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            rows[f'box_{len(rows)}'] = {
                "y1": start_y,
                "y2": height,
                "color": color,
            }

        # 세그먼트가 없는 경우
        if not segments:
            print("분할 가능한 세그먼트가 없습니다. 원본 이미지를 그대로 반환 합니다.")
            return [resized_image], [], display_image, {}

        return segments, valid_boundaries, display_image, rows

    def visualize_segment_image(self, image, boundary_points, segments):

        window_original = 'Original Image'
        window_boundaries = 'Image with Boundary Lines'

        # 원본 이미지
        cv2.namedWindow(window_original, cv2.WINDOW_NORMAL)
        cv2.imshow(window_original, image)

        # 경계선 이미지
        image_with_lines = image.copy()
        for point in boundary_points:
            cv2.line(image_with_lines, (0, point), (image.shape[1], point), (0, 0, 255), 2)

        cv2.namedWindow(window_boundaries, cv2.WINDOW_NORMAL)
        cv2.imshow(window_boundaries, image_with_lines)

        # 각 분할된 세그먼트 이미지 표시
        for i, segment in enumerate(segments):
            segment_window = f'Segment {i + 1}'
            cv2.namedWindow(segment_window, cv2.WINDOW_NORMAL)

            cv2.namedWindow(segment_window, cv2.WINDOW_NORMAL)
            cv2.imshow(segment_window, segment)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rolling_window(self, image_path, spacing=4):
        """
            1차 분류(STUDIO_CATEGORY, DETAIL_CATEGORY)를 위한 이미지 리사이즈 및 윈도우 생성
        """
        # 이미지 읽기
        index = os.path.basename(image_path).split("_")[1].split(".")[0]
        filename = os.path.basename(image_path)
        image = cv2.imread(image_path)

        thumb_width, thumb_height = self.args.thumbnail_size
        height, width = image.shape[:2]

        resized_image, resize_ratio = self.resize_image(image, target_width=thumb_width)

        if self.args.debug:
            # 리사이즈 이미지 저장
            temp_dir = './resize'
            os.makedirs(temp_dir, exist_ok=True)
            cv2.imwrite(os.path.join(temp_dir, filename), resized_image)
            self.logger.info(f"리사이즈 이미지 저장 완료 : {os.path.join(temp_dir, filename)}")

        step_size = thumb_height // spacing  # 윈도우 높이의 1/spacing 만큼 이동
        windows = int((int(height / resize_ratio) - thumb_height) / step_size) + 1
        self.logger.info(f"윈도우 개수 {windows}개에 대해 구간 추출 시작")

        window_image = resized_image.copy()
        rows={}

        for i in range(windows):
            # 랜덤 BGR 색상 생성
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            y1 = i * step_size
            y2 = min(int(height / resize_ratio), y1 + thumb_height)
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
                cv2.imwrite(f'{self.images_path}/{self.product_code}_{index}_windows.png', window_image)
                self.logger.info(f"구간 추출 이미지 저장 완료 : {self.images_path}/{self.product_code}_{index}_windows.png")

        return rows

    def choose_box(self, image_path, box_rows):
        '''
        구간 추출 이미지 중 썸네일로 사용할 수 있도록 객체가 중앙에 오는 이미지 선별 및 1차 분류 진행
        '''
        prompt = THUMBNAIL.choose_box_prompt()
        index = os.path.basename(image_path).split("_")[1].split(".")[0]

        image = self.client.files.upload(file=f'{self.images_path}/{self.product_code}_{index}_windows.png')
        self.logger.info(f"이미지 api에 업로드 : {self.images_path}/{self.product_code}_{index}_windows.png")

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
                img = cv2.imread(f'{self.images_path}/{self.product_code}_{index}_resized.png')
                cropped = img[box_info['y1']:box_info['y2'], :]

                # 저장
                os.makedirs(f'{self.images_path}/{selected_box["category"]}', exist_ok=True)
                thumbnail_path = f'{self.images_path}/{selected_box["category"]}/{self.product_code}_{index}_{selected_box["box"]}.png'
                cv2.imwrite(thumbnail_path, cropped)
                self.logger.info(f" 저장 완료: {thumbnail_path}")

        return result

    def box_classification(self, exclude_category: list):
        '''
        선별된 윈도우에 대한 2차 분류 진행
        '''
        # 연출 이미지에 대한 분류
        studio_images = glob.glob(f'{self.images_path}/연출 이미지/*.png')
        include_category = _category(exclude_category, self.studio_category)
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
        detail_images = glob.glob(f'{self.images_path}/디테일 이미지/*.png')
        include_category = _category(exclude_category, self.detail_category)
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
                os.makedirs(f'{self.images_path}/디테일 이미지/{category}', exist_ok=True)

                # 이미지 이동
                new_path = f'{self.images_path}/디테일 이미지/{category}/{os.path.basename(d_img)}'
                os.rename(d_img, new_path)
                self.logger.info(f"이미지 이동 완료: {d_img} -> {new_path}")

    def run(self):
        image_list = self.read_images()
        for image_path in image_list:

            segments, valid_boundaries, resized_image, rows = self.segment_image(image_path)
            if self.args.debug:
                print("image_path : ", image_path)
                self.visualize_segment_image(segments=segments, boundary_points=valid_boundaries, image=resized_image)

            # box_rows = self.rolling_window(image_path)
            # print(box_rows)
            #
            # # 윈도우 선별 및 크롭
            # box_choice_result = self.choose_box(image_path, box_rows)
            # print(box_choice_result)
            #
            # # 분류
            # self.box_classification([])

if __name__ == "__main__":

    thumbnail = ImageProcessor()
    thumbnail.run()
