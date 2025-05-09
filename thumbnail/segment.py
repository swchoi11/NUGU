import cv2
import os
import random
import numpy as np
from common.config import parser


class ImageSegmentor:
    def __init__(self):
        self.args = parser()
        self.is_visual = False

    def extract_product_code(self, image_path):
        """
            이미지의 상위 폴더명을 기반으로 제품 코드를 추출 합니다.
            향후 제품 코드 명 규칙이 변경될 경우, 해당 로직을 수정해야 합니다.
        """
        return os.path.dirname(image_path).split("/")[-1]

    def resize_image(self, image_path):
        """
           이미지를 self.args.thumbnail_size 비율에 맞춰 리사이즈 합니다.
        """
        self.product_code = self.extract_product_code(image_path)
        filename = os.path.basename(image_path)

        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        thumb_height, thumb_width = self.args.thumbnail_size

        # 리사이즈 비율 계산
        resize_ratio = width / thumb_width
        resized_height = int(height * resize_ratio)

        # 이미지 리사이즈
        resized_image = cv2.resize(image, (thumb_width, resized_height))
        self.logger.info(f" 원본 이미지 크기: {image.shape}  ---> 리사이즈 된 이미지 크기 : {resized_image.shape}")

        if self.args.debug:
            # 리사이즈 이미지 저장
            self.resize_dir = f'{self.root_dir}/resize'
            os.makedirs(self.resize_dir, exist_ok=True)
            cv2.imwrite(os.path.join(self.resize_dir, filename), resized_image)
            self.logger.info(f"리사이즈 이미지 저장 완료 : {os.path.join(self.resize_dir, filename)}")

        return resized_image

    def _detect_edges(self, image):
        """
            이미지의 에지를 감지합니다.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 다중 접근법 사용
        # 1. 에지(sobel filter) 강도에 따른 분류
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        sobel_8u = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # 2. 색상 변화량에 따른 분류(HSV 색 공간 사용)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 채도 수직 변화량 계산
        sobel_s = cv2.Sobel(s, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_s = np.absolute(sobel_s)
        sobel_s_8u = np.uint8(255 * abs_sobel_s / np.max(abs_sobel_s) if np.max(abs_sobel_s) > 0 else abs_sobel_s)

        # 3. 두 방법을 조합
        return cv2.addWeighted(sobel_8u, 0.7, sobel_s_8u, 0.3, 0)

    def _find_boundary_candidates(self, edge_image):
        """
            에지 이미지에서 경계 후보를 찾습니다.
        """
        window_size = 100
        projection = np.sum(edge_image, axis=1)
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

        return local_min_indices, smoothed, mean_value

    def _filter_boundaries(self, candidates, min_distance=100):
        """
        경계 후보를 필터링합니다.
        """
        filtered_indices = []
        if candidates:
            filtered_indices.append(candidates[0])
            for idx in candidates[1:]:
                if idx - filtered_indices[-1] >= min_distance:
                    filtered_indices.append(idx)
        return filtered_indices

    def _valid_boundaries(self, image, filtered_indices, smoothed, mean_value):
        """
        컨텐츠 기반 필터링 - 각 경계가 실제로 내용 변화가 있는지 확인
        필터링된 경계의 유효성을 검증합니다.
        """
        valid_boundaries = []
        padding = 20
        height = image.shape[0]

        for idx in filtered_indices:
            # 경계 주변의 위아래 영역
            top_region = max(0, idx - padding)
            bottom_region = min(height, idx + padding)

            # 경계 상/하 영역 이미지 특성 비교
            top_area = image[max(0, top_region - padding * 2):top_region, :]
            bottom_area = image[bottom_region:min(height, bottom_region + padding * 2), :]

            if top_area.shape[0] > 0 and bottom_area.shape[0] > 0:
                # 영역의 평균 색상 비교
                top_mean = np.mean(top_area, axis=(0, 1))
                bottom_mean = np.mean(bottom_area, axis=(0, 1))

                # 색상 차이를 유클리드 거리로 계산
                color_diff = np.sqrt(np.sum((top_mean - bottom_mean) ** 2))

                if color_diff > 20 or smoothed[idx] < mean_value * 0.3:
                    valid_boundaries.append(idx)

        return sorted(valid_boundaries)

    def _create_segments(self, image, valid_boundaries):
        """
        유효한 경계를 기반으로 세그먼트를 생성합니다.
        """
        segments = []
        rows = {}
        start_y = 0
        height = image.shape[0]
        segment_height_threshold = self.args.thumbnail_size[1] / 2

        for i, boundary in enumerate(valid_boundaries):
            if boundary - start_y >= segment_height_threshold:
                segment = image[start_y:boundary, :]
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
            segment = image[start_y:height, :]
            segments.append(segment)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            rows[f'box_{len(rows)}'] = {
                "y1": start_y,
                "y2": height,
                "color": color,
            }

        return segments, valid_boundaries, image, rows

    def segment_image(self, image_path:str):

        self.filename = os.path.basename(image_path)
        resized_image = self.resize_image(image_path)
        display_image = resized_image.copy()

        # 1. 엣지 감지
        edge_image = self._detect_edges(resized_image)

        # 2. 경계 후보 찾기
        local_min_indices, smoothed, mean_value = self._find_boundary_candidates(edge_image)

        # 3. 경계 필터링
        filtered_indices = self._filter_boundaries(local_min_indices)

        # 4. 경계 유효성 검증
        valid_boundaries = self._valid_boundaries(resized_image, filtered_indices, smoothed, mean_value)

        if not valid_boundaries:
            self.logger.info("유효한 경계가 감지되지 않았습니다. 원본 이미지를 그대로 반환합니다.")
            return [resized_image], [], display_image, {}

        # 세그먼트 생성
        segments, valid_boundaries, image, rows = self._create_segments(resized_image, valid_boundaries)
        if self.is_visual:
            # 세그먼트 중간 결과물 확인
            self.visualize(image, valid_boundaries, segments)

        if not segments:
            self.logger.info("분할 가능한 세그먼트가 없습니다. 원본 이미지를 그대로 반환 합니다.")
            return [resized_image], [], display_image, {}

        # 윈도우 이미지 저장
        window_image = display_image.copy()
        for box_id, box_info in rows.items():
            cv2.rectangle(window_image, (0, box_info['y1']), (window_image.shape[1], box_info['y2']), box_info['color'],
                          2)
            cv2.putText(window_image, f'{box_id}', (0, box_info['y2']), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        box_info['color'], 2)

        self.segment_dir = f'{self.root_dir}/window'
        os.makedirs(self.segment_dir, exist_ok=True)

        cv2.imwrite(f'{self.segment_dir}/{self.filename}', window_image)
        self.logger.info(f"윈도우 이미지 저장 완료 : {self.segment_dir}/{self.filename}")

        return rows

    def visualize(self, image, boundary_points, segments):

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