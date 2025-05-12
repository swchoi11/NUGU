import time
import os
import cv2
import json
import glob
from PIL import Image
from google import genai
from google.genai import types, errors

from common import init_logger
from common.config import parser
from common.prompt import THUMBNAIL
from common.utils import valid_category
from common.path_manager import PathManager
from segment import ImageSegmentor


class ImageProcessor(ImageSegmentor):
    def __init__(self):
        self.args = parser()
        self.path_manager = None
        super().__init__(self.path_manager)

        self.client = genai.Client(api_key=self.args.api_key)
        self.logger = init_logger()

        self.studio_category = ['모델-스튜디오', '모델-연출', '상품-연출']
        self.detail_category = ['누끼', '마네킹', '옷걸이이미지', '상품소재디테일이미지']

    def run(self):
        '''
        resource/thumbnail 폴더에 있는 이미지를 제품 코드별로 처리합니다.
        '''
        # 제품 디렉토리 목록 가져오기
        product_dirs = glob.glob(f'{self.args.input_dir}/{self.args.project}/*')
        self.logger.info(f"찾은 제품 디렉토리: {[str(d) for d in product_dirs]}")

        for product in product_dirs:
            self.path_manager = PathManager(product)

            self.product_code = self.path_manager.product_code
            self.logger.info(f"처리할 경로: {product}")

            image_list = []
            for ext in ['png', 'jpg', 'jpeg']:
                image_list.extend(glob.glob(f'{str(self.path_manager.product_input_dir)}/*.{ext}'))
            self.logger.info(f"찾은 이미지 파일: {[str(f) for f in image_list]}")

            if not image_list:
                self.logger.warning(f'경로에서 이미지를 찾을 수 없습니다 : {product}')
                continue

            for image_path in image_list:
                self.logger.info(f"이미지 처리 시작: {image_path}")
                segment_path, segments = self.segment_image(image_path)

                self.choose_segment(segment_path, segments)
                self.box_classification()
                self.select_thumbnail()

    def choose_segment(self, image_path, segments):
        '''
        구간 추출 이미지 중 썸네일로 사용할 수 있도록 객체가 중앙에 오는 이미지 선별 및 1차 분류 진행
        '''
        prompt = THUMBNAIL.choose_box_prompt()
        image = self.client.files.upload(file=image_path)

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
        for attempt in range(self.args.max_retries + 1):
            try:
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
                break
            except  errors.ServerError as e:
                self.logger.warning(f"Gemini API 서버 오류 (시도 {attempt + 1}/{self.args.max_retries + 1}): {e}")
                if attempt < self.args.max_retries:
                    delay = self.args.initial_delay * (2 ** attempt)
                    self.logger.info(f"다음 시도까지 {delay}초 대기...")
                    time.sleep(delay)
                else:
                    self.logger.error("Gemini API 호출 실패 (최대 재시도 횟수 초과)")
                    raise
            except errors.APIError as e:
                self.logger.error(f"Gemini API 클라이언트 오류: {e}")
                raise
            except Exception as e:
                self.logger.error(f"예상치 못한 오류 발생: {e}")
                raise
            else:
                pass

            if 'response' not in locals():
                self.logger.error("Gemini API 응답을 받지 못했습니다.")
                # 적절한 에러 처리 또는 기본값 설정

            if 'response' in locals() and response:
                # response 처리 로직
                pass

        result = json.loads(response.text)
        self.logger.info(f"분류 결과: {result}")

        for selected_box in result:
            box_info = segments.get(selected_box['box'])
            if box_info:
                self.logger.info(f"박스 정보: {box_info}")
                img = cv2.imread(image_path)
                if img is None:
                    self.logger.error(f"이미지를 읽을 수 없습니다: {image_path}")
                    continue

                self.logger.info(f"원본 이미지 크기: {img.shape}")
                cropped = img[box_info['y1']:box_info['y2'], :]
                self.logger.info(f"크롭된 이미지 크기: {cropped.shape}")

                # 카테고리별 저장
                category = selected_box["category"]

                # 카테고리 디렉토리 생성
                category_dir = self.path_manager.product_output_dir / category
                os.makedirs(category_dir, exist_ok=True)
                self.logger.info(f"카테고리 디렉토리 생성/확인: {category_dir}")

                # 세그먼트 이미지 저장
                segment_filename = f"{os.path.basename(image_path).split('.')[0]}_{selected_box['box']}.png"
                segment_path = category_dir / segment_filename

                try:
                    # BGR에서 RGB로 변환 (OpenCV는 BGR, PIL은 RGB 사용)
                    rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    # numpy 배열을 PIL Image로 변환
                    pil_image = Image.fromarray(rgb_cropped)
                    # 이미지 저장
                    pil_image.save(str(segment_path), 'PNG')
                    self.logger.info(f"세그먼트 이미지 저장 완료: {segment_path}")
                except Exception as e:
                    self.logger.error(f"이미지 저장 중 오류 발생: {str(e)}")

        return result

    def box_classification(self):
        """
            선별된 윈도우에 대한 2차 분류 진행
        """

        # 연출 이미지에 대한 분류
        exclude_category = self.args.exclude_category
        studio_images = glob.glob(f'{self.path_manager.product_output_dir}/연출 이미지/*.*')
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
                os.makedirs(f'{self.path_manager.product_output_dir}/연출 이미지/{category}', exist_ok=True)

                # 이미지 이동
                new_path = f'{self.path_manager.product_output_dir}/연출 이미지/{category}/{os.path.basename(s_img)}'
                os.rename(s_img, new_path)
                self.logger.info(f"이미지 이동 완료: {s_img} -> {new_path}")

        # 디테일 이미지에 대한 분류
        detail_images = glob.glob(f'{self.path_manager.product_output_dir}/디테일 이미지/*.*')
        include_category = valid_category(exclude_category, self.detail_category)
        prompt = THUMBNAIL.classification_detail_prompt(include_category)

        for d_img in detail_images:
            img_file = self.client.files.upload(file=d_img)

            self.logger.info("Gemini API 호출 시작")
            for attempt in range(self.args.max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=self.args.model,
                        contents=[
                            prompt,
                            img_file
                        ]
                    )
                    self.logger.info("Gemini API 호출 성공")
                    break
                except errors.ServerError as e:
                    self.logger.warning(f"Gemini API 서버 오류 (시도 {attempt + 1}/{self.args.max_retries + 1}): {e}")
                    if attempt < self.args.max_retries:
                        delay = self.args.initial_delay * (2 ** attempt)
                        self.logger.info(f"다음 시도까지 {delay}초 대기...")
                        time.sleep(delay)
                    else:
                        self.logger.error("Gemini API 호출 실패 (최대 재시도 횟수 초과)")
                        raise  # 마지막 재시도 실패 시 예외 다시 발생
                except errors.APIError as e:
                    self.logger.error(f"Gemini API 클라이언트 오류: {e}")
                    raise
                except Exception as e:
                    self.logger.error(f"예상치 못한 오류 발생: {e}")
                    raise

            # 분류 결과에 따라 이미지 이동
            category = response.text.strip()
            if category in include_category:
                # 카테고리 폴더 생성
                os.makedirs(f'{self.path_manager.product_output_dir}/연출 이미지/{category}', exist_ok=True)

                # 이미지 이동
                new_path = f'{self.path_manager.product_output_dir}/연출 이미지/{category}/{os.path.basename(d_img)}'
                os.rename(d_img, new_path)
                self.logger.info(f"이미지 이동 완료: {d_img} -> {new_path}")

    def resize_for_thumbnail(self, image_path):
        """
        이미지를 썸네일 사이즈에 맞게 처리합니다.
        * 세로와 가로가 모두 큰 경우 : 비율에 맞춰서 대각선으로 줄이기
        * 세로는 크고 가로는 작은 경우 : 세로만 자르기 나머지는 하얀 배경
        * 가로는 크고 세로는 작은 경우 : 가로만 자르기 나머지는 하얀 배경
        * 세로와 가로가 모두 작은 경우 : 비율에 맞춰서 대각선으로 늘리기
        """
        try:
            # 이미지 로드
            img = Image.open(image_path)

            # 타겟 사이즈
            target_height, target_width = self.args.thumbnail_size

            # 현재 이미지 크기
            current_width, current_height = img.size

            # 새로운 흰색 배경 이미지 생성
            final_img = Image.new('RGB', (target_width, target_height), (255, 255, 255))

            # 1. 세로와 가로가 모두 큰 경우
            if current_height > target_height and current_width > target_width:
                # 비율에 맞춰서 대각선으로 줄이기
                ratio = min(target_width / current_width, target_height / current_height)
                new_width = int(current_width * ratio)
                new_height = int(current_height * ratio)
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # 가운데에 배치
                paste_x = (target_width - new_width) // 2
                paste_y = (target_height - new_height) // 2
                final_img.paste(resized_img, (paste_x, paste_y))

            # 2. 세로는 크고 가로는 작은 경우
            elif current_height > target_height and current_width <= target_width:
                # 세로만 자르기
                ratio = target_width / current_width
                new_width = target_width
                new_height = int(current_height * ratio)
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # 가운데에서 세로만 자르기
                paste_x = 0
                paste_y = (new_height - target_height) // 2
                final_img = resized_img.crop((0, paste_y, target_width, paste_y + target_height))

            # 3. 가로는 크고 세로는 작은 경우
            elif current_height <= target_height and current_width > target_width:
                # 가로만 자르기
                ratio = target_height / current_height
                new_width = int(current_width * ratio)
                new_height = target_height
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # 가운데에서 가로만 자르기
                paste_x = (new_width - target_width) // 2
                paste_y = 0
                final_img = resized_img.crop((paste_x, 0, paste_x + target_width, target_height))

            # 4. 세로와 가로가 모두 작은 경우
            else:
                # 비율에 맞춰서 대각선으로 늘리기
                ratio = max(target_width / current_width, target_height / current_height)
                new_width = int(current_width * ratio)
                new_height = int(current_height * ratio)
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # 가운데에 배치
                paste_x = (target_width - new_width) // 2
                paste_y = (target_height - new_height) // 2
                final_img.paste(resized_img, (paste_x, paste_y))

            return final_img

        except Exception as e:
            self.logger.error(f"이미지 리사이즈 중 오류 발생: {str(e)}")
            return None

    def select_thumbnail(self):
        min_thumbnail_count = self.args.select_frame
        thum_list = []

        # 연출 이미지 처리
        for category in self.studio_category:
            for img in glob.glob(f'{self.path_manager.product_output_dir}/연출 이미지/{category}/*.*'):
                if min_thumbnail_count <= 0:
                    break

                # 썸네일 사이즈에 맞게 이미지 리사이즈
                resized_img = self.resize_for_thumbnail(img)
                if resized_img:
                    # 리사이즈된 이미지 저장
                    output_path = self.path_manager.product_output_dir / 'final' / f"thumbnail_연출이미지_{category}_{os.path.basename(img)}"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    resized_img.save(output_path, 'PNG')
                    thum_list.append(str(output_path))
                    min_thumbnail_count -= 1
                    self.logger.info(f"썸네일 생성 완료: {output_path}")

        # 디테일 이미지 처리
        for category in self.detail_category:
            for img in glob.glob(f'{self.path_manager.product_output_dir}/디테일 이미지/{category}/*.*'):
                if min_thumbnail_count <= 0:
                    break

                # 썸네일 사이즈에 맞게 이미지 리사이즈
                resized_img = self.resize_for_thumbnail(img)
                if resized_img:
                    # 리사이즈된 이미지 저장
                    output_path = self.path_manager.product_output_dir / 'final' / f"thumbnail_디테일이미지_{category}_{os.path.basename(img)}"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    resized_img.save(output_path, 'PNG')
                    thum_list.append(str(output_path))
                    min_thumbnail_count -= 1
                    self.logger.info(f"썸네일 생성 완료: {output_path}")

        # 썸네일 생성이 완료된 후 세그먼트 이미지 삭제
        if not self.args.debug:
            self._cleanup_segment_images()

        return thum_list

    def _cleanup_segment_images(self):
        """
        썸네일 생성이 완료된 후 세그먼트 이미지들을 삭제합니다.
        """
        try:
            # 연출 이미지와 디테일 이미지 디렉토리 삭제
            for category in self.studio_category + self.detail_category:
                category_dir = self.path_manager.product_output_dir / '연출 이미지' / category
                if os.path.exists(category_dir):
                    for file in os.listdir(category_dir):
                        file_path = os.path.join(category_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                                self.logger.info(f"세그먼트 이미지 삭제 완료: {file_path}")
                        except Exception as e:
                            self.logger.error(f"파일 삭제 중 오류 발생: {file_path} - {str(e)}")

                    # 빈 디렉토리 삭제
                    try:
                        os.rmdir(category_dir)
                        self.logger.info(f"디렉토리 삭제 완료: {category_dir}")
                    except Exception as e:
                        self.logger.error(f"디렉토리 삭제 중 오류 발생: {category_dir} - {str(e)}")

            # 연출 이미지 상위 디렉토리 삭제
            parent_dir = self.path_manager.product_output_dir / '연출 이미지'
            if not self.args.debug:
                if os.path.exists(parent_dir):
                    try:
                        os.rmdir(parent_dir)
                        self.logger.info(f"상위 디렉토리 삭제 완료: {parent_dir}")
                    except Exception as e:
                        self.logger.error(f"상위 디렉토리 삭제 중 오류 발생: {parent_dir} - {str(e)}")

            for category in self.studio_category + self.detail_category:
                category_dir = self.path_manager.product_output_dir / '디테일 이미지' / category
                if os.path.exists(category_dir):
                    for file in os.listdir(category_dir):
                        file_path = os.path.join(category_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                                self.logger.info(f"세그먼트 이미지 삭제 완료: {file_path}")
                        except Exception as e:
                            self.logger.error(f"파일 삭제 중 오류 발생: {file_path} - {str(e)}")

                    # 빈 디렉토리 삭제
                    try:
                        os.rmdir(category_dir)
                        self.logger.info(f"디렉토리 삭제 완료: {category_dir}")
                    except Exception as e:
                        self.logger.error(f"디렉토리 삭제 중 오류 발생: {category_dir} - {str(e)}")

            # 연출 이미지 상위 디렉토리 삭제
            parent_dir = self.path_manager.product_output_dir / '디테일 이미지'
            if os.path.exists(parent_dir):
                try:
                    os.rmdir(parent_dir)
                    self.logger.info(f"상위 디렉토리 삭제 완료: {parent_dir}")
                except Exception as e:
                    self.logger.error(f"상위 디렉토리 삭제 중 오류 발생: {parent_dir} - {str(e)}")

        except Exception as e:
            self.logger.error(f"세그먼트 이미지 정리 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    thumb = ImageProcessor()
    thumb.run()