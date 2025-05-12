import os
import re

import cv2
from PIL import Image, ImageDraw, ImageFont
import easyocr

import numpy as np
from pathlib import Path
from common.config import parser
from gemini import process_texts


def check_font_language_support(font_path, text="日本語テスト"):
    """폰트가 특정 언어(기본값: 일본어)를 지원하는지 확인"""
    try:
        from PIL import ImageFont

        # 폰트 로드
        font = ImageFont.truetype(font_path, 12)

        # 각 문자에 대해 폰트 지원 확인
        for char in text:
            try:
                # 문자를 렌더링할 수 있는지 확인 (getbbox 메서드는 PIL 9.2.0 이상 필요)
                width = font.getbbox(char)[2]
                if width <= 0:
                    print(f"[WARNING] 문자 '{char}' (U+{ord(char):04X})가 폰트 '{font_path}'에서 지원되지 않음")
                    return False
            except:
                # getbbox가 없으면 getsize 시도 (구 버전 PIL)
                width, _ = font.getsize(char)
                if width <= 0:
                    print(f"[WARNING] 문자 '{char}' (U+{ord(char):04X})가 폰트 '{font_path}'에서 지원되지 않음")
                    return False

        return True
    except Exception as e:
        print(f"[ERROR] 폰트 확인 실패: {e}")
        return False


class Translation:
    def __init__(self):
        self.args = parser()
        self.reader = easyocr.Reader(self.args.src_lang)

    def extract_product_code(self, image_path):
        """
            이미지의 상위 폴더명을 기반으로 제품 코드를 추출 합니다.
            향후 제품 코드 명 규칙이 변경될 경우, 해당 로직을 수정해야 합니다.
        """
        return os.path.basename(os.path.dirname(image_path))

    def load_image(self, image_path):
        """한글 경로 때문에 추가"""
        image_path = str(Path(image_path).resolve())
        if os.name == 'nt' and any(ord(c) > 127 for c in image_path):
            try:
                stream = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
                return image
            except Exception as e:
                print(f"[ERROR] Failed to load image from {image_path}: {e}")
                return None
        else:
            return cv2.imread(image_path)

    def read_images(self, extensions=['jpg', 'jpeg', 'png']) -> list:
        root_dir = Path(self.args.input_dir) / self.args.project
        abs_root = root_dir.resolve()
        # print(f"[DEBUG] Absolute search path: {abs_root}")

        image_list = []
        for ext in extensions:
            matches = list(abs_root.rglob(f"*.{ext}"))
            # print(f"[DEBUG] Found {len(matches)} *.{ext} files")

            for p in matches:
                path_str = str(p.resolve())
                image = self.load_image(path_str)
                if image is not None:
                    image_list.append((path_str, image))
                else:
                    print(f"[WARN] Could not load image: {path_str}")

        return image_list

    def preprocess_for_ocr(self, image):
        resized = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)

        # hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        # v = hsv[:, :, 2]  # 밝기 채널
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # v_equalized = clahe.apply(v)

        alpha, beta = 1.8, 0
        enhanced = cv2.addWeighted(gray, alpha, gray, 0, beta)

        gray = cv2.medianBlur(enhanced, 3)

        # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # thresh = cv2.adaptiveThreshold(
        #     gray, 255,
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY, 11, 2
        # )

        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        processed = cv2.erode(dilated, kernel, iterations=1)

        return processed

    # def extract_korean_text(self, image_path, image):
    #
    #     if image is None:
    #         print(f"Error: Could not load image: {image_path}")
    #         return None, [], []
    #
    #     processed_image = self.preprocess_for_ocr(image)
    #     results = self.reader.readtext(
    #         processed_image,
    #         paragraph=False,
    #         min_size=1,
    #         text_threshold=0.5,
    #         low_text=0.3,
    #         link_threshold=0.3,
    #         mag_ratio=1.5,
    #         contrast_ths=0.5,
    #         adjust_contrast=0.7,
    #         decoder='greedy'
    #     )
    #
    #     korean_texts = []
    #     bboxes = []
    #
    #     for bbox, text, confidence in results:
    #         if confidence >= 0.2 and re.search(r'[가-힣]', text):
    #             bboxes.append(bbox)
    #             korean_texts.append(text)
    #
    #     if self.args.debug:
    #         if image is not None and bboxes:
    #             image_with_boxes = image.copy()
    #             for box in bboxes:
    #                 pts = np.array(box, np.int32)
    #                 pts = pts.reshape((-1, 1, 2))
    #                 cv2.polylines(image_with_boxes, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    #
    #             # 디버그 디렉토리 생성 및 확인
    #             debug_dir = os.path.join(self.output_dir, "bbox")
    #             os.makedirs(debug_dir, exist_ok=True)
    #
    #             # 결과 이미지 저장
    #             debug_image_path = os.path.join(debug_dir, f"{self.filename}.png")
    #             cv2.imwrite(debug_image_path, image_with_boxes)
    #             print(f"[DEBUG] 바운딩 박스 이미지 저장됨: {debug_image_path}")
    #
    #             # 결과 이미지 화면에 표시
    #             cv2.namedWindow("Detected Korean Text (Polylines)", cv2.WINDOW_NORMAL)
    #             cv2.imshow("Detected Korean Text (Polylines)", image_with_boxes)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #
    #     # 전처리에서 1.5배 해줬기 때문에 원복 필요
    #     scale = 1 / 1.5
    #     adjusted_bboxes = []
    #
    #     for bbox in bboxes:
    #         adjusted_bbox = [[int(x * scale), int(y * scale)] for x, y in bbox]
    #         adjusted_bboxes.append(adjusted_bbox)
    #
    #     return image, adjusted_bboxes, korean_texts

    def extract_korean_text(self, image_path, image):
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return None, [], []

        # 이미지 전처리 (1.5배 확대 포함)
        processed_image = self.preprocess_for_ocr(image)

        # OCR 수행
        results = self.reader.readtext(
            processed_image,
            paragraph=False,
            min_size=1,
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.3,
            mag_ratio=1.5,
            contrast_ths=0.5,
            adjust_contrast=0.7,
            decoder='greedy'
        )

        korean_texts = []
        bboxes = []

        # 한국어 텍스트 필터링
        for bbox, text, confidence in results:
            if confidence >= 0.2 and re.search(r'[가-힣]', text):
                bboxes.append(bbox)
                korean_texts.append(text)

        # 전처리에서 1.5배 해줬기 때문에 원복 필요
        scale = 1 / 1.5
        adjusted_bboxes = []

        for bbox in bboxes:
            adjusted_bbox = [[int(x * scale), int(y * scale)] for x, y in bbox]
            adjusted_bboxes.append(adjusted_bbox)

        # 원본 이미지 크기에 맞게 조정된 바운딩 박스 사용
        if self.args.debug and image is not None and adjusted_bboxes:
            image_with_boxes = image.copy()

            # 각 바운딩 박스에 인덱스 번호 표시를 위한 폰트 설정
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 1

            for i, box in enumerate(adjusted_bboxes):
                # 다각형 그리기
                pts = np.array(box, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image_with_boxes, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # 텍스트 인덱스 표시 (시각화를 위해)
                center_x = int(sum(pt[0] for pt in box) / len(box))
                center_y = int(sum(pt[1] for pt in box) / len(box))

                # 인덱스 번호를 바운딩 박스 중앙에 표시
                cv2.putText(image_with_boxes, str(i), (center_x, center_y),
                            font, font_scale, (0, 0, 255), font_thickness)

            # 디버그 디렉토리 생성 및 확인
            debug_dir = os.path.join(self.output_dir, "bbox")
            os.makedirs(debug_dir, exist_ok=True)

            # 결과 이미지 저장
            debug_image_path = os.path.join(debug_dir, f"{self.filename}.png")
            cv2.imwrite(debug_image_path, image_with_boxes)
            print(f"[DEBUG] 바운딩 박스 이미지 저장됨: {debug_image_path}")

            # OCR 텍스트와 인덱스를 함께 저장
            text_debug_path = os.path.join(debug_dir, f"{self.filename}_text.txt")
            with open(text_debug_path, "w", encoding="utf-8") as f:
                for i, text in enumerate(korean_texts):
                    f.write(f"Box {i}: {text}\n")

            # 결과 이미지 화면에 표시
            cv2.namedWindow("Detected Korean Text", cv2.WINDOW_NORMAL)
            cv2.imshow("Detected Korean Text", image_with_boxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image, adjusted_bboxes, korean_texts

    def translate_text(self, text, src_lang, target_lang):
        try:
            if src_lang == 'ko' and target_lang == 'jp':
                # step1. translate Korean to English
                english_text = process_texts([text], src_lang=src_lang, target_lang='en')
                if not english_text or not isinstance(english_text, list):
                    print(f"영어 번역 실패: {text}")
                    return None
                english = english_text[0]

                # step2. English to Japanese
                japanese_text = process_texts([english], src_lang='en', target_lang=target_lang)

                if not japanese_text or not isinstance(japanese_text, list):
                    print(f"일본어 번역 실패: {english}")
                    return None
                result = japanese_text[0]

            else:
                result = process_texts([text], src_lang=src_lang, target_lang=target_lang)[0]

            if isinstance(result, dict):
                # 명시적으로 각 키를 확인하고 로깅하는 방식으로 변경
                for key in ["translation", "request", "jp"]:
                    if key in result and result[key]:
                        return result[key]

                # 아무 키도 찾지 못했을 경우 로깅
                print(f"번역 결과에 유효한 키가 없습니다: {result}")
                return str(result)  # 사전 전체를 문자열로 반환

            return str(result)

        except Exception as e:
            print(f"[ERROR] '{text}' 번역 실패: {e}")
            return None

    def inpaint_image(self, image, bbox):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

        inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return inpainted_image

    def add_text_to_image(self, image, bbox, text):

        if not text:
            return image

        if isinstance(text, dict):
            text = text.get("translation")
        elif isinstance(text, list):
            text = text[0] if text else "번역 실패"

        text = str(text)

        # Convert OpenCV image to PIL image
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        # Calculate bounding box dimensions
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Convert to OpenCV format for background color estimation
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        pad = self.args.text_padding
        roi = image_cv[max(y_min - pad, 0):min(y_max + pad, image_cv.shape[0]),
              max(x_min - pad, 0):min(x_max + pad, image_cv.shape[1])]
        mean_color = tuple(np.mean(roi.reshape(-1, 3), axis=0).astype(np.uint8).tolist()) if roi.size else (
        255, 255, 255)
        mean_color_rgb = (mean_color[0], mean_color[1], mean_color[2])

        # Calculate font size based on box dimensions and text_size_ratio
        font_size = int(min(box_width, box_height) * self.args.text_size_ratio / len(text))
        font_size = max(font_size, 30)  # Set minimum font size 10px

        try:
            font = ImageFont.truetype(self.args.font_family, font_size)
        except:
            font = ImageFont.load_default()

        # Calculate text position (center of box)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 텍스트 위치 계산
        text_x = x_min + (box_width - text_width) / 2
        text_y = y_min + (box_height - text_height) / 2

        # Draw background rectangle with mean color
        draw.rectangle(
            [(text_x - 2, text_y - 2), (text_x + text_width + 2, text_y + text_height + 2)],
            fill=mean_color_rgb
        )

        if isinstance(text, str):
            try: # UTF-8로 텍스트 인코딩 확인
                text = text.encode('utf-8', errors='ignore').decode('utf-8')
            except:
                pass

        # Draw text
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

        # Convert back to OpenCV image
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)



    def enhance_image_quality(self, image):
        """(optional) Enhance image quality using Super-resolution"""
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            model_path = "ESPCN_x4.pb"  # Path to pre-trained model

            if not os.path.exists(model_path):
                print(f"[ERROR] Super-resolution model not found at {model_path}")
                return image

            sr.readModel(model_path)
            sr.setModel("espcn", 4)  # Set model name and scale
            enhanced_image = sr.upsample(image)
            return enhanced_image
        except Exception as e:
            print(f"[WARNING] Super-resolution enhancement failed: {e}")
            return image

    def process_image(self, image_path, image):

        print(f"Processing image: {image_path}")

        # Step 1: Extract Korean text and bounding boxes
        image, bboxes, ko_texts = self.extract_korean_text(image_path, image)
        if image is None:
            return None
        print(f"Found {len(ko_texts)} Korean text regions")

        # Make a copy of the original image for processing
        result_image = image.copy()

        # Step 3: Process each text region
        jp_texts =[]
        for i, (bbox, ko_text) in enumerate(zip(bboxes, ko_texts)):
            # print(f"  Processing text region {i + 1}: '{ko_text}'")

            # Step 3-1: Inpaint the text region
            result_image = self.inpaint_image(result_image, bbox)

            # Step 3-2: Translate Korean to Japanese (via English)
            jp_text = self.translate_text(ko_text, src_lang=self.args.src_lang, target_lang=self.args.trg_lang)
            jp_texts.append(jp_text)

            # print(f"  Translated to: '{jp_text}'")
            # print(f"Translated '{ko_text}' -> '{jp_text}'")

            # Step 4: Add translated text to the image
            result_image = self.add_text_to_image(result_image, bbox, jp_text)

        # Save translation results as text file
        if self.args.debug:
            self.save_translation_pairs(ko_texts=ko_texts, jp_texts=jp_texts)

        # Step 5 (Optional): Enhance image quality
        if self.args.enhance_quality:
            result_image = self.enhance_image_quality(result_image)

        return result_image

    def run(self):

        image_paths = self.read_images()
        if not image_paths:
            print(f"  No images found in {os.path.join(self.args.input_dir, self.args.project)}")
            return

        print(f"Found {len(image_paths)} images to process")

        # Processesing....
        for image_path, image in image_paths:
            self.filename = os.path.splitext(os.path.basename(image_path))[0]
            product_code = self.extract_product_code(image_path)

            self.output_dir = os.path.join(self.args.output_dir, self.args.project, product_code)
            os.makedirs(self.output_dir, exist_ok=True)

            result_image = self.process_image(image_path, image)
            save_path = os.path.join(self.output_dir, f"{self.filename}.png")

            # Step 6: Save the result
            cv2.imwrite(save_path, result_image)
            print(f"  Saved translated image to: {save_path}")

    def save_translation_pairs(self, ko_texts, jp_texts):
        """번역 결과를 텍스트 파일로 저장"""

        debug_dir = os.path.join(self.output_dir, "txt")
        os.makedirs(debug_dir, exist_ok=True)

        # 번역 쌍 저장
        with open(os.path.join(debug_dir, f"{self.filename}_translations.txt"), "w", encoding="utf-8") as f:
            for ko, jp in zip(ko_texts, jp_texts):
                f.write(f"KO: {ko}\nJP: {jp}\n\n")

if __name__ == "__main__":
    translator = Translation()
    translator.run()
