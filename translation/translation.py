import os
import glob

import cv2
from PIL import Image, ImageDraw, ImageFont
import easyocr

import numpy as np
from common.config import parser
from gemini import process_texts


class Translation:
    def __init__(self):
        self.args = parser()
        self.reader = easyocr.Reader([self.args.src_lang])

        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(self.args.output_dir, self.args.project)
        os.makedirs(self.output_dir, exist_ok=True)

    def read_images(self, extensions=['jpg', 'jpeg', 'png'])->list:
        root_dir = os.path.join(self.args.input_dir, self.args.project)
        image_list = []
        for ext in extensions:
            image_list.extend(glob.glob(os.path.join(root_dir, f"**/*.{ext}"), recursive=True))
        return image_list

    def extract_korean_text(self, image_path):
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return None, [], []

        results = self.reader.readtext(image)

        korean_texts = []
        bboxes = []

        for bbox, text, confidence in results:
            if confidence < 0.3:
                continue
            if any('\uac00' <= ch <= '\ud7a3' for ch in text):  # Check for Korean characters
                bboxes.append(bbox)
                korean_texts.append(text)

        if self.args.debug:
            if image is not None and bboxes:
                image_with_boxes = image.copy()
                for bbox in bboxes:
                    cv2.rectangle(image_with_boxes, bbox[0], bbox[2], (0, 255, 0), 2)  # Green rectangle

                cv2.namedWindow("Detected Korean Text", cv2.WINDOW_NORMAL)
                cv2.imshow("Detected Korean Text", image_with_boxes)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return image, bboxes, korean_texts

    def translate_text(self, text, src_lang, target_lang):
        if src_lang == 'ko' and target_lang == 'jp':
            # step1. translate Korean to English
            english_text = process_texts([text], src_lang=src_lang, target_lang='en')
            # step2. English to Japanese
            japanese_text = process_texts([english_text[0]], src_lang='en', target_lang=target_lang)
            return japanese_text[0]
        else:
            # Direct translation for other language pairs
            translated = process_texts([text], src_lang=src_lang, target_lang=target_lang)
            return translated[0]

    def inpaint_image(self, image, bbox):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        points = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

        inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return inpainted_image

    def add_text_to_image(self, image, bbox, text):
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

        # Calculate font size based on box dimensions and text_size_ratio
        font_size = int(min(box_width, box_height) * self.args.text_size_ratio / len(text) * 2)
        font_size = max(font_size, 10)  # Set minimum font size 10px

        try:
            font = ImageFont.truetype(self.args.font_family, font_size)
        except:
            font = ImageFont.load_default()

        if not isinstance(text, str):
            text = str(text[0])

        # Calculate text position (center of box)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 텍스트 위치 계산
        text_x = x_min + (box_width - text_width) / 2
        text_y = y_min + (box_height - text_height) / 2

        # Draw text
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

        # Convert back to OpenCV image
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    def enhance_image_quality(self, image):
        """(optional) Enhance image quality using Super-resolution"""
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            path = "ESPCN_x4.pb"  # Path to pre-trained model
            sr.readModel(path)
            sr.setModel("espcn", 4)  # Set model name and scale
            enhanced_image = sr.upsample(image)
            return enhanced_image
        except:
            print("Warning: Super-resolution enhancement failed, using original image")
            return image

    def process_image(self, image_path):

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(self.output_dir, f"{base_filename}.png")

        print(f"Processing image: {image_path}")

        # Step 1: Extract Korean text and bounding boxes
        image, bboxes, ko_texts = self.extract_korean_text(image_path)
        if image is None:
            return
        print(f"Found {len(ko_texts)} Korean text regions")

        # Make a copy of the original image for processing
        result_image = image.copy()

        # Step 3: Process each text region
        for i, (bbox, ko_text) in enumerate(zip(bboxes, ko_texts)):
            print(f"  Processing text region {i + 1}: '{ko_text}'")

            # Step 3-1: Inpaint the text region
            result_image = self.inpaint_image(result_image, bbox)

            # Step 3-2: Translate Korean to Japanese (via English)
            jp_text = self.translate_text(ko_text, src_lang=self.args.src_lang, target_lang=self.args.trg_lang)
            print(f"  Translated to: '{jp_text}'")

            # Step 4: Add translated text to the image
            result_image = self.add_text_to_image(result_image, bbox, jp_text)

        # Step 5 (Optional): Enhance image quality
        if self.args.enhance_quality:
            result_image = self.enhance_image_quality(result_image)

        # Step 6: Save the result
        cv2.imwrite(output_path, result_image)
        print(f"  Saved translated image to: {output_path}")

        return output_path

    def run(self):

        image_paths = self.read_images()
        if not image_paths:
            print(f"  No images found in {os.path.join(self.args.input_dir, self.args.project)}")
            return

        print(f"Found {len(image_paths)} images to process")

        # Process all images
        processed_images = []
        for image_path in image_paths:
            output_path = self.process_image(image_path)
            if output_path:
                processed_images.append(output_path)

        print(f"Successfully processed {len(processed_images)} images")
        return processed_images


if __name__ == "__main__":
    translator = Translation()
    translator.run()