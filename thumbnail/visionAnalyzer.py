import os
import glob

import cv2
import pandas as pd

from common import feature_map as fm
from common.config import parser

class VisionAnalyzer:
    def __init__(self):
        self.args = parser()
        self.args.input_dir = '../resource/thumbnail'
        self.columns = ['product_code', 'filename', 'product_category', 'product_attributes', 'mode_and_style', 'target_user']

    def read_images(self, extensions=['jpg', 'jpeg', 'png'])->list:
        root_dir = os.path.join(self.args.input_dir, self.args.project)
        image_list = []
        for ext in extensions:
            image_list.extend(glob.glob(os.path.join(root_dir, f"**/*.{ext}"), recursive=True))

        return image_list

    def product_category(self, image):
        return print("product_category")

    def product_attributes(self, image):
        return print("product_attributes")

    def mode_and_style(self, image):
        return print("mode_and_style")

    def target_user(self, image):
        return print("target_user")

    def run(self):
        image_list = self.read_images()
        rows = {}
        for image_path in image_list:
            product_code = os.path.dirname(image_path).split('//')[-1]
            filename = os.path.basename(image_path)
            image = cv2.imread(image_path)
            print(image_path)
            row ={
                "product_code": product_code,
                "filename": filename,
                "product_category":  self.product_category(image),
                "product_attributes": self.product_attributes(image),
                "mode_and_style":   self.mode_and_style(image),
                "target_user":      self.target_user(image),
            }

        df = pd.DataFrame(rows, columns=self.columns)
        df.to_excel(f"./visionAnalyzer.xlsx", index=False)
        return print("run")





