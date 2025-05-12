import os
import re

from pathlib import Path
from common import init_logger
from common.config import parser


class PathManager:
    def __init__(self, product_path):
        self.logger = init_logger()
        self.args = parser()
        self.input_dir = Path(self.args.input_dir)
        self.output_dir = Path(self.args.output_dir)
        self.project = self.args.project
        self.product_path = product_path

        # 기본 디렉토리 설정
        self.base_input_dir = self.input_dir / self.project
        self.base_output_dir = self.output_dir / self.project
        self.product_code = self.extract_product_code()

        # 결과물 디렉토리 설정
        self.product_input_dir = self.base_input_dir / self.product_code
        self.product_output_dir = self.base_output_dir / self.product_code
        self.resize_dir =  self.base_output_dir  / self.product_code / 'resize'
        self.segment_dir = self.base_output_dir / self.product_code / 'segment'
        self.final_output = self.product_output_dir / 'final'

    def extract_product_code(self):
        """
            이미지의 상위 폴더명을 기반으로 제품 코드를 추출 합니다.
            향후 제품 코드 명 규칙이 변경될 경우, 해당 로직을 수정해야 합니다.
            영문과 숫자만 남기고 한글, 특수문자 등 다른 문자는 삭제하여 폴더링 합니다.

        """
        root_dir = os.path.dirname(self.product_path)
        file_name = os.path.basename(self.product_path)
        clean_file_name = re.sub(r'[^a-zA-Z0-9]', '', file_name)

        os.rename(self.product_path, os.path.join(root_dir, clean_file_name))
        self.logger.info(f"파일명 정리 : {self.product_path} -> {os.path.join(root_dir, clean_file_name)}")

        # 제품 코드 추출
        return clean_file_name