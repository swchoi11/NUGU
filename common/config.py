import argparse
import os
from dotenv import load_dotenv
from google.oauth2 import service_account
import vertexai

def parser():
    load_dotenv()

    ## vertex ai parm
    parser = argparse.ArgumentParser(description="NUGU")
    parser.add_argument('--project_id', type=str, default=os.getenv("PROJECT_ID"), help='vertex ai project name')
    parser.add_argument('--bucket_name', type=str, default=os.getenv("BUCKET_NAME"), help='bucket name')
    parser.add_argument('--credentials', type=str,
                        default=service_account.Credentials.from_service_account_file(os.getenv("CREDENTIALS")),
                        help='google vertex ai credentials')
    parser.add_argument('--api_key', type=str, default=os.getenv("API_KEY"), help='gemini api key')
    vertexai.init(project=os.getenv("PROJECT_ID"), location="us-central1")

    ## common arguments
    parser.add_argument('--project', type=str, default="thumbnail", choices=['translation', 'thumbnail', 'visionAnalyser'],
                        help='choose the project type: [translation, thumbnail, visionAnalyser]')
    parser.add_argument('--input_dir', type=str, default='../resource', help='path to resource directory')
    parser.add_argument('--output_dir', type=str, default='../output', help='path to output directory')
    parser.add_argument('--debug', type=bool, default=True, help='debug image ON/OFF')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash-001', help='gemini model name')
    parser.add_argument('--prompt', type=str, default='thumbnail')

    args, _ = parser.parse_known_args()

    if args.project == 'translation':
        parser.add_argument('--src_lang', type=str, default='ko', help='source language')
        parser.add_argument('--trg_lang', type=str, default='jp', help='target language')
        parser.add_argument('--font-family', type=str, default="../resource/font/FZPahyamarukana.otf", help='font family')
        parser.add_argument('--text_size_ratio', type=float, default=0.3, help='text size ratio')
        ## option
        parser.add_argument('--enhance_quality', type=bool, default=False, required=False, help='enhance image quality')

    if args.project == 'thumbnail':
        parser.add_argument('--thumbnail_size', type=int, nargs=2, default=[820, 1024], help='Thumbnail size [width height]')
        parser.add_argument('--select_frame_min', type=int, default=5, help='Frame selection for minimum thumbnail extraction')
        parser.add_argument('--select_frame_max', type=int, default=10)

        parser.add_argument('--exclude_type', type=str, default= ['마네킹 착장 이미지'],
                            choices=['모델-스튜디오','모델-연출','상품-연출', '누끼 이미지', '마네킹 착장 이미지', '옷걸이(행거) 이미지', '상품 소재 디테일 이미지'],
                            help='exclude category type')
        ## option
        parser.add_argument('--remove_background', type=bool, default=False, required=False,
                            help='remove background(누끼 이미지 생성)')

    if args.project == 'visionAnalyser':
        parser.add_argument('--product_category', type=bool, default=True, help='')
        parser.add_argument('--product_attributes', type=bool, default=False, help='')
        parser.add_argument('--mode_and_style', type=bool, default=False, help='')
        parser.add_argument('--target_user', type=bool, default=False, help='')

    return parser.parse_args()


