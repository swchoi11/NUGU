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
                        help='google credentials')

    vertexai.init(project=os.getenv("PROJECT_ID"), location="us-central1")

    ## common arguments
    parser.add_argument('--project', type=str, default="translation",
                        help='choose the project type: translation or thumbnail')  # 기본값 설정
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
        parser.add_argument('--text_size_ratio', type=float, default=0.8, help='text size ratio')
        parser.add_argument('--enhance_quality', type=bool, default=False, help='enhance image quality')

    if args.project == 'thumbnail':
        parser.add_argument('--thumbnail_size', type=int, nargs=2, default=[320, 180],
                                      help='Thumbnail size [width height]')
        parser.add_argument('--select_frame', type=int, default=5,
                                      help='Frame selection for minimum thumbnail extraction.')

    return parser.parse_args()
