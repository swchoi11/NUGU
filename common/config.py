import argparse

def parser():
    parser = argparse.ArgumentParser(description="NUGU")

    parser.add_argument('--project', type=str, choices=["translation","thumbnail"], required=True,
                            help='Choose the project type: translation or thumbnail')
    parser.add_argument('--input_data_path', type=str, default='./resource', help='Path to resource directory')
    parser.add_argument('--output_data_path', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash-001', help='Gemini model name')
    parser.add_argument('--prompt', type=str, default='thumbnail')

    # translation
    parser.add_argument('--src_lang', type=str, default='ko', help='Source language')
    parser.add_argument('--trg_lang', type=str, default='en', help='Target language')

    # thumbnail
    parser.add_argument('--product_code', type=str, required=True, help='Product code')
    parser.add_argument('--thumbnail_size', type=int, nargs=2, default=[320, 180], help='Thumbnail size [width height]')
    parser.add_argument('--select_frame', type=int, default=5, help='Frame selection for thumbnail extraction.')

    return parser.parse_args()