# from common.config import parser
# from thumbnail.thumbnail import Image

# def run_translation():
#     return print("run_translation")

# def run_thumbnail():
#     return print("run_thumbnail")




import sys
from thumbnail.ThumbnailProcessor import ThumbnailProcessor

if __name__ == "__main__":

    sys.argv.extend([
        '--project', 'thumbnail',
        '--model', 'gemini-2.0-flash-001',
        '--select_frame', '5'
    ])
    
    t_processor = ThumbnailProcessor('./resource/thumbnail/DGSWKE6166')
    t_processor.process_image(exclude_category=[])
