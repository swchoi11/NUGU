from common.config import parser
from common.thumbnail.thumbnail import Image

def run_translation():
    return print("run_translation")

def run_thumbnail():
    return print("run_thumbnail")

args = parser()
image_path = args.input_data_path + '/' + args.project + '/' + args.product_code + "/"
print(image_path)
image = Image(image_path = image_path, ext="png")
image.process_image()