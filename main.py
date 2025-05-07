# from common.config import parser
# from thumbnail.thumbnail import Image

# def run_translation():
#     return print("run_translation")

# def run_thumbnail():
#     return print("run_thumbnail")


# import cv2
# import numpy as np

# def split_obj(image):
#     image = cv2.imread(image)

#     if image is None:
#         raise ValueError("Image is None")
    
#     h, w = image.shape[:2]

#     saliency = cv2.saliency.StaticSaliencyFineGrained_create()
#     (success, saliencyMap) = saliency.computeSaliency(image)

#     saliencyMap = (saliencyMap * 255).astype("uint8")

#     _, binary_map = cv2.threshold(saliencyMap, 50, 255, cv2.THRESH_BINARY)

#     contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for idx, contour in enumerate(contours):
#         if cv2.contourArea(contour) < 2000:
#             continue

#         x,y,w,h = cv2.boundingRect(contour)

#         cropped = image[y:y+h, x:x+w]

#         cv2.imwrite(f'./resource/thumbnail/DAOXGZ5738/cropped_{idx}.png', cropped)


# split_obj('./resource/thumbnail/DAOXGZ5738/DAOXGZ5738_2.png')

from thumbnail.thumbnail import Image

image = Image('./resource/thumbnail/DOYMAW9090/DOYMAW9090_3.png',".")

# result = image.get_segment('./resource/thumbnail/DOYMAW9090/DOYMAW9090_3.png')

# print(result)

# import cv2

# image = cv2.imread('./resource/thumbnail/DOYMAW9090/resized.png')


# for item in result:
#     x, y = item['segment']

#     cv2.rectangle(image, (0, int(x)), (200, int(y)), (0, 0, 255), 2)

#     cv2.imwrite(f'./resource/thumbnail/DOYMAW9090/cropped_{item["index"]}.png', image)


# image.rolling_segment('./resource/thumbnail/DOYMAW9090/DOYMAW9090_3.png', 8)

print(image.choose_image())