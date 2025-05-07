"""
Translation Algorithm (예시)

 - 작성자: 엄다빈

1. 한글 텍스트 출력: 입력된 이미지 내에서 한글(ko) 텍스트를 인식 하고 각 텍스트에 대한 바운딩 박스 좌표도 함께 추출 한다.
   left_top = 좌측 상단(x_min, y_mim), right_bottom = 우측 하단(x_max, y_max)
3. 번역 프로세스
   추출된 박스 영역(left_top, right_bottom) 및 추출된 박스 영역 내 한글(ko) 대해
   3-1. source 이미지에서 추출된 박스 영역을 마스킹(masking) 한 뒤 해당 영역 inpainting 한다.
   3-2. 추출된 한글(ko)-> 영어(en)-> 일본어(jp) 순서로 번역 한다.
   3-3. 번역된 일본어 텍스트의 기본 글꼴은(default: font-family: sans-serif)이며 사용자 지정 폰트 설정도 가능 하다.
   3-4. 번역된 일본어 폰트 사이즈는 기본 바운딩 박스 사이즈 크기에 80% 영역( default: 80% )이며 사용자 지정 설정 가능 하다.
4. 번역된 일본어는 inpainting 된 영역(masking)의 정중앙에 위치 하도록 삽입 한다.
5. 번역이 완료된 이미지는 (default: outDir/raw_image.png)경로에 저장 되며 사용자가 지정 설정 가능 하다.
   5-1. (옵션) 이미지 품질이 낮을 경우, Super-resolution 알고리즘 활용하여 화질을 향상 시킬 수 있다.

"""