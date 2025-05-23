"""
Translation Algorithm (예시)

 - 작성자: 엄다빈

1. 한글 텍스트 출력: 입력된 이미지 내에서 한글(ko) 텍스트를 인식 하고 각 텍스트에 대한 바운딩 박스 좌표도 함께 추출 한다.
   left_top = 좌측 상단(x_min, y_mim), right_bottom = 우측 하단(x_max, y_max)
3. 번역 프로세스
   추출된 박스 영역(left_top, right_bottom) 및 추출된 박스 영역 내 한글(ko) 대해
   3-1. source 이미지에서 추출된 박스 영역을 마스킹(masking) 한 뒤 해당 영역 inpainting 한다.
   3-2. 추출된 한글(ko)-> 영어(en)-> 일본어(jp) 순서로 번역 한다.
   3-3. 번역된 일본어 텍스트의 기본 글꼴은(default: font-family: NotoSansJP-Regular.ttf)이며 사용자 지정 폰트 설정도 가능 하다.
   3-4. 번역된 일본어 폰트 사이즈는 기본 바운딩 박스 사이즈 크기에 70% 영역( default: 70% )이며 사용자 지정 설정 가능 하다.
        * 추가 개발 사항 *
          한글 글자수에 일본어 번역 글자수 길이가 가변적이며 그에 따른 후처리 알고리즘 추가 구현 필요
4. 번역된 일본어는 inpainting 된 영역(masking)의 정중앙에 위치 하도록 삽입 한다.
        * 추가 개발 사항 *
          현재는 각 바운딩 박스에 따른 글자에 대해 번역 -> 개별 박스에 따른 번역 진행후 붙여 쓰기를 하지만 문장 단위로 값을 가지고 와서
          바운딩 박스 길이에 맞게 split 하여 붙여넣는 형태로 추가 알고리즘 구현 필요

5. 번역이 완료된 이미지는 (default: outDir/raw_image.png)경로에 저장 되며 사용자가 지정 설정 가능 하다.
   5-1. (옵션) 이미지 품질이 낮을 경우, Super-resolution 알고리즘 활용하여 화질을 향상 시킬 수 있다.
        * 추가 개발 사항 *
          현재는 오픈소스(import 가능한) super-resolution 추가 해놓았습니다. 테스트 진행 하신 이후에 필요에 따라
          알고리즘 및 모델 변경 하여 사용하시면 됩니다. imagen3 이용한 super-resolution 테스트 추가 필요합니다.


"""