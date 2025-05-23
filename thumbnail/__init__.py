"""
Thumbnail Algorithm (예시)

 - 작성자: 최서우
    ** 이슈 사항 **
        gemini API 호출 시 과부하로 인해 간헐적으로 503 오류가 발생할 수 있습니다.
        해당 이슈는 config.py > --max_retries(default:5), --initial_delay(default:1) 통해 지연 시간 및 재시도 횟수를 설정하여
        일시적으로 회피해 놓은 상태입니다.

    1. 이미지 분류
      1-1.1차 분류
        - 연출 이미지: 배경과 연출 요소가 포함된 이미지
        - 디테일 이미지: 제품의 질감, 소재, 부분 확대 등 세부 정보를 중심으로 한 이미지
      1-2. 2차 분류: 연출 및 디테일 이미지 각각에 대해 아래 세부 유형으로 추가 분류
        - 모델-스튜디오 / 모델-연출 / 상품-연출 / 누끼 이미지 (배경 제거된 클린컷)/ 마네킹 착장 이미지/ 옷걸이(행거) 이미지/ 상품 소재 디테일 이미지
          제외 유형 설정 (default: exclude_type = ['마네킹 착장 이미지'])
        - (옵션) 제품 컷에 대한 누끼 이미지 생성
    2. 썸네일 후보 구간 탐색
      2-1. 통이미지를 가로 820px 기준의 동일 비율로 리사이즈하여 썸네일 규격으로 통일합니다.
      2-2. 리사이즈된 이미지에 가로 820px 크기의 고정된 윈도우(mask)를 적용해 이동시키며 썸네일 후보가 될 bounding box 좌표를 추출합니다.
      2-3. 각 이미지 내 주요 객체(예: 상품, 모델 등)를 기준 으로 시각적 중심 또는 강조 포인트 영역을 식별 합니다.
      2-4. 가로 세로 비율, 객체 크기, 구도 등을 고려하여 썸네일로 적합한 프레임을 자동 추출합니다.
    3. 썸네일 이미지 생성
      3-1. 추출된 프레임은 지정된 해상도(default: 820X1204)에 맞춰 리사이즈 또는 크롭 처리 됩니다.
      3-2. 썸네일 추출 개수를 정의 합니다.(default: select_frame=5)

 (option) visionAnalyzer
 - 작성자: 류한나
     * 상세 이미지 분석과 관련된 속성 값은 common > feature_map.py에 사전 정의된 딕셔너리 형태로 정의 된다.
     1. 제품 단위(폴더 기준)로 각 이미지에 대해 사람(모델) 포함 여부를 분류 한다.
     2. 사람 미포함(제품 단독) 이미지에 대해서만 제품 속성(product_category)값을 추출 한다.
     3. 이후 추출 대상 속성(product_category, product_attributes, mode_and_style, target_user) 중 type=True 설정된 항목에 대해,
        각 딕셔너리 value(속성 설명)을 기반으로 gemini가 가장 유사도가 높은 key(속성 값)를 추론 한다.
     4. 최종 추출된 속성 값은 제품명(폴더명), 파일명, product_category, product_attributes, mode_and_style, target_user 컬럼 형태로
        (default: ../resource/visionAnalyzer.xlsx) 엑셀 파일로 저장 된다.
"""