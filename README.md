# nugu & 클로잇 프로젝트

### 프로젝트 개요: 해당 프로젝트는 이미지 내 텍스트 번역 및 썸네일 추출 기능 관련 프로젝트
### 수행 기간 : 2025-05-07~2025-05-09 (3일)
### 수행 기관 : 클로잇 (PM. 류한나 수석)

## 폴더 구조
```
nugu/
    ├── README.md
    ├── main.py
    ├── common/
        ├── __init__.py
        ├── config.py
        ├── prompt.py
        ├── utils.py
    ├── resource/
        ├── 번역전후-20250429T010719Z-001/
        └── 썸네일-20250429T010606Z-001/  
    ├── thumbnail/
    └── translation/
```

### gemini credentials 설정 방법
GOOGLE Cloud Console > API 및 서비스 > 사용자 인증 정보 

좌측 상단의 '+ 사용자 인증정보 만들기' > 토글에서 API 키 선택 

생성된 api 키의 제한된 권한 등에 'Generative Language API'가 포함되어 있지 않아야 합니다. 

이후 키 내용을 확인하여 .env 파일에 삽입합니다. 


# thumbnail work process

process_image

상위 경로에 포함된 모든 이미지에 대해 가장 적합한 썸네일 후보군을 선별하는 과정입니다.
        1. 이미지 리사이즈 및 윈도우 생성 : rolling_window
        2. 윈도우 선별 및 크롭 : choose_box
        3. 분류 : box_classification

        최종 결과물 : 
        썸네일 후보군이 상위 경로 하위에 분류명으로 저장됩니다. 

        prod123 제품에 대한 최종 결과물 예시 : 
            resource/thumbnail/
            └── prod123/
            │   ├── 연출 이미지/
            │   │   ├── 모델-연출/
            │   │   │   ├── prod123_1_box_1.jpg
            │   │   │   ├── prod123_1_box_2.jpg
            │   │   │   └── prod123_2_box_3.jpg
            │   │   └── 상품-연출/
            │   │       ├── prod123_2_box_8.jpg
            │   ├── 디테일 이미지/
            │   │   ├── 누끼/
            │   │   │   └── prod123_1_box_4.jpg
            │   │   └── 마네킹/
            │   │       └── prod123_2_box_2.jpg
            ├── prod123_1.png
            ├── prod123_1_resized.png
            ├── prod123_1_windows.png
            ├── prod123_2.png
            ├── prod123_2_resized.png
            └── prod123_2_windows.png