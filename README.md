# nugu & 클로잇 프로젝트

- 프로젝트 목표: 해당 프로젝트는 이미지 내 텍스트 번역 및 썸네일 추출 기능 관련 tutorial 코드 제공
- 수행 기간 : 2025-05-07~2025-05-09 (3일)
- 수행 기관 : 클로잇 (문의처: 류한나(hnryu@itcen.com) )

### 폴더 구조
```
nugu/
    ├── README.md
    ├── requirements.txt
    ├── .env
    ├── common/
    |   ├── __init__.py
    |   ├── config.py
    |   ├── feature_map.py
    |   ├── path_manager.py : (tumbnail) 관련 경로 설정 
    |   ├── prompt.py
    |   └── utils.py
    ├── output/
    |    ├── translation
    |    |   └── DMEIBQ1786
    |    |           ├── bbox: 원본 이미지 내 OCR 영역의 바운딩 박스 및 텍스트 추출
    |    |           |    ├── DMEIBQ1786_ko_001.png
    |    |           |    └── DMEIBQ1786_ko_001_text.txt
    |    |           ├── txt : ko->jp 번역 결과 pair 저장
    |    |           |    └── DMEIBQ1786_ko_001_translations.txt
    |    |           └── DMEIBQ1786_ko_001.png (최종 결과물)
    |    └── thumbnail
    |        └── DGSWKE6166
    |                ├── final (최종 결과물)
    |                |    ├── thumbnail_연출이미지_모델-스튜디오_DGSWKE6166_2_box_0.png
    |                |    ├── thumbnail_연출이미지_모델-연출_DGSWKE6166_1_box_0.png
    |                |    └── thumbnail_연출이미지_모델-연출_DGSWKE6166_1_box_1.png
    |                ├── segment(config > --is_visual=True 인 경우, 실시간 이미지 출력 확인 가능)
    |                |    ├── DGSWKE6166_1.png
    |                |    ├── DGSWKE6166_2.png
    |                |    └── DGSWKE6166_3.png   
    |                ├── 디테일 이미지
    |                └── 연출 이미지
    ├── resource/
    |    ├── font
    |    |     └── NotoSansJP-Regular.ttf
    |    ├── translation : 원본 이미지 
    |    |     ├── DMEIBQ1786 : 제품코드  
    |    |           ├── DMEIBQ1786_ko_001.jpg
    |    |           ├── DMEIBQ1786_ko_002.jpg
    |    |           └── DMEIBQ1786_ko_002.jpg
    |    └── thumbnail : 원본 이미지
    |          └── DGSWKE6166
    |                ├── DGSWKE6166_1.png
    |                ├── DGSWKE6166_2.png
    |                └── DGSWKE6166_3.png 
    ├── translation/
    |    ├── __init__.py : 알고리즘 설명  
    |    ├── gemini.py
    |    └── trnaslation.py : main 코드 
    └── thumbnail
         ├── __init__.py : 알고리즘 설명  
         ├── segment.py
         └── thumbnail.py : main 코드 
    
```
### 환경 설정
### Translation 
### Thumbnail


