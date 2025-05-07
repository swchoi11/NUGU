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