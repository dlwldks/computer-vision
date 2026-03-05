# Computer Vision Practice Repository

컴퓨터비전 과목 실습 및 과제를 정리한 레포지토리입니다.  
OpenCV를 활용한 기본적인 이미지 처리부터 사용자 입력 기반 인터랙션, ROI 추출 등 컴퓨터비전의 핵심 개념을 단계적으로 구현합니다.

본 레포지토리는 **동아대학교 컴퓨터AI공학부 AI학과 컴퓨터비전 실습 과제**를 기반으로 작성되었습니다.

---

# Repository Structure


computer-vision
│
├─ README.md
│
├─ E01_OpenCV
│ ├─ README.md
│ ├─ cv01_grayscale.py
│ ├─ cv02_paint.py
│ ├─ cv03_roi.py
│ │
│ ├─ images
│ │ ├─ girl_laughing.jpg
│ │ └─ soccer.jpg
│ │
│ ├─ outputs
│ │
│ └─ results
│ ├─ cv01_result.png
│ ├─ cv02_paint.png
│ └─ cv03_roi.png


---

# Environment

개발 환경

- Python 3.x
- OpenCV
- NumPy
- Windows 10 / 11

필수 라이브러리 설치


pip install opencv-python numpy


---

# Assignments

## E01 - OpenCV Basic Practice

OpenCV의 기본 기능을 활용한 이미지 처리 및 사용자 입력 기반 프로그램 구현.

과제 내용

1️⃣ **Image Loading and Grayscale Conversion**

- OpenCV를 이용하여 이미지 불러오기
- 컬러 이미지를 그레이스케일로 변환
- 원본 이미지와 변환 이미지를 나란히 출력

2️⃣ **Mouse Painting Program**

- 마우스를 이용해 이미지 위에 그림 그리기
- 좌클릭 / 우클릭 색상 구분
- 드래그를 통한 연속 그리기
- 키보드를 이용한 붓 크기 조절

3️⃣ **ROI (Region of Interest) Selection**

- 마우스로 영역 선택
- 선택된 영역을 별도의 이미지로 출력
- 선택 영역 저장 기능

---

# Assignment Folder

각 과제는 별도의 폴더에서 관리됩니다.

| Assignment | Description |
|---|---|
| E01_OpenCV | OpenCV 기초 실습 |

과제 상세 설명 및 실행 결과는 아래 문서에 정리되어 있습니다.

➡ **[E01_OpenCV README](./E01_OpenCV/README.md)**

---

# Learning Objectives

본 실습을 통해 다음 개념을 학습합니다.

- OpenCV 이미지 로딩
- 색상 공간 변환
- 이미지 배열 처리
- 마우스 이벤트 처리
- 키보드 이벤트 처리
- ROI(Region of Interest) 추출
- 인터랙티브 이미지 처리 프로그램 구현

---

# Author

Computer Vision Practice  
Dong-A University  
Department of Computer AI Engineering