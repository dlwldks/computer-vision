# E01. OpenCV 실습 과제

## 0. 과제 개요

이번 과제에서는 OpenCV를 사용하여 기본적인 이미지 처리 및 마우스 이벤트 처리를 구현한다.

### 과제 구성

- **과제 1**: 이미지 불러오기 + Grayscale 변환 후 나란히 출력  
- **과제 2**: 마우스 페인팅 + 붓 크기 조절 (+ / -) + 좌/우 클릭 색상 + 드래그 연속 그리기  
- **과제 3**: 마우스 드래그로 ROI 선택 + ROI 출력 + r 리셋 + s 저장  

---

## 0-1. 실행 환경

- Windows 10 / 11  
- Python 3.x  
- OpenCV (cv2)  
- NumPy  

### 설치

pip install opencv-python numpy

---

# Problem 1 — Grayscale Conversion

## 문제

OpenCV를 사용하여 이미지를 불러오고 Grayscale로 변환한 후  
원본 이미지와 변환된 이미지를 가로로 나란히 출력하는 프로그램을 구현한다.

---

## 요구사항

- cv.imread() 를 사용하여 이미지 로드
- cv.cvtColor() 를 사용하여 Grayscale 변환
- np.hstack() 를 사용하여 원본 이미지와 Grayscale 이미지를 가로로 연결
- cv.imshow() 와 cv.waitKey() 를 사용하여 화면 출력

---

## 전체 코드

cv01_grayscale.py 파일에 구현한다.

---

## 핵심 코드

- cv.imread() 로 이미지 로드
- cv.cvtColor() 로 Grayscale 변환
- np.hstack() 으로 이미지 연결
- cv.imshow() 로 결과 출력

---

## 실행 결과

실행 결과 예시  

cv01_result.png

---

# Problem 2 — Mouse Paint

## 문제

마우스를 사용하여 이미지 위에 그림을 그릴 수 있는 프로그램을 구현한다.  
키보드 입력을 통해 붓 크기를 조절할 수 있도록 한다.

---

## 요구사항

- 초기 붓 크기: 5
- + 입력 → 붓 크기 1 증가
- - 입력 → 붓 크기 1 감소
- 붓 크기 범위: 1 ~ 15
- 좌클릭 → 파란색
- 우클릭 → 빨간색
- 드래그로 연속 그리기
- q 입력 시 종료

---

## 전체 코드

cv02_paint.py 파일에 구현한다.

---

## 핵심 코드

- cv.setMouseCallback() 로 마우스 이벤트 처리
- cv.circle() 을 사용하여 그림 그리기
- 키보드 입력을 통해 붓 크기 조절

---

## 실행 결과

실행 결과 예시  

cv02_result.png

---

# Problem 3 — ROI Selection

## 문제

사용자가 마우스를 이용해 이미지에서 관심 영역(ROI)을 선택하고  
선택한 영역을 출력하거나 저장하는 프로그램을 구현한다.

---

## 요구사항

- cv.setMouseCallback() 로 마우스 이벤트 처리
- 드래그 중 사각형 표시
- 마우스를 놓으면 ROI 추출
- r 키 → 선택 리셋
- s 키 → ROI 저장

---

## 전체 코드

cv03_roi.py 파일에 구현한다.

---

## 핵심 코드

- cv.setMouseCallback() 로 마우스 이벤트 처리
- cv.rectangle() 로 드래그 영역 표시
- 슬라이싱을 이용하여 ROI 추출
- cv.imwrite() 로 ROI 이미지 저장

---

## 실행 결과

실행 결과 예시  

cv03_result.png

---

# 실행 방법

python cv01_grayscale.py  
python cv02_paint.py  
python cv03_roi.py
