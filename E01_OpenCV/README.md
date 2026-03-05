# E01. OpenCV 실습 과제

## 0) 과제 개요
- 과제 1: 이미지 불러오기 + 그레이스케일 변환 후 나란히 출력
- 과제 2: 마우스 페인팅 + 붓 크기 조절(+/-) + 좌/우 클릭 색상 + 드래그 연속 그리기
- 과제 3: 마우스 드래그로 ROI 선택 + ROI 출력 + r 리셋 + s 저장

## 0-1) 실행 환경
- Windows 10/11
- Python 3.x
- OpenCV (cv2)
- numpy

설치: pip install opencv-python numpy
```bash
Problem 1 — Grayscale Conversion
문제

OpenCV를 사용하여 이미지를 불러오고 Grayscale로 변환한 후
원본 이미지와 변환된 이미지를 가로로 나란히 출력하는 프로그램을 구현한다.

요구사항

cv.imread() 를 사용하여 이미지 로드

cv.cvtColor() 를 사용하여 Grayscale 변환

np.hstack() 를 사용하여 원본 이미지와 Grayscale 이미지를 가로로 연결

cv.imshow() 와 cv.waitKey() 를 사용하여 화면 출력

전체 코드
# cv01_grayscale.py

import os
import sys
import argparse
import cv2 as cv
import numpy as np
try:
    import tkinter as tk
except Exception:
    tk = None


def main():
    parser = argparse.ArgumentParser(description="원본 이미지와 그레이스케일 이미지를 나란히 표시합니다.")
    default_image = os.path.join(os.path.dirname(__file__), "images", "soccer.jpg")
    parser.add_argument("image", nargs="?", default=default_image)
    args = parser.parse_args()

    img = cv.imread(args.image)
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        sys.exit(1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray_color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    result = np.hstack((img, gray_color))

    cv.imshow("Original | Grayscale", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
핵심 코드
img = cv.imread(args.image)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray_color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

result = np.hstack((img, gray_color))

cv.imshow("Original | Grayscale", result)
실행 결과

(여기에 실행 결과 이미지 삽입)

[ cv01_result.png ]
Problem 2 — Mouse Paint
문제

마우스를 사용하여 이미지 위에 그림을 그릴 수 있는 프로그램을 구현한다.
키보드 입력을 통해 붓 크기를 조절할 수 있도록 한다.

요구사항

초기 붓 크기 5

+ 입력 → 붓 크기 1 증가

- 입력 → 붓 크기 1 감소

붓 크기 범위 1 ~ 15

좌클릭 → 파란색

우클릭 → 빨간색

드래그로 연속 그리기

q 입력 시 종료

전체 코드
# cv02_paint.py

import cv2 as cv
import numpy as np

drawing = False
brush_size = 5
brush_min = 1
brush_max = 15
color = (255, 0, 0)
img = None
win_name = "Paint"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def on_mouse(event, x, y, flags, param):
    global drawing, color, img

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        color = (255, 0, 0)
        cv.circle(img, (x, y), brush_size, color, -1)

    elif event == cv.EVENT_RBUTTONDOWN:
        drawing = True
        color = (0, 0, 255)
        cv.circle(img, (x, y), brush_size, color, -1)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.circle(img, (x, y), brush_size, color, -1)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

    elif event == cv.EVENT_RBUTTONUP:
        drawing = False


def main():
    global img, brush_size

    src = cv.imread("images/soccer.jpg")
    img = src.copy()

    cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, on_mouse)

    while True:
        cv.imshow(win_name, img)

        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('+'):
            brush_size += 1
            brush_size = clamp(brush_size, brush_min, brush_max)

        if key == ord('-'):
            brush_size -= 1
            brush_size = clamp(brush_size, brush_min, brush_max)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
핵심 코드
cv.setMouseCallback(win_name, on_mouse)

cv.circle(img, (x, y), brush_size, color, -1)

key = cv.waitKey(1) & 0xFF

if key == ord('+'):
    brush_size += 1

if key == ord('-'):
    brush_size -= 1
실행 결과

(여기에 실행 결과 이미지 삽입)

[ cv02_result.png ]
Problem 3 — ROI Selection
문제

사용자가 마우스를 이용해 이미지에서 관심 영역(ROI) 을 선택하고
선택한 영역을 별도로 출력하거나 저장하는 프로그램을 구현한다.

요구사항

cv.setMouseCallback() 로 마우스 이벤트 처리

드래그 중 사각형 표시

마우스를 놓으면 ROI 추출

r 키 → 선택 리셋

s 키 → ROI 저장

전체 코드
# cv03_roi.py

import os
import time
import cv2 as cv
import numpy as np

start_pt = None
end_pt = None
dragging = False
roi_img = None
src = None
display = None

win_name = "ROI"


def normalize_rect(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    return left, top, right, bottom


def on_mouse(event, x, y, flags, param):
    global start_pt, end_pt, dragging, roi_img, display

    if event == cv.EVENT_LBUTTONDOWN:
        start_pt = (x, y)
        end_pt = (x, y)
        dragging = True

    elif event == cv.EVENT_MOUSEMOVE:
        if dragging:
            end_pt = (x, y)
            display = src.copy()
            l, t, r, b = normalize_rect(start_pt, end_pt)
            cv.rectangle(display, (l, t), (r, b), (0, 255, 0), 2)

    elif event == cv.EVENT_LBUTTONUP:
        dragging = False
        end_pt = (x, y)

        l, t, r, b = normalize_rect(start_pt, end_pt)

        roi_img = src[t:b, l:r]
        cv.imshow("ROI", roi_img)


def main():
    global src, display

    src = cv.imread("images/soccer.jpg")
    display = src.copy()

    cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, on_mouse)

    while True:
        cv.imshow(win_name, display)

        key = cv.waitKey(1) & 0xFF

        if key == ord('r'):
            display = src.copy()

        if key == ord('s'):
            if roi_img is not None:
                cv.imwrite("roi.png", roi_img)

        if key == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
핵심 코드
cv.setMouseCallback(win_name, on_mouse)

cv.rectangle(display, (l, t), (r, b), (0, 255, 0), 2)

roi_img = src[t:b, l:r]

cv.imshow("ROI", roi_img)

cv.imwrite("roi.png", roi_img)
실행 결과

(여기에 실행 결과 이미지 삽입)

[ cv03_result.png ]


실행 방법
python cv01_grayscale.py
python cv02_paint.py
python cv03_roi.py
참고: 폴더 구조(권장)
E01_OpenCV/
  cv01_grayscale.py
  cv02_paint.py
  cv03_roi.py
  images/
    soccer.jpg
    girl_laughing.jpg
  outputs/
  results/
    (실행 결과 스크린샷 저장 후 README에 첨부)
