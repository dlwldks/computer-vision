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

# cv01_grayscale.py
# OpenCV로 이미지를 불러와 그레이스케일 변환 후
# 원본 이미지와 나란히 출력하는 예제

import os                    # 파일 및 경로 처리를 위한 표준 라이브러리
import sys                   # 프로그램 종료 등 시스템 관련 기능을 위한 라이브러리
import argparse              # 명령줄 인자(argument)를 처리하기 위한 라이브러리
import cv2 as cv             # OpenCV 라이브러리 (이미지 처리)
import numpy as np           # 배열 연산 및 이미지 결합(np.hstack) 사용
try:                         # tkinter가 사용 가능한지 확인하기 위한 예외 처리
    import tkinter as tk     # 화면 해상도 확인을 위해 tkinter 사용
except Exception:            # tkinter가 설치되지 않았거나 사용 불가능할 경우
    tk = None                # tk를 None으로 설정하여 이후 코드에서 처리


def main():                  # 프로그램의 메인 함수 정의
    parser = argparse.ArgumentParser(description="원본 이미지와 그레이스케일 이미지를 나란히 표시합니다.")  # 명령줄 인자 파서 생성
    default_image = os.path.join(os.path.dirname(__file__), "images", "soccer.jpg")  # 기본 이미지 경로 설정
    parser.add_argument("image", nargs="?", default=default_image, help=f"불러올 이미지 경로 (기본: {default_image})")  # 명령줄에서 이미지 경로 입력 가능
    args = parser.parse_args()  # 입력된 명령줄 인자를 파싱하여 args 객체에 저장

    # 1) 이미지 로드 (OpenCV는 BGR 형식으로 읽음)
    img = cv.imread(args.image)  # cv.imread()를 사용하여 지정된 경로의 이미지를 읽어옴
    if img is None:              # 이미지 로드 실패 여부 확인
        print(f"이미지를 불러올 수 없습니다: {args.image}")  # 오류 메시지 출력
        sys.exit(1)              # 프로그램을 종료

    # 2) 그레이스케일 변환 (cv.COLOR_BGR2GRAY 사용)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 컬러(BGR) 이미지를 그레이스케일 이미지로 변환

    # 그레이스케일은 1채널이므로 원본과 가로로 연결하려면 3채널로 변환
    gray_color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)  # 그레이스케일 이미지를 다시 3채널(BGR) 형태로 변환

    # 3) np.hstack()으로 가로 연결
    result = np.hstack((img, gray_color))  # 원본 이미지와 그레이스케일 이미지를 가로로 결합

    # 4) 화면 크기에 맞춰 자동 축소 후 표시, 아무 키나 누르면 닫힘
    # 화면 크기 확인 (tkinter 사용, 실패하면 기본값 사용)
    margin = 100  # 화면 가장자리 여유 공간 설정
    if tk is not None:  # tkinter 사용 가능 여부 확인
        root = tk.Tk()  # tkinter 루트 윈도우 생성
        root.withdraw()  # tkinter 창을 화면에 표시하지 않도록 숨김
        screen_w = root.winfo_screenwidth()  # 현재 모니터의 화면 너비 가져오기
        screen_h = root.winfo_screenheight()  # 현재 모니터의 화면 높이 가져오기
        root.destroy()  # tkinter 창 객체 제거
    else:  # tkinter를 사용할 수 없는 경우
        screen_w, screen_h = 1280, 720  # 기본 화면 크기 값을 설정

    res_h, res_w = result.shape[0], result.shape[1]  # 결과 이미지의 높이와 너비 추출
    scale = min(1.0, (screen_w - margin) / res_w, (screen_h - margin) / res_h)  # 화면에 맞도록 축소 비율 계산
    if scale < 1.0:  # 이미지가 화면보다 클 경우
        new_w = int(res_w * scale)  # 축소된 너비 계산
        new_h = int(res_h * scale)  # 축소된 높이 계산
        result_show = cv.resize(result, (new_w, new_h), interpolation=cv.INTER_AREA)  # 이미지 크기 축소
    else:  # 이미지가 화면보다 작거나 같을 경우
        result_show = result  # 원본 결과 이미지를 그대로 사용

    cv.namedWindow("Original | Grayscale", cv.WINDOW_NORMAL)  # 창 크기를 조절할 수 있는 OpenCV 창 생성
    cv.imshow("Original | Grayscale", result_show)  # 결과 이미지를 화면에 표시
    cv.waitKey(0)  # 아무 키나 입력될 때까지 프로그램 대기
    cv.destroyAllWindows()  # 모든 OpenCV 창 닫기


if __name__ == "__main__":  # 현재 파일이 직접 실행된 경우
    main()                  # main 함수 실행

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

## cv02_paint.py

```python
# cv02_paint.py
# 과제 02: 마우스 입력으로 이미지 위에 붓질 + 키보드로 붓 크기 조절

import cv2 as cv
import numpy as np

# ---- 전역 상태(마우스 콜백에서 접근해야 함) ----
drawing = False
brush_size = 5
brush_min = 1
brush_max = 15
color = (255, 0, 0)
img = None
win_name = "Paint (+/- brush, q quit)"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def on_mouse(event, x, y, flags, param):
    global drawing
    global color
    global img
    global brush_size

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
    global img
    global brush_size

    src = cv.imread("images/soccer.jpg")
    if src is None:
        print("이미지를 불러올 수 없습니다. 경로를 확인하세요: images/soccer.jpg")
        return

    img = src.copy()

    cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, on_mouse)

    while True:
        preview = img.copy()
        cv.putText(preview, f"Brush: {brush_size}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv.imshow(win_name, preview)

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


---

```markdown
## cv03_roi.py

```python
# cv03_roi.py
# 과제 03: 마우스로 영역 선택(드래그 사각형) + ROI 출력 + r 리셋 + s 저장

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
win_name = "ROI Select (drag) | r:reset, s:save, q:quit"

def normalize_rect(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    return left, top, right, bottom

def on_mouse(event, x, y, flags, param):
    global start_pt
    global end_pt
    global dragging
    global roi_img
    global display
    global src

    if event == cv.EVENT_LBUTTONDOWN:
        start_pt = (x, y)
        end_pt = (x, y)
        dragging = True
        roi_img = None

    elif event == cv.EVENT_MOUSEMOVE:
        if dragging:
            end_pt = (x, y)
            display = src.copy()
            l, t, r, b = normalize_rect(start_pt, end_pt)
            cv.rectangle(display, (l, t), (r, b), (0, 255, 0), 2)

    elif event == cv.EVENT_LBUTTONUP:
        if dragging:
            dragging = False
            end_pt = (x, y)
            l, t, r, b = normalize_rect(start_pt, end_pt)

            if (r - l) >= 2 and (b - t) >= 2:
                roi_img = src[t:b, l:r].copy()
                cv.imshow("ROI", roi_img)

            display = src.copy()
            cv.rectangle(display, (l, t), (r, b), (0, 255, 0), 2)

def main():
    global src
    global display
    global start_pt
    global end_pt
    global roi_img
    global dragging

    base_dir = r"D:\computer-vision\E01_OpenCV"
    img_path = os.path.join(base_dir, "images", "soccer.jpg")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    src = cv.imread(img_path)
    if src is None:
        print(f"[ERROR] 이미지 로드 실패: {img_path}")
        return

    display = src.copy()

    cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, on_mouse)

    while True:
        cv.imshow(win_name, display)
        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('r'):
            start_pt = None
            end_pt = None
            dragging = False
            roi_img = None
            display = src.copy()
            try:
                cv.destroyWindow("ROI")
            except:
                pass

        if key == ord('s'):
            if roi_img is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(out_dir, f"roi_{ts}.png")
                ok = cv.imwrite(save_path, roi_img)
                if ok:
                    print(f"[SAVED] {save_path}")
                else:
                    print("[ERROR] ROI 저장 실패")
            else:
                print("[INFO] 저장할 ROI가 없음 (먼저 드래그로 ROI 선택)")

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

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
