# E01. OpenCV 실습 과제

## 0. 과제 개요

이번 과제에서는 OpenCV를 사용하여 기본적인 이미지 처리 및 마우스 이벤트 처리를 구현합니다.

---

## 요구사항 및 설치

- Python 3.7 이상
- OpenCV (`opencv-python`)
- NumPy (`numpy`)

설치

```bash
pip install opencv-python numpy
```

---

## 폴더 구조 (요약)

```
E01_OpenCV/
│
├── cv01_grayscale.py
├── cv02_paint.py
├── cv03_roi.py
│
├── images/
│   └── soccer.jpg
│
└── outputs/
    ├── cv01_result.png
    ├── cv02_result.png
    └── cv03_result.png
```

- `cv01_grayscale.py` : 이미지 불러오기 + 그레이스케일 변환
- `cv02_paint.py` : 마우스 페인팅
- `cv03_roi.py` : ROI 선택 및 저장

---

# 실행 방법

```bash
python cv01_grayscale.py
python cv02_paint.py
python cv03_roi.py
```

또는

```bash
python E01_OpenCV/cv01_grayscale.py
python E01_OpenCV/cv02_paint.py
python E01_OpenCV/cv03_roi.py
```

---

# Problem 1 — Grayscale Conversion

이미지를 불러온 후 Grayscale로 변환하고  
원본 이미지와 나란히 출력합니다.

---

## 실행 결과

<figure>
  <img src="outputs/cv01_result.png" alt="Grayscale 결과" width="400"/>
  <figcaption>원본 이미지와 Grayscale 변환 결과</figcaption>
</figure>

---

<details>
<summary>전체 코드 — cv01_grayscale.py</summary>

```python
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
```

</details>

---

# Problem 2 — Mouse Paint

마우스로 이미지 위에 그림을 그리는 프로그램입니다.

기능

- 좌클릭 : 파란색 그리기
- 우클릭 : 빨간색 그리기
- `+` : 붓 크기 증가
- `-` : 붓 크기 감소
- `q` : 종료

---

## 실행 결과

<figure>
  <img src="outputs/cv02_result.png" alt="페인팅 결과" width="400"/>
  <figcaption>마우스로 직접 그린 그림 예시</figcaption>
</figure>

---

<details>
<summary>전체 코드 — cv02_paint.py</summary>

```python
# cv02_paint.py
# 과제 02: 마우스 입력으로 이미지 위에 붓질 + 키보드로 붓 크기 조절

import cv2 as cv            # OpenCV 사용
import numpy as np          # 이미지 복사 및 배열 처리를 위해 사용(필수는 아니지만 유지)

# ---- 전역 상태(마우스 콜백에서 접근해야 함) ----
drawing = False             # 드래그(그리기) 중인지 여부
brush_size = 5              # 초기 붓 크기 (요구사항: 5)
brush_min = 1               # 붓 크기 최소 (요구사항: 1)
brush_max = 15              # 붓 크기 최대 (요구사항: 15)
color = (255, 0, 0)         # 기본 색 (파란색, BGR)
img = None                  # 그림이 그려질 이미지
win_name = "Paint (+/- brush, q quit)"  # 창 이름

def clamp(v, lo, hi):       # 붓 크기 제한 함수
    return max(lo, min(hi, v))  # lo~hi 범위로 제한

def on_mouse(event, x, y, flags, param):  # 마우스 이벤트 콜백
    global drawing          # 전역 drawing 사용
    global color            # 전역 color 사용
    global img              # 전역 img 사용
    global brush_size       # 전역 brush_size 사용 (명시적으로)

    if event == cv.EVENT_LBUTTONDOWN:             # 좌클릭 다운
        drawing = True                            # 그리기 시작
        color = (255, 0, 0)                       # 파란색(BGR)
        cv.circle(img, (x, y), brush_size, color, -1)  # 원을 채워서 그림

    elif event == cv.EVENT_RBUTTONDOWN:           # 우클릭 다운
        drawing = True                            # 그리기 시작
        color = (0, 0, 255)                       # 빨간색(BGR)
        cv.circle(img, (x, y), brush_size, color, -1)  # 원을 채워서 그림

    elif event == cv.EVENT_MOUSEMOVE:             # 마우스 이동
        if drawing:                               # 드래그 중이면
            cv.circle(img, (x, y), brush_size, color, -1)  # 연속 그리기

    elif event == cv.EVENT_LBUTTONUP:             # 좌클릭 업
        drawing = False                           # 그리기 종료

    elif event == cv.EVENT_RBUTTONUP:             # 우클릭 업
        drawing = False                           # 그리기 종료

def main():                         # 메인 함수
    global img                       # 전역 img 사용
    global brush_size                # 전역 brush_size 사용

    # 이미지 로드 (상대경로: E01_OpenCV 폴더에서 실행한다고 가정)
    src = cv.imread("images/soccer.jpg")           # 이미지 불러오기
    if src is None:                                # 로드 실패 체크
        print("이미지를 불러올 수 없습니다. 경로를 확인하세요: images/soccer.jpg")
        return                                     # 종료

    img = src.copy()                               # 복사본에 그림 그리기

    cv.namedWindow(win_name)                       # 창 생성
    cv.setMouseCallback(win_name, on_mouse)        # 마우스 콜백 등록 (힌트)

    while True:                                    # 키 입력을 위한 루프 (힌트)
        preview = img.copy()                       # 안내 텍스트용 복사
        cv.putText(preview, f"Brush: {brush_size}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)  # 붓 크기 표시

        cv.imshow(win_name, preview)               # 화면 출력

        key = cv.waitKey(1) & 0xFF                 # 1ms 대기하며 키 입력 받기 (힌트)

        if key == ord('q'):                        # q 누르면 종료 (요구사항)
            break                                  # 루프 탈출

        if key == ord('+'):                        # + 입력 시 붓 크기 증가 (요구사항)
            brush_size += 1                        # 1 증가
            brush_size = clamp(brush_size, brush_min, brush_max)  # 1~15 제한

        if key == ord('-'):                        # - 입력 시 붓 크기 감소 (요구사항)
            brush_size -= 1                        # 1 감소
            brush_size = clamp(brush_size, brush_min, brush_max)  # 1~15 제한

    cv.destroyAllWindows()                         # 창 닫기

if __name__ == "__main__":          # 직접 실행 시
    main()                          # main 실행
```

</details>

---

# Problem 3 — ROI Selection

마우스 드래그를 이용하여 관심 영역(ROI)을 선택하고 저장하는 프로그램입니다.

기능

- 드래그 : ROI 선택
- `r` : 리셋
- `s` : ROI 저장
- `q` : 종료

---

## 실행 결과

<figure>
  <img src="outputs/cv03_result.png" alt="ROI 결과" width="400"/>
  <figcaption>마우스로 선택한 ROI 영역</figcaption>
</figure>

---

<details>
<summary>전체 코드 — cv03_roi.py</summary>

```python
# cv03_roi.py
# cv03_roi.py
# 과제 03: 마우스로 영역 선택(드래그 사각형) + ROI 출력 + r 리셋 + s 저장

import os  # 경로 처리
import time  # 파일명 타임스탬프
import cv2 as cv  # OpenCV
import numpy as np  # 이미지 처리

# ---- 전역 상태 ----
start_pt = None  # 드래그 시작점 (x, y)
end_pt = None  # 드래그 끝점 (x, y)
dragging = False  # 드래그 중 여부
roi_img = None  # 선택된 ROI 저장
src = None  # 원본 이미지
display = None  # 화면에 보여줄 이미지(사각형 표시용)
win_name = "ROI Select (drag) | r:reset, s:save, q:quit"  # 창 이름

def normalize_rect(p1, p2):  # (x1,y1),(x2,y2) 정규화해서 좌상/우하로 만들기
    x1, y1 = p1  # 시작점 분해
    x2, y2 = p2  # 끝점 분해
    left = min(x1, x2)  # 좌측 x
    right = max(x1, x2)  # 우측 x
    top = min(y1, y2)  # 상단 y
    bottom = max(y1, y2)  # 하단 y
    return left, top, right, bottom  # 정규화 결과 반환

def on_mouse(event, x, y, flags, param):  # 마우스 콜백
    global start_pt  # 시작점
    global end_pt  # 끝점
    global dragging  # 드래그 상태
    global roi_img  # ROI 결과
    global display  # 표시 이미지
    global src  # 원본

    if event == cv.EVENT_LBUTTONDOWN:  # 좌클릭 다운
        start_pt = (x, y)  # 시작점 저장
        end_pt = (x, y)  # 끝점 초기화
        dragging = True  # 드래그 시작
        roi_img = None  # 기존 ROI 초기화

    elif event == cv.EVENT_MOUSEMOVE:  # 마우스 이동
        if dragging:  # 드래그 중이면
            end_pt = (x, y)  # 끝점 갱신
            display = src.copy()  # 원본 복사
            l, t, r, b = normalize_rect(start_pt, end_pt)  # 사각형 정규화
            cv.rectangle(display, (l, t), (r, b), (0, 255, 0), 2)  # 드래그 중 사각형 표시(초록)

    elif event == cv.EVENT_LBUTTONUP:  # 좌클릭 업
        if dragging:  # 드래그 중이었다면
            dragging = False  # 드래그 종료
            end_pt = (x, y)  # 끝점 확정
            l, t, r, b = normalize_rect(start_pt, end_pt)  # 정규화

            # 너무 작은 ROI 방지(클릭 실수로 0크기 ROI 나오는 걸 막음)
            if (r - l) >= 2 and (b - t) >= 2:  # 최소 크기 체크
                roi_img = src[t:b, l:r].copy()  # numpy 슬라이싱으로 ROI 추출
                cv.imshow("ROI", roi_img)  # ROI 별도 창에 표시

            display = src.copy()  # 표시 이미지 초기화
            cv.rectangle(display, (l, t), (r, b), (0, 255, 0), 2)  # 확정 사각형 표시

def main():  # 메인
    global src  # 원본 전역
    global display  # 표시 전역
    global start_pt  # 시작점 전역
    global end_pt  # 끝점 전역
    global roi_img  # ROI 전역
    global dragging  # 드래그 전역

    base_dir = r"D:\computer-vision\E01_OpenCV"  # 작업 폴더
    img_path = os.path.join(base_dir, "images", "soccer.jpg")  # 입력 이미지
    out_dir = os.path.join(base_dir, "outputs")  # 출력 폴더
    os.makedirs(out_dir, exist_ok=True)  # outputs 폴더 없으면 생성

    src = cv.imread(img_path)  # 이미지 로드
    if src is None:  # 로드 실패 시
        print(f"[ERROR] 이미지 로드 실패: {img_path}")  # 에러 출력
        return  # 종료

    display = src.copy()  # 표시용 이미지 초기화

    cv.namedWindow(win_name)  # 창 생성
    cv.setMouseCallback(win_name, on_mouse)  # 마우스 콜백 등록

    while True:  # 루프
        cv.imshow(win_name, display)  # 현재 상태 출력
        key = cv.waitKey(1) & 0xFF  # 키 입력

        if key == ord('q'):  # q 종료
            break  # 종료

        if key == ord('r'):  # r 리셋
            start_pt = None  # 시작점 초기화
            end_pt = None  # 끝점 초기화
            dragging = False  # 드래그 상태 초기화
            roi_img = None  # ROI 초기화
            display = src.copy()  # 화면 초기화
            # ROI 창이 열려있으면 닫아줌(없으면 무시)
            try:  # 예외 방지
                cv.destroyWindow("ROI")  # ROI 창 닫기
            except:  # 실패해도 무시
                pass  # 아무것도 안 함

        if key == ord('s'):  # s 저장
            if roi_img is not None:  # ROI가 있을 때만 저장
                ts = time.strftime("%Y%m%d_%H%M%S")  # 타임스탬프
                save_path = os.path.join(out_dir, f"roi_{ts}.png")  # 저장 경로
                ok = cv.imwrite(save_path, roi_img)  # ROI 저장
                if ok:  # 저장 성공
                    print(f"[SAVED] {save_path}")  # 저장 경로 출력
                else:  # 저장 실패
                    print("[ERROR] ROI 저장 실패")  # 에러 출력
            else:  # ROI가 없으면
                print("[INFO] 저장할 ROI가 없음 (먼저 드래그로 ROI 선택)")  # 안내

    cv.destroyAllWindows()  # 모든 창 닫기

if __name__ == "__main__":  # 직접 실행 시
    main()  # main 실행
```

</details>

---
