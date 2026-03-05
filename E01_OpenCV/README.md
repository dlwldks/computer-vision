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

설치:
```bash
pip install opencv-python numpy

1) Problem 1 — Grayscale + Side-by-side 출력
✅ 문제

OpenCV로 이미지를 로드하고 화면에 출력한다.

이미지를 그레이스케일로 변환한다.

원본 이미지와 그레이스케일 이미지를 np.hstack()으로 가로 결합해 나란히 표시한다.

cv.imshow()와 cv.waitKey()로 결과를 출력하고, 아무 키나 누르면 창이 닫히도록 한다.

✅ 전체 코드 (cv01_grayscale.py)
# cv01_grayscale.py
# OpenCV로 이미지를 불러와 그레이스케일 변환 후
# 원본 이미지와 나란히 출력하는 예제

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
    parser.add_argument("image", nargs="?", default=default_image, help=f"불러올 이미지 경로 (기본: {default_image})")
    args = parser.parse_args()

    # 1) 이미지 로드 (OpenCV는 BGR 형식으로 읽음)
    img = cv.imread(args.image)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {args.image}")
        sys.exit(1)

    # 2) 그레이스케일 변환 (cv.COLOR_BGR2GRAY 사용)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 그레이스케일은 1채널이므로 원본과 가로로 연결하려면 3채널로 변환
    gray_color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    # 3) np.hstack()으로 가로 연결
    result = np.hstack((img, gray_color))

    # 4) 화면 크기에 맞춰 자동 축소 후 표시, 아무 키나 누르면 닫힘
    # 화면 크기 확인 (tkinter 사용, 실패하면 기본값 사용)
    margin = 100
    if tk is not None:
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
    else:
        screen_w, screen_h = 1280, 720

    res_h, res_w = result.shape[0], result.shape[1]
    scale = min(1.0, (screen_w - margin) / res_w, (screen_h - margin) / res_h)
    if scale < 1.0:
        new_w = int(res_w * scale)
        new_h = int(res_h * scale)
        result_show = cv.resize(result, (new_w, new_h), interpolation=cv.INTER_AREA)
    else:
        result_show = result

    cv.namedWindow("Original | Grayscale", cv.WINDOW_NORMAL)
    cv.imshow("Original | Grayscale", result_show)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
✅ 핵심 코드
img = cv.imread(args.image)                          # 1) 이미지 로드
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)           # 2) 그레이스케일 변환
gray_color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)    # (hstack을 위해 3채널로 변환)
result = np.hstack((img, gray_color))                # 3) 가로 결합
cv.imshow("Original | Grayscale", result_show)       # 4) 출력
cv.waitKey(0)                                        # 키 입력 대기 후 종료
✅ 실행 결과

D:\computer-vision\E01_OpenCV\outputs\cv01_result.png

2) Problem 2 — Paint (Mouse + Keyboard)
✅ 문제

마우스로 이미지 위에 붓질(페인팅)한다.

키보드 입력으로 붓 크기를 조절한다.

초기 붓 크기: 5

+ 입력 시 1 증가, - 입력 시 1 감소

붓 크기 범위: 최소 1, 최대 15

좌클릭: 파란색 / 우클릭: 빨간색 / 드래그로 연속 그리기

q 키를 누르면 종료

힌트: cv.setMouseCallback() + cv.circle() / 키 입력은 루프에서 cv.waitKey(1)로 처리

✅ 전체 코드 (cv02_paint.py)
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
✅ 핵심 코드
cv.setMouseCallback(win_name, on_mouse)                  # 마우스 이벤트 처리 등록
cv.circle(img, (x, y), brush_size, color, -1)            # 현재 붓 크기로 원을 그려 페인팅
key = cv.waitKey(1) & 0xFF                               # 루프 안에서 키 입력 받기
brush_size = clamp(brush_size ± 1, brush_min, brush_max) # 붓 크기 1~15 제한
✅ 실행 결과

아래에 실행 결과 이미지를 붙여 넣으세요.

(붙여넣기) results/cv02_result.png

3) Problem 3 — ROI 선택 / 출력 / 저장
✅ 문제

이미지를 불러오고, 마우스 클릭+드래그로 관심영역(ROI)을 선택한다.

드래그 중에는 사각형으로 선택 영역을 시각화한다.

마우스를 놓으면 해당 영역을 잘라 별도의 창에 출력한다.

r 키: 선택 리셋(처음부터 다시 선택)

s 키: 선택한 ROI를 이미지 파일로 저장

힌트: cv.rectangle()로 드래그 중 영역 표시, ROI는 numpy slicing, 저장은 cv.imwrite()

✅ 전체 코드 (cv03_roi.py)
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
✅ 핵심 코드
cv.setMouseCallback(win_name, on_mouse)     # 마우스 이벤트 처리 등록
cv.rectangle(display, (l, t), (r, b), ...)  # 드래그 중 사각형 시각화
roi_img = src[t:b, l:r].copy()              # numpy slicing으로 ROI 추출
cv.imshow("ROI", roi_img)                   # ROI 별도 창 출력
cv.imwrite(save_path, roi_img)              # ROI 저장
✅ 실행 결과

아래에 실행 결과 이미지를 붙여 넣으세요.

(붙여넣기) results/cv03_drag_select.png

(붙여넣기) results/cv03_roi_window.png

(붙여넣기) outputs/roi_YYYYMMDD_HHMMSS.png (저장 파일 예시)

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
