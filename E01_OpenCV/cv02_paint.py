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