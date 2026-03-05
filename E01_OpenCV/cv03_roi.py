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