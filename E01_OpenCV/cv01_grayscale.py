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