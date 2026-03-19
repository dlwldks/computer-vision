#cv01_sobel.py
import cv2 as cv                      # OpenCV 라이브러리 (이미지 처리)
import numpy as np                   # 수치 연산용 라이브러리
import matplotlib.pyplot as plt      # 결과 시각화를 위한 라이브러리
import os                            # 파일/폴더 경로 관련 라이브러리

# =========================
# 1. 이미지 경로 설정
# =========================
image_path = r"D:/computer-vision/E03_Edge and Region/images/edgeDetectionImage.jpg"  # 입력 이미지 경로
output_path = r"D:/computer-vision/E03_Edge and Region/outputs/01_sobel_result.png"   # 결과 저장 경로

# =========================
# 2. 이미지 불러오기
# =========================
img_bgr = cv.imread(image_path)      # 이미지를 BGR 형식으로 읽어옴

# 이미지가 정상적으로 로드되지 않았을 경우 예외 처리
if img_bgr is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

# =========================
# 3. 색상 변환
# =========================
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)   # Matplotlib 출력용으로 BGR → RGB 변환
gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)     # Sobel 적용을 위한 그레이스케일 변환

# =========================
# 4. Sobel 에지 검출
# =========================
# x 방향 에지 검출 (수직 방향 변화 감지)
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)

# y 방향 에지 검출 (수평 방향 변화 감지)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# =========================
# 5. 에지 강도 계산
# =========================
# x, y 방향 에지를 결합하여 전체 에지 강도 계산
magnitude = cv.magnitude(sobel_x, sobel_y)

# float64 → uint8 변환 (이미지로 표현 가능하게)
magnitude_abs = cv.convertScaleAbs(magnitude)

# =========================
# 6. 출력 폴더 생성
# =========================
# outputs 폴더가 없으면 자동 생성
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# =========================
# 7. 결과 시각화
# =========================
plt.figure(figsize=(12, 5))   # 전체 출력 크기 설정

# (1) 원본 이미지 출력
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")               # 축 제거

# (2) Sobel 에지 강도 이미지 출력
plt.subplot(1, 2, 2)
plt.imshow(magnitude_abs, cmap="gray")  # 흑백 이미지로 출력
plt.title("Sobel Edge Magnitude")
plt.axis("off")

plt.tight_layout()            # 레이아웃 자동 정리

# =========================
# 8. 결과 저장 및 출력
# =========================
plt.savefig(output_path, bbox_inches="tight")  # 이미지 파일로 저장
plt.show()                                    # 화면에 출력

# 저장 완료 메시지 출력
print("저장 완료:", output_path)