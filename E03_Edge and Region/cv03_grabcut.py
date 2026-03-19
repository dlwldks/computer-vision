#cv03_grabcut.py
import cv2 as cv                      # OpenCV 라이브러리 (이미지 처리)
import numpy as np                   # 수치 연산 라이브러리
import matplotlib.pyplot as plt      # 결과 시각화를 위한 라이브러리
import os                            # 파일/폴더 경로 처리용 라이브러리

# =========================
# 1. 이미지 경로 설정
# =========================
image_path = r"D:/computer-vision/E03_Edge and Region/images/coffee cup.JPG"   # 입력 이미지 경로
output_path = r"D:/computer-vision/E03_Edge and Region/outputs/03_grabcut_result.png"  # 결과 저장 경로

# =========================
# 2. 이미지 불러오기
# =========================
img_bgr = cv.imread(image_path)      # 이미지를 BGR 형식으로 불러옴

# 이미지가 정상적으로 로드되지 않았을 경우 예외 처리
if img_bgr is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

# =========================
# 3. 색상 변환
# =========================
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)   # Matplotlib 출력용으로 BGR → RGB 변환

# =========================
# 4. GrabCut 초기 설정
# =========================
mask = np.zeros(img_bgr.shape[:2], np.uint8)   # 이미지 크기에 맞는 초기 마스크 생성 (모두 0으로 초기화)

# GrabCut 알고리즘에서 사용하는 배경/전경 모델 초기화
bgdModel = np.zeros((1, 65), np.float64)   # 배경 모델
fgdModel = np.zeros((1, 65), np.float64)   # 전경 모델

# =========================
# 5. 초기 사각형 영역 설정
# =========================
h, w = img_bgr.shape[:2]   # 이미지 높이(h), 너비(w) 추출

# 객체가 포함되도록 사각형 영역 설정 (x, y, width, height)
rect = (int(w * 0.15), int(h * 0.10), int(w * 0.70), int(h * 0.80))

# =========================
# 6. GrabCut 실행
# =========================
cv.grabCut(
    img_bgr,               # 입력 이미지
    mask,                  # 초기 마스크
    rect,                  # 초기 사각형 영역
    bgdModel,              # 배경 모델
    fgdModel,              # 전경 모델
    5,                     # 반복 횟수 (클수록 정밀하지만 느림)
    cv.GC_INIT_WITH_RECT   # 사각형 기반 초기화 방식
)

# =========================
# 7. 마스크 후처리 (0/1 변환)
# =========================
# 배경(GC_BGD)과 배경 가능성(GC_PR_BGD)은 0
# 전경(GC_FGD)과 전경 가능성(GC_PR_FGD)은 1로 변환
mask2 = np.where(
    (mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),
    0,
    1
).astype("uint8")

# =========================
# 8. 배경 제거
# =========================
# 마스크를 RGB 이미지에 곱하여 배경 제거
result = img_rgb * mask2[:, :, np.newaxis]

# 마스크를 시각화하기 위해 0~255 범위로 변환
mask_display = mask2 * 255

# =========================
# 9. 출력 폴더 생성
# =========================
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # outputs 폴더 없으면 생성

# =========================
# 10. 결과 시각화
# =========================
plt.figure(figsize=(15, 5))   # 전체 출력 크기 설정

# (1) 원본 이미지 출력
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

# (2) GrabCut 마스크 출력
plt.subplot(1, 3, 2)
plt.imshow(mask_display, cmap="gray")   # 흑백 이미지로 표시
plt.title("GrabCut Mask")
plt.axis("off")

# (3) 배경 제거 결과 출력
plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title("Background Removed")
plt.axis("off")

plt.tight_layout()   # 레이아웃 자동 정리

# =========================
# 11. 결과 저장 및 출력
# =========================
plt.savefig(output_path, bbox_inches="tight")  # 결과 이미지 저장
plt.show()                                    # 화면에 출력

# =========================
# 12. 결과 정보 출력
# =========================
print("저장 완료:", output_path)        # 저장 경로 출력
print("사용한 초기 사각형(rect):", rect)  # 사용된 rect 값 출력