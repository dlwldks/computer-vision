#cv02_hough.py
import cv2 as cv                      # OpenCV 라이브러리 (이미지 처리)
import numpy as np                   # 수치 연산 라이브러리
import matplotlib.pyplot as plt      # 결과 시각화를 위한 라이브러리
import os                            # 파일/폴더 경로 처리용 라이브러리

# =========================
# 1. 이미지 경로 설정
# =========================
image_path = r"D:/computer-vision/E03_Edge and Region/images/dabo.jpg"   # 입력 이미지 경로
output_path = r"D:/computer-vision/E03_Edge and Region/outputs/02_hough_lines.png"  # 결과 저장 경로

# =========================
# 2. 이미지 불러오기
# =========================
img_bgr = cv.imread(image_path)      # 이미지를 BGR 형식으로 불러옴

# 이미지가 정상적으로 로드되지 않았을 경우 예외 처리
if img_bgr is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

# =========================
# 3. 원본 이미지 복사
# =========================
line_img = img_bgr.copy()            # 직선을 그릴 이미지를 별도로 복사

# =========================
# 4. 전처리 (그레이스케일 변환)
# =========================
gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)   # Canny 적용을 위해 흑백 이미지로 변환

# =========================
# 5. Canny 에지 검출
# =========================
edges = cv.Canny(gray, 100, 200)    # threshold1=100, threshold2=200으로 에지 검출

# =========================
# 6. Hough 변환을 이용한 직선 검출
# =========================
lines = cv.HoughLinesP(
    edges,                          # 입력: 에지 이미지
    rho=1,                          # 거리 해상도 (픽셀 단위)
    theta=np.pi / 180,              # 각도 해상도 (라디안 단위)
    threshold=80,                   # 최소 교차점 수 (값이 클수록 엄격)
    minLineLength=50,               # 최소 직선 길이
    maxLineGap=10                   # 선 사이 최대 허용 간격
)

# =========================
# 7. 검출된 직선 그리기
# =========================
if lines is not None:               # 직선이 검출된 경우
    for line in lines:              # 각 직선에 대해 반복
        x1, y1, x2, y2 = line[0]   # 시작점(x1,y1), 끝점(x2,y2) 추출
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색(0,0,255), 두께 2로 선 그리기

# =========================
# 8. 색상 변환 (BGR → RGB)
# =========================
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)         # 원본 이미지 변환
line_img_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)   # 직선이 그려진 이미지 변환

# =========================
# 9. 출력 폴더 생성
# =========================
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # outputs 폴더가 없으면 생성

# =========================
# 10. 결과 시각화
# =========================
plt.figure(figsize=(12, 5))       # 전체 출력 크기 설정

# (1) 원본 이미지 출력
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")                   # 축 제거

# (2) 직선 검출 결과 출력
plt.subplot(1, 2, 2)
plt.imshow(line_img_rgb)
plt.title("Detected Lines (HoughLinesP)")
plt.axis("off")

plt.tight_layout()                # 레이아웃 자동 정리

# =========================
# 11. 결과 저장 및 출력
# =========================
plt.savefig(output_path, bbox_inches="tight")  # 결과 이미지 저장
plt.show()                                    # 화면에 출력

# =========================
# 12. 결과 정보 출력
# =========================
print("저장 완료:", output_path)  # 저장 경로 출력

# 검출된 직선 개수 출력
if lines is not None:
    print("검출된 직선 개수:", len(lines))
else:
    print("검출된 직선이 없습니다.")