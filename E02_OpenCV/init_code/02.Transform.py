# OpenCV 라이브러리 불러오기 (이미지 처리용)
import cv2

# 파일 경로를 편하게 다루기 위한 Path 라이브러리
from pathlib import Path


# 현재 실행 중인 파이썬 파일(02.Transform.py)의 폴더 경로를 가져옴
base_dir = Path(__file__).resolve().parent

# 이미지 폴더 안에 있는 rose.png 파일 경로 생성
# base_dir의 상위 폴더(E02_OpenCV) → images → rose.png
image_path = base_dir.parent / "images" / "rose.png"

# 결과 이미지를 저장할 outputs 폴더 경로 생성
output_dir = base_dir.parent / "outputs"

# outputs 폴더가 없으면 새로 생성
output_dir.mkdir(parents=True, exist_ok=True)


# 이미지 파일을 읽어서 img 변수에 저장
img = cv2.imread(str(image_path))

# 만약 이미지가 제대로 불러와지지 않으면 오류 발생
if img is None:
    raise FileNotFoundError(f"이미지를 찾지 못했습니다: {image_path}")


# 이미지의 높이(height)와 너비(width)를 가져옴
h, w = img.shape[:2]

# 이미지 중심 좌표 계산 (회전의 기준점으로 사용)
center = (w // 2, h // 2)


# -----------------------------
# 1. 회전 + 스케일 변환
#    - 30도 회전
#    - 0.8배 축소
# -----------------------------

# 회전 + 스케일 변환 행렬 생성
# center : 회전 중심
# 30 : 회전 각도 (도 단위)
# 0.8 : 이미지 크기를 80%로 축소
M = cv2.getRotationMatrix2D(center, 30, 0.8)


# -----------------------------
# 2. 평행이동 추가
#    - x 방향 +80
#    - y 방향 -40
# -----------------------------

# x 방향으로 80픽셀 이동 (오른쪽 이동)
M[0, 2] += 80

# y 방향으로 -40픽셀 이동 (위쪽 이동)
M[1, 2] -= 40


# -----------------------------
# 3. Affine 변환 적용
# -----------------------------

# 위에서 만든 변환 행렬 M을 이용해 이미지에 Affine 변환 적용
# (회전 + 스케일 + 이동이 동시에 적용됨)
transformed = cv2.warpAffine(img, M, (w, h))


# -----------------------------
# 4. 결과 이미지 저장
# -----------------------------

# 원본 이미지를 outputs 폴더에 저장
cv2.imwrite(str(output_dir / "02_rose_original.png"), img)

# 변환된 이미지를 outputs 폴더에 저장
cv2.imwrite(str(output_dir / "02_rose_transformed.png"), transformed)


# -----------------------------
# 5. 화면에 이미지 출력
# -----------------------------

# 원본 이미지 창 표시
cv2.imshow("Original Image", img)

# 변환된 이미지 창 표시
cv2.imshow("Transformed Image", transformed)

# 키 입력이 있을 때까지 창 유지
cv2.waitKey(0)

# 모든 OpenCV 창 닫기
cv2.destroyAllWindows()