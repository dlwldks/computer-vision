# 01.Calibration.py
# 체커보드 이미지를 이용하여 카메라 내부 파라미터(Camera Matrix)와
# 렌즈 왜곡 계수(Distortion Coefficients)를 계산하는 카메라 보정 예제

import cv2
import numpy as np
import glob
import os

# 체크보드 내부 코너 개수 (가로 9, 세로 6)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ----------------------------------
# 1. 실제 세계 좌표 생성
# ----------------------------------
# 예: (0,0,0), (1,0,0), (2,0,0) ... 형태로 만들고 square_size를 곱함
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 모든 이미지에 대해 누적 저장할 리스트
objpoints = []   # 3D 실제 좌표
imgpoints = []   # 2D 이미지 좌표

# 현재 파일 기준으로 calibration 이미지 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
image_pattern = os.path.join(base_dir, "..", "images", "calibration_images", "left*.jpg")
images = glob.glob(image_pattern)

if len(images) == 0:
    raise FileNotFoundError(f"체커보드 이미지를 찾지 못했습니다: {image_pattern}")

img_size = None

# ----------------------------------
# 2. 체크보드 코너 검출
# ----------------------------------
for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print(f"[로드 실패] {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]  # (width, height)

    # 체커보드 코너 검출
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # 성공하면 실제 좌표와 이미지 좌표 저장
        objpoints.append(objp)

        # 코너를 sub-pixel 단위로 더 정확하게 보정
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )
        imgpoints.append(corners2)

        # 검출 결과 시각화
        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, ret)
        cv2.imshow("Chessboard Corners", vis)
        cv2.waitKey(300)

    else:
        print(f"[코너 검출 실패] {fname}")

cv2.destroyAllWindows()

if len(objpoints) == 0 or len(imgpoints) == 0:
    raise ValueError("체커보드 코너를 검출한 이미지가 없습니다.")

# ----------------------------------
# 3. 카메라 캘리브레이션
# ----------------------------------
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,   # 3D 실제 좌표
    imgpoints,   # 2D 이미지 좌표
    img_size,    # 이미지 크기
    None,        # cameraMatrix 초기값 없음
    None         # distCoeffs 초기값 없음
)

print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# ----------------------------------
# 4. 왜곡 보정 시각화
# ----------------------------------
test_img = cv2.imread(images[0])

if test_img is None:
    raise FileNotFoundError(f"테스트 이미지를 불러올 수 없습니다: {images[0]}")

undistorted = cv2.undistort(test_img, K, dist)

# 보기 좋게 원본/보정 결과 나란히 붙이기
result = np.hstack((test_img, undistorted))

cv2.imshow("Original | Undistorted", result)
cv2.waitKey(0)
cv2.destroyAllWindows()