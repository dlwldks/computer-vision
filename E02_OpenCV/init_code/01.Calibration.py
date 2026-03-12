import cv2
import numpy as np
import glob
import os

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

base_dir = os.path.dirname(os.path.abspath(__file__))
image_pattern = os.path.join(base_dir, "..", "images", "calibration_images", "left*.jpg")
images = glob.glob(image_pattern)

img_size = None

for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print(f"이미지 로드 실패: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, ret)
        cv2.imshow("Chessboard Corners", vis)
        cv2.waitKey(300)
    else:
        print(f"코너 검출 실패: {fname}")

cv2.destroyAllWindows()

if len(images) == 0:
    raise FileNotFoundError(f"이미지를 찾지 못했습니다: {image_pattern}")

if len(objpoints) == 0 or len(imgpoints) == 0:
    raise ValueError("체크보드 코너를 검출한 이미지가 없습니다.")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None
)

print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

test_img = cv2.imread(images[0])
if test_img is None:
    raise FileNotFoundError(f"테스트 이미지를 불러올 수 없습니다: {images[0]}")

undistorted = cv2.undistort(test_img, K, dist)

cv2.imshow("Original Image", test_img)
cv2.imshow("Undistorted Image", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()