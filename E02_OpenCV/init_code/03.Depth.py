import cv2  # OpenCV 라이브러리 (이미지 처리용)
import numpy as np  # 수치 계산 및 배열 처리를 위한 NumPy

# 좌/우 이미지 불러오기
left_color = cv2.imread(r"D:/computer-vision/E02_OpenCV/images/left.png")  # 왼쪽 카메라 이미지 읽기
right_color = cv2.imread(r"D:/computer-vision/E02_OpenCV/images/right.png")  # 오른쪽 카메라 이미지 읽기

# 이미지가 제대로 로드되지 않았을 경우 예외 처리
if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")  # 이미지 파일이 없으면 오류 발생

# 카메라 파라미터
f = 700.0  # 카메라 초점거리 (focal length)
B = 0.12  # 두 카메라 사이 거리 (baseline)

# ROI 설정 (관심 영역: Painting, Frog, Teddy)
rois = {
    "Painting": (55, 50, 130, 110),  # (x, y, width, height)
    "Frog": (90, 265, 230, 95),  # Frog 영역 좌표
    "Teddy": (310, 35, 115, 90)  # Teddy 영역 좌표
}

# -----------------------------
# 그레이스케일 변환
# -----------------------------
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)  # 왼쪽 이미지를 그레이스케일로 변환
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)  # 오른쪽 이미지를 그레이스케일로 변환

# -----------------------------
# 1. Disparity 계산
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=16*6, blockSize=15)  # Stereo Block Matching 객체 생성
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0  # 두 이미지의 시차(disparity) 계산

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
depth_map = np.zeros_like(disparity, dtype=np.float32)  # disparity와 같은 크기의 depth 배열 생성
valid_mask = disparity > 0  # disparity가 0보다 큰 유효한 영역만 선택
depth_map[valid_mask] = (f * B) / disparity[valid_mask]  # 깊이(depth) 계산 공식 적용

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}  # ROI별 결과를 저장할 딕셔너리 생성

for name, (x, y, w, h) in rois.items():  # 각 ROI 영역 반복

    roi_disp = disparity[y:y+h, x:x+w]  # ROI 영역의 disparity 값 추출
    roi_depth = depth_map[y:y+h, x:x+w]  # ROI 영역의 depth 값 추출

    roi_valid = roi_disp > 0  # disparity 값이 유효한 픽셀만 선택

    if np.any(roi_valid):  # 유효한 픽셀이 하나라도 있는 경우
        mean_disp = np.mean(roi_disp[roi_valid])  # ROI 영역의 평균 disparity 계산
        mean_depth = np.mean(roi_depth[roi_valid])  # ROI 영역의 평균 depth 계산
    else:  # 유효한 disparity가 없는 경우
        mean_disp = np.nan  # Not a Number로 처리
        mean_depth = np.nan

    results[name] = {  # 결과 딕셔너리에 저장
        "mean_disparity": mean_disp,
        "mean_depth": mean_depth
    }

# -----------------------------
# 4. 결과 출력 (PPT처럼)
# -----------------------------
closest_roi = max(results.items(), key=lambda x: x[1]["mean_disparity"])[0]  # disparity가 가장 큰 ROI (가장 가까운 물체)
farthest_roi = min(results.items(), key=lambda x: x[1]["mean_disparity"])[0]  # disparity가 가장 작은 ROI (가장 먼 물체)

print("가장 가까운 ROI:", closest_roi)  # 가장 가까운 ROI 출력
print("가장 먼 ROI:", farthest_roi)  # 가장 먼 ROI 출력

# -----------------------------
# 5. disparity 시각화
# -----------------------------
disp_tmp = disparity.copy()  # disparity 배열 복사
disp_tmp[disp_tmp <= 0] = np.nan  # 0 이하 값은 NaN으로 처리 (유효하지 않은 값 제거)

d_min = np.nanpercentile(disp_tmp, 5)  # disparity의 하위 5% 값 계산
d_max = np.nanpercentile(disp_tmp, 95)  # disparity의 상위 95% 값 계산

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)  # disparity 값을 0~1 범위로 정규화
disp_scaled = np.clip(disp_scaled, 0, 1)  # 값 범위를 0~1로 제한

disp_vis = np.zeros_like(disparity, dtype=np.uint8)  # 시각화용 disparity 배열 생성
valid_disp = ~np.isnan(disp_tmp)  # NaN이 아닌 유효한 disparity 영역 선택
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)  # 0~255 범위로 변환

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)  # 컬러맵 적용 (JET 색상)

# -----------------------------
# 화면 출력 (PPT 스타일)
# -----------------------------
cv2.imshow("Original", left_color)  # 원본 왼쪽 이미지 출력
cv2.imshow("Disparity map", disparity_color)  # disparity 컬러맵 이미지 출력

cv2.waitKey(0)  # 키 입력이 있을 때까지 대기
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기