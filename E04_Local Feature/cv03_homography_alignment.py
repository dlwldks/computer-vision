# cv04_sift_panorama.py

# OpenCV 불러오기
import cv2 as cv

# numpy 불러오기
import numpy as np

# matplotlib 불러오기
import matplotlib.pyplot as plt

# os 불러오기
import os

# =========================
# 1. 경로 설정
# =========================

# 첫 번째 이미지 경로
img1_path = r"D:\computer-vision\E04_Local Feature\images\img1.jpg"

# 두 번째 이미지 경로
img2_path = r"D:\computer-vision\E04_Local Feature\images\img2.jpg"

# 결과 저장 폴더
output_dir = r"D:\computer-vision\E04_Local Feature\outputs"

# 출력 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# =========================
# 2. 이미지 불러오기
# =========================

# 첫 번째 이미지 불러오기
img1 = cv.imread(img1_path)

# 두 번째 이미지 불러오기
img2 = cv.imread(img2_path)

# 이미지 로드 확인
if img1 is None:
    print("img1.jpg를 불러올 수 없습니다.")
    raise SystemExit

if img2 is None:
    print("img2.jpg를 불러올 수 없습니다.")
    raise SystemExit

# =========================
# 3. 그레이스케일 변환
# =========================

# SIFT는 보통 grayscale에서 동작
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# =========================
# 4. SIFT 특징점 검출
# =========================

# SIFT 객체 생성
sift = cv.SIFT_create(nfeatures=500)

# img1 특징점과 descriptor 계산
kp1, des1 = sift.detectAndCompute(gray1, None)

# img2 특징점과 descriptor 계산
kp2, des2 = sift.detectAndCompute(gray2, None)

# descriptor가 없으면 종료
if des1 is None or des2 is None:
    print("descriptor 계산 실패")
    raise SystemExit

print(f"img1 keypoints: {len(kp1)}")
print(f"img2 keypoints: {len(kp2)}")

# =========================
# 5. BFMatcher + knnMatch
# =========================

# SIFT는 L2 distance 사용
bf = cv.BFMatcher(cv.NORM_L2)

# 각 특징점에 대해 최근접 2개 찾기
matches = bf.knnMatch(des1, des2, k=2)

# =========================
# 6. Ratio Test로 좋은 매칭 선별
# =========================

# 좋은 매칭 저장 리스트
good_matches = []

# 각 knn 결과 순회
for m, n in matches:
    # Lowe's ratio test
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(f"전체 knn 매칭 수: {len(matches)}")
print(f"좋은 매칭 수: {len(good_matches)}")

# 최소 4개 이상이어야 호모그래피 계산 가능
if len(good_matches) < 4:
    print("좋은 매칭점이 부족해서 호모그래피 계산 불가")
    raise SystemExit

# =========================
# 7. 대응점 추출
# =========================

# img1의 대응점 좌표
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# img2의 대응점 좌표
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# =========================
# 8. 호모그래피 계산
# =========================

# img1을 img2에 맞추는 호모그래피 계산
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# 실패 시 종료
if H is None:
    print("호모그래피 계산 실패")
    raise SystemExit

print("Homography Matrix:")
print(H)

# =========================
# 9. 파노라마용 전체 영역 계산
# =========================

# img1 크기
h1, w1 = img1.shape[:2]

# img2 크기
h2, w2 = img2.shape[:2]

# img1의 네 꼭짓점 좌표
corners_img1 = np.float32([
    [0, 0],
    [0, h1],
    [w1, h1],
    [w1, 0]
]).reshape(-1, 1, 2)

# img2의 네 꼭짓점 좌표
corners_img2 = np.float32([
    [0, 0],
    [0, h2],
    [w2, h2],
    [w2, 0]
]).reshape(-1, 1, 2)

# img1의 꼭짓점을 호모그래피로 변환
transformed_corners_img1 = cv.perspectiveTransform(corners_img1, H)

# 두 이미지의 꼭짓점을 하나로 합침
all_corners = np.concatenate((transformed_corners_img1, corners_img2), axis=0)

# 전체 좌표 범위 계산
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# 전체 파노라마 크기 계산
panorama_width = x_max - x_min
panorama_height = y_max - y_min

print(f"x_min: {x_min}, y_min: {y_min}")
print(f"x_max: {x_max}, y_max: {y_max}")
print(f"panorama size: {panorama_width} x {panorama_height}")

# =========================
# 10. Translation Matrix 생성
# =========================

# 음수 좌표가 있을 수 있으므로 전체 이미지를 오른쪽/아래로 평행이동
translation = np.array([
    [1, 0, -x_min],
    [0, 1, -y_min],
    [0, 0, 1]
], dtype=np.float64)

# 최종 warp 행렬 = translation * homography
H_translated = translation @ H

# =========================
# 11. img1 워핑
# =========================

# 보정된 호모그래피로 img1을 파노라마 캔버스에 warp
panorama = cv.warpPerspective(img1, H_translated, (panorama_width, panorama_height))

# =========================
# 12. img2를 올바른 위치에 배치
# =========================

# img2가 들어갈 시작 위치 계산
x_offset = -x_min
y_offset = -y_min

# 파노라마 위에 img2 배치
panorama[y_offset:y_offset + h2, x_offset:x_offset + w2] = img2

# =========================
# 13. 매칭 결과 시각화
# =========================

# RANSAC inlier만 표시하기 위한 mask
matches_mask = mask.ravel().tolist()

# 특징점 매칭 결과 그리기
matching_result = cv.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches,
    None,
    matchColor=(0, 255, 0),
    singlePointColor=None,
    matchesMask=matches_mask,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# =========================
# 14. 결과 저장
# =========================

# 저장 경로 설정
match_output_path = os.path.join(output_dir, "matching_result.jpg")
panorama_output_path = os.path.join(output_dir, "panorama_result.jpg")

# 이미지 저장
cv.imwrite(match_output_path, matching_result)
cv.imwrite(panorama_output_path, panorama)

# =========================
# 15. 결과 출력
# =========================

# BGR -> RGB 변환
matching_result_rgb = cv.cvtColor(matching_result, cv.COLOR_BGR2RGB)
panorama_rgb = cv.cvtColor(panorama, cv.COLOR_BGR2RGB)

# figure 생성
plt.figure(figsize=(18, 8))

# 왼쪽: 매칭 결과
plt.subplot(1, 2, 1)
plt.imshow(matching_result_rgb)
plt.title("Matching Result")
plt.axis("off")

# 오른쪽: 파노라마 결과
plt.subplot(1, 2, 2)
plt.imshow(panorama_rgb)
plt.title("Panorama Result")
plt.axis("off")

# 레이아웃 정리
plt.tight_layout()

# 출력
plt.show() 