#cv03_homography_alignment.py
import cv2 as cv                  # OpenCV (이미지 처리)
import numpy as np               # 수치 연산 (좌표, 행렬 계산)
import matplotlib.pyplot as plt  # 시각화
import os                        # 파일/폴더 관리

# =========================
# 경로 설정
# =========================

# 첫 번째 이미지 (기준이 되는 이미지)
img1_path = r"D:\computer-vision\E04_Local Feature\images\img1.jpg"

# 두 번째 이미지 (정렬 대상)
img2_path = r"D:\computer-vision\E04_Local Feature\images\img3.jpg"

# 결과 저장 폴더
output_path = r"D:\computer-vision\E04_Local Feature\outputs"

# 출력 폴더가 없으면 생성
os.makedirs(output_path, exist_ok=True)

# =========================
# 파라미터 설정
# =========================

ratio_thresh = 0.7      # Lowe's ratio test 기준 (0.7이 일반적으로 많이 사용됨)
top_k_matches = 30      # 좋은 매칭 중 상위 몇 개만 사용할지
ransac_thresh = 5.0     # RANSAC에서 outlier 제거 기준 (픽셀 단위)

# =========================
# 이미지 불러오기
# =========================

# 이미지 읽기 (BGR 형식)
img1 = cv.imread(img1_path)
img2 = cv.imread(img2_path)

# 이미지 로드 실패 시 종료
if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다.")
    raise SystemExit

# SIFT는 grayscale 기반 → 변환
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# =========================
# SIFT 특징점 검출
# =========================

# SIFT 객체 생성 (최대 300개 특징점)
sift = cv.SIFT_create(nfeatures=300)

# 각 이미지에서 keypoint + descriptor 추출
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# descriptor 생성 실패 시 종료
if des1 is None or des2 is None:
    print("디스크립터를 생성하지 못했습니다.")
    raise SystemExit

# 특징점 개수 출력
print("img1 keypoints:", len(kp1))
print("img2 keypoints:", len(kp2))

# =========================
# 특징점 매칭 (BFMatcher + knnMatch)
# =========================

# BFMatcher 생성 (L2 거리 사용 → SIFT에 적합)
bf = cv.BFMatcher(cv.NORM_L2)

# 각 descriptor마다 가장 가까운 2개 후보 찾기
knn_matches = bf.knnMatch(des1, des2, k=2)

good_matches = []  # 좋은 매칭 저장 리스트

# Lowe's Ratio Test 적용
for pair in knn_matches:
    if len(pair) < 2:
        continue  # 2개 미만이면 비교 불가 → skip

    m, n = pair  # m: 가장 가까운 매칭, n: 두 번째 후보

    # m이 n보다 충분히 더 가깝다면 좋은 매칭으로 인정
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# 거리 기준으로 정렬 (작을수록 좋은 매칭)
good_matches = sorted(good_matches, key=lambda x: x.distance)

# 상위 K개만 선택 (노이즈 줄이기)
good_matches = good_matches[:top_k_matches]

print("good matches:", len(good_matches))

# 호모그래피 최소 조건: 4개 이상 필요
if len(good_matches) < 4:
    print("호모그래피 계산에 필요한 매칭점이 부족합니다.")
    raise SystemExit

# =========================
# 호모그래피 계산
# =========================

# img1의 keypoint 좌표 (출발점)
src_pts = np.float32(
    [kp1[m.queryIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

# img2의 keypoint 좌표 (도착점)
dst_pts = np.float32(
    [kp2[m.trainIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

# RANSAC 기반으로 호모그래피 행렬 계산
# → 이상치(outlier) 자동 제거
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransac_thresh)

# 계산 실패 시 종료
if H is None:
    print("호모그래피 계산에 실패했습니다.")
    raise SystemExit

# =========================
# 이미지 정합 (warp)
# =========================

# 이미지 크기 추출
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# img1을 img2 기준으로 변환
# 출력 크기: 두 이미지를 나란히 포함하도록 설정
warped = cv.warpPerspective(img1, H, (w1 + w2, max(h1, h2)))

# 기준 이미지(img2)를 결과 이미지 왼쪽에 덮어쓰기
result = warped.copy()
result[0:h2, 0:w2] = np.maximum(result[0:h2, 0:w2], img2)

# =========================
# Inlier만 시각화
# =========================

# RANSAC에서 살아남은 매칭만 표시 (1이면 inlier)
matches_mask = mask.ravel().tolist() if mask is not None else None

# 매칭 결과 시각화
match_vis = cv.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches, None,
    matchColor=(0, 255, 0),   # 매칭선 색 (초록)
    singlePointColor=None,
    matchesMask=matches_mask, # inlier만 표시
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# =========================
# 결과 저장
# =========================

# 매칭 결과 저장 경로
match_file = os.path.join(output_path, "03_matching_result.jpg")

# 정합 이미지 저장 경로
warped_file = os.path.join(output_path, "03_warped.jpg")

# 이미지 파일 저장
cv.imwrite(match_file, match_vis)
cv.imwrite(warped_file, result)

print(f"매칭 이미지 저장: {match_file}")
print(f"정합 이미지 저장: {warped_file}")

# =========================
# 결과 출력
# =========================

plt.figure(figsize=(18, 8))

# 매칭 결과 표시
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(match_vis, cv.COLOR_BGR2RGB))
plt.title("Matching Result")
plt.axis("off")

# 정합 결과 표시
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title("Warped Image")
plt.axis("off")

# 레이아웃 정리
plt.tight_layout()

# 화면 출력
plt.show()