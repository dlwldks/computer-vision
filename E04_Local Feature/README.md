# E04. Local Feature & Image Matching

## 0. 과제 개요

이번 과제에서는 OpenCV를 사용하여 **SIFT 기반 특징점 검출**과  
**이미지 간 특징점 매칭**, 그리고 **호모그래피 기반 이미지 정합(Image Alignment)**을 구현합니다.

- SIFT를 이용한 특징점(Keypoint) 검출
- BFMatcher를 이용한 특징점 매칭
- Lowe’s Ratio Test를 통한 좋은 매칭 선택
- RANSAC을 이용한 이상치 제거
- Homography를 이용한 이미지 정합

---

## 요구사항 및 설치

- Python 3.7 이상
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

설치

```bash
pip install opencv-python numpy matplotlib
```

---

## 폴더 구조 (요약)

```
E04_Local Feature/
│
├── images/
│   ├── img1.jpg
│   ├── img3.jpg
│   ├── mot_color70.jpg
│   └── mot_color83.jpg
│
├── outputs/
│   ├── 01_sift_keypoints.jpg
│   ├── 02_sift_matching.jpg
│   ├── 03_matching_result.jpg
│   └── 03_warped.jpg
│
├── cv01_sift_keypoints.py
├── cv02_sift_matching.py
├── cv03_homography.py
│
└── README.md
```

- `cv01_sift_keypoints.py` : SIFT를 이용한 특징점 검출 및 시각화
- `cv02_sift_matching.py` : BFMatcher를 이용한 특징점 매칭 및 유사도 기반 필터링
- `cv03_homography.py` : Lowe’s Ratio Test + RANSAC을 이용한 호모그래피 계산 및 이미지 정합

---

# 실행 방법

```bash
python E04_Local Feature/cv01_sift_keypoints.py
python E04_Local Feature/cv02_sift_matching.py
python E04_Local Feature/cv03_homography.py
```

또는

```bash
python cv01_sift_keypoints.py
python cv02_sift_matching.py
python cv03_homography.py
```

---

# Problem 1 — SIFT Keypoint Detection

이미지를 그레이스케일로 변환한 후 **SIFT 알고리즘을 사용하여 특징점(Keypoints)**을 검출하는 과정입니다.

---

## 주요 내용

- **이미지 그레이스케일 변환**
- **SIFT를 이용한 특징점 검출**
- **특징점의 크기와 방향 시각화**
- **Keypoint 및 Descriptor 생성**

---

### 실행 결과

#### Original Image (원본 이미지)

<img src="https://github.com/dlwldks/computer-vision/blob/main/E04_Local%20Feature/images/mot_color70.jpg" width="400">

---

#### SIFT Keypoints (특징점 검출 결과)

<img src="https://github.com/dlwldks/computer-vision/blob/main/E04_Local%20Feature/outputs/01_sift_keypoints.png" width="400">

---

**실행 결과**

검출된 keypoints 수: 300  
descriptor shape: (300, 128)

---


<details>
<summary>전체 코드 — cv01_sift_keypoints.py</summary>

```python
# cv01_sift_keypoints.py

import cv2 as cv                 # OpenCV 라이브러리 (이미지 처리)
import matplotlib.pyplot as plt # 이미지 시각화를 위한 matplotlib
import os                       # 파일 및 폴더 경로 관리

# =========================
# 1. 경로 설정
# =========================

img_path = r"D:\computer-vision\E04_Local Feature\images\mot_color70.jpg"  
# 입력 이미지 경로 (raw string 사용해서 \ 오류 방지)

output_dir = r"D:\computer-vision\E04_Local Feature\outputs"  
# 결과 이미지 저장 폴더 경로

os.makedirs(output_dir, exist_ok=True)  
# 출력 폴더가 없으면 생성 (이미 있으면 무시)

# =========================
# 2. 이미지 불러오기
# =========================

img = cv.imread(img_path)  
# 이미지 파일을 읽어옴 (BGR 형식으로 로드됨)

if img is None:  
    # 이미지가 정상적으로 불러와지지 않은 경우
    print("이미지를 불러올 수 없습니다.")
    raise SystemExit  # 프로그램 강제 종료

img_original = img.copy()  
# 원본 이미지 보존 (drawKeypoints에서 이미지 변경 방지)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
# 컬러 이미지를 그레이스케일로 변환
# SIFT는 grayscale 이미지에서 수행됨

# =========================
# 3. SIFT 객체 생성
# =========================

sift = cv.SIFT_create(nfeatures=300)  
# SIFT 알고리즘 객체 생성
# nfeatures=300 → 최대 300개의 특징점만 검출하도록 제한

# =========================
# 4. 특징점 검출 및 descriptor 계산
# =========================

keypoints, descriptors = sift.detectAndCompute(gray, None)  
# keypoints: 특징점 위치, 크기, 방향 정보
# descriptors: 각 특징점의 128차원 벡터

# =========================
# 5. 특징점 시각화
# =========================

result = cv.drawKeypoints(
    img_original.copy(),   # 원본을 복사해서 사용 (원본 훼손 방지)
    keypoints,             # 검출된 특징점
    None,                  # 결과 이미지를 새로 생성
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  
    # 특징점의 크기와 방향까지 표시
)

# =========================
# 6. 결과 정보 출력
# =========================

print("검출된 keypoints 수:", len(keypoints))  
# 검출된 특징점 개수 출력

if descriptors is not None:  
    print("descriptor shape:", descriptors.shape)  
    # descriptor 배열 형태 출력 (개수 x 128)

# =========================
# 7. 결과 이미지 저장
# =========================

save_path = os.path.join(output_dir, "01_sift_keypoints.jpg")  
# 저장할 파일 경로 생성

cv.imwrite(save_path, result)  
# 결과 이미지를 파일로 저장

print("결과 이미지 저장 완료:", save_path)  
# 저장 완료 메시지 출력

# =========================
# 8. 원본 이미지와 결과 이미지 나란히 출력
# =========================

plt.figure(figsize=(14, 6))  
# 전체 출력 창 크기 설정

# ----- 원본 이미지 -----
plt.subplot(1, 2, 1)  
# 1행 2열 중 첫 번째 위치

plt.imshow(cv.cvtColor(img_original, cv.COLOR_BGR2RGB))  
# BGR → RGB 변환 후 출력 (matplotlib은 RGB 사용)

plt.title("Original Image")  
# 제목 설정

plt.axis("off")  
# 축 제거 (깔끔한 출력)

# ----- SIFT 결과 이미지 -----
plt.subplot(1, 2, 2)  
# 두 번째 위치

plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))  
# 결과 이미지 출력 (RGB 변환)

plt.title("SIFT Keypoints")  
# 제목 설정

plt.axis("off")  
# 축 제거

plt.tight_layout()  
# subplot 간 간격 자동 조정

plt.show()  
# 화면에 출력
```

</details>

---

## Problem 2 — SIFT Feature Matching

두 이미지 간 특징점을 비교하여 **대응점(Matching Points)**을 찾는 문제입니다.

### 적용한 처리

- SIFT를 이용한 특징점 및 descriptor 생성  
- BFMatcher를 이용한 특징점 매칭  
- distance 기준 정렬 및 상위 매칭 선택  
- 매칭된 특징점을 선으로 시각화  

---

## 실행 결과

<figure>
  <img src="https://github.com/dlwldks/computer-vision/blob/main/E04_Local%20Feature/outputs/02_sift_matching.jpg" alt="SIFT 매칭 결과" width="400"/>
  <figcaption>SIFT Feature Matching 결과</figcaption>
</figure>

---

<details>
<summary>전체 코드 — cv02_sift_matching.py</summary>

```python
#cv02_sift_matching.py
import cv2 as cv              # OpenCV 라이브러리 (이미지 처리)
import matplotlib.pyplot as plt  # 결과 시각화용
import os                    # 파일/폴더 관리

# 두 이미지 경로 설정
img1_path = r"D:/computer-vision/E04_Local Feature/images/mot_color70.jpg"
img2_path = r"D:/computer-vision/E04_Local Feature/images/mot_color83.jpg"

# 결과 저장 폴더 경로
output_path = r"D:/computer-vision/E04_Local Feature/outputs"

# 출력 폴더가 없으면 생성
os.makedirs(output_path, exist_ok=True)

# 이미지 읽기 (BGR 형식)
img1 = cv.imread(img1_path)
img2 = cv.imread(img2_path)

# 이미지 로드 실패 시 예외 처리
if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# SIFT는 grayscale 기반 → 그레이 변환
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift = cv.SIFT_create()

# 각 이미지에서 특징점(keypoints)과 descriptor 추출
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# BFMatcher 생성
# NORM_L2: SIFT descriptor 비교 방식 (유클리드 거리)
# crossCheck=True: 양방향 매칭만 허용 (더 정확하지만 매칭 수는 줄어듦)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

# 두 이미지 descriptor 간 매칭 수행
matches = bf.match(des1, des2)

# 거리(distance) 기준으로 매칭 정렬 (작을수록 좋은 매칭)
matches = sorted(matches, key=lambda x: x.distance)

# 상위 50개 좋은 매칭만 선택
good_matches = matches[:50]

# 매칭 결과 시각화
matched_img = cv.drawMatches(
    img1, kp1,                # 첫 번째 이미지와 특징점
    img2, kp2,                # 두 번째 이미지와 특징점
    good_matches, None,       # 선택된 좋은 매칭들
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  
    # 매칭된 점만 표시 (단독 keypoint는 제외)
)

# 각 이미지의 특징점 개수 출력
print("img1 keypoints:", len(kp1))
print("img2 keypoints:", len(kp2))

# 전체 매칭 개수 출력
print("total matches:", len(matches))

# =========================
# 🔥 이미지 저장
# =========================

# 저장 경로 생성
save_file = os.path.join(output_path, "02_sift_matching.jpg")

# 결과 이미지 저장
cv.imwrite(save_file, matched_img)

# 저장 완료 메시지
print(f"결과 이미지 저장 완료: {save_file}")

# =========================
# 화면 출력
# =========================

# 출력 창 크기 설정
plt.figure(figsize=(18, 8))

# OpenCV는 BGR → matplotlib은 RGB 필요 → 변환 후 출력
plt.imshow(cv.cvtColor(matched_img, cv.COLOR_BGR2RGB))

# 제목 설정
plt.title("SIFT Feature Matching")

# 축 제거
plt.axis("off")

# 레이아웃 정리
plt.tight_layout()

# 화면에 출력
plt.show()
```

</details>

---

# Problem 3 — Homography & Image Alignment

두 이미지 간 특징점을 기반으로 **호모그래피(Homography)**를 계산하고  
한 이미지를 다른 이미지에 맞게 정렬하는 문제입니다.

---

## 주요 과정

1. 이미지 불러오기  
2. SIFT를 이용한 특징점 검출  
3. BFMatcher + knnMatch 수행  
4. Lowe’s Ratio Test 적용 (잘못된 매칭 제거)  
5. RANSAC을 이용한 이상치 제거  
6. Homography 행렬 계산  
7. warpPerspective를 이용한 이미지 정합  

---

## 핵심 개념

- Homography는 두 평면 간 변환을 나타내는 **3x3 행렬**
- RANSAC은 잘못된 매칭(outlier)을 제거하여 안정적인 변환을 계산
- 특징점 기반 정합은 파노라마, AR, 영상 정렬 등에 활용됨

---

## 실행 결과

<figure>
  <img src="https://github.com/dlwldks/computer-vision/blob/main/E04_Local%20Feature/outputs/03_matching_result.jpg" alt="매칭 결과" width="400"/>
  <figcaption>특징점 매칭 결과 (Inlier만 표시)</figcaption>
</figure>

<figure>
  <img src="https://github.com/dlwldks/computer-vision/blob/main/E04_Local%20Feature/outputs/03_panorama-result.png" alt="정합 결과" width="400"/>
  <figcaption>호모그래피 기반 이미지 정합 결과</figcaption>
</figure>

---

## 결과 설명

- 특징점 매칭을 통해 두 이미지 간 대응 관계를 찾음  
- Lowe’s Ratio Test와 RANSAC을 이용하여 잘못된 매칭 제거  
- Homography를 통해 한 이미지를 다른 이미지에 맞게 변환  
- 결과적으로 두 이미지가 하나의 좌표계에서 자연스럽게 정렬됨  

---

<details>
<summary>전체 코드 — cv03_homography_alignment.py</summary>

```python
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
```

</details>

---

