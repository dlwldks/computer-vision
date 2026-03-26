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

<img src="https://github.com/dlwldks/computer-vision/blob/main/E04_Local%20Feature/outputs/01_sift_keypoints.jpg" width="400">

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
  <img src="https://github.com/dlwldks/computer-vision/blob/main/E04_Local%20Feature/outputs/03_warped.jpg" alt="정합 결과" width="400"/>
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
```

</details>

---

