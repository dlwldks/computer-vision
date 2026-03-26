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