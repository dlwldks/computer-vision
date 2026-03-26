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