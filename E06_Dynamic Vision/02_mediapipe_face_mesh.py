# OpenCV 라이브러리를 cv라는 이름으로 불러오기 (영상 처리용)
import cv2 as cv

# MediaPipe 라이브러리를 mp라는 이름으로 불러오기 (얼굴 인식용)
import mediapipe as mp


# =========================
# 1. Mediapipe FaceMesh 초기화
# =========================

# MediaPipe의 얼굴 랜드마크(FaceMesh) 모듈을 가져옴
mp_face_mesh = mp.solutions.face_mesh

# FaceMesh 객체 생성 (얼굴 추적기)
face_mesh = mp_face_mesh.FaceMesh(

    static_image_mode=False,      
    # False → 영상(실시간) 모드 / True → 이미지 1장 처리

    max_num_faces=1,              
    # 동시에 추적할 최대 얼굴 수 (여기서는 1명)

    refine_landmarks=True,        
    # 눈, 입술 등 더 정밀한 랜드마크 추가

    min_detection_confidence=0.5, 
    # 얼굴을 "이게 얼굴이다"라고 판단하는 최소 신뢰도

    min_tracking_confidence=0.5   
    # 이미 찾은 얼굴을 계속 추적할 때 필요한 최소 신뢰도
)


# =========================
# 2. 웹캠 열기
# =========================

# 기본 웹캠(0번 카메라) 열기
cap = cv.VideoCapture(0)

# 웹캠이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")  # 실패 시 메시지 출력
    raise SystemExit              # 프로그램 강제 종료

# 실행 안내 메시지 출력
print("웹캠 실행 중... 종료하려면 ESC 키를 누르세요.")


# =========================
# 3. 프레임 반복 처리
# =========================

# 무한 루프 → 영상 계속 읽기
while True:

    # 웹캠에서 한 프레임 읽기
    ret, frame = cap.read()

    # 프레임을 제대로 못 읽었으면 종료
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 좌우 반전 (셀카처럼 보이게)
    frame = cv.flip(frame, 1)

    # OpenCV는 BGR, Mediapipe는 RGB → 색상 변환 필요
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # 처리 속도를 높이기 위해 메모리 쓰기 금지
    rgb_frame.flags.writeable = False

    # 얼굴 랜드마크 추출 실행
    results = face_mesh.process(rgb_frame)

    # 다시 쓰기 가능 상태로 변경
    rgb_frame.flags.writeable = True


    # 얼굴이 검출된 경우만 실행
    if results.multi_face_landmarks:

        # 현재 프레임의 높이, 너비 가져오기
        h, w, _ = frame.shape

        # 각 얼굴에 대해 반복 (여기선 1개)
        for face_landmarks in results.multi_face_landmarks:

            # 얼굴의 468개 랜드마크 하나씩 반복
            for landmark in face_landmarks.landmark:

                # (0~1 사이 값) → 실제 픽셀 좌표로 변환
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # 해당 위치에 초록색 점 찍기
                cv.circle(frame, (x, y), 1, (0, 255, 0), -1)


    # 화면에 안내 문구 출력
    cv.putText(
        frame,                        # 출력할 이미지
        "Press ESC to exit",          # 텍스트 내용
        (10, 30),                     # 위치 (x, y)
        cv.FONT_HERSHEY_SIMPLEX,      # 폰트 종류
        0.8,                          # 글자 크기
        (0, 255, 255),                # 색상 (노랑)
        2                             # 두께
    )

    # 결과 영상 창에 출력
    cv.imshow("Mediapipe FaceMesh - 468 Landmarks", frame)

    # 키 입력 대기 (1ms)
    key = cv.waitKey(1) & 0xFF

    # ESC 키(27)를 누르면 종료
    if key == 27:
        break


# =========================
# 4. 자원 해제
# =========================

# 웹캠 해제
cap.release()

# 모든 OpenCV 창 닫기
cv.destroyAllWindows()

# MediaPipe 객체 종료 (메모리 정리)
face_mesh.close()