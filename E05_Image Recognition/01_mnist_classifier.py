# D:\computer-vision\E05_Image Recognition\01_mnist_classifier.py

# TensorFlow 라이브러리를 tf라는 이름으로 불러옴
import tensorflow as tf

# TensorFlow 안의 Keras에서 층(layer)과 모델(model) 구성 도구를 불러옴
from tensorflow.keras import layers, models

# 그래프와 이미지를 화면에 출력하기 위한 라이브러리
import matplotlib.pyplot as plt

# 배열 계산과 argmax 같은 수치 연산을 위한 라이브러리
import numpy as np


# =========================
# 1. MNIST 데이터셋 불러오기
# =========================

# MNIST 손글씨 숫자 데이터셋을 불러옴
# x_train, y_train은 학습용 데이터
# x_test, y_test는 평가용 데이터
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 학습용 이미지 데이터의 크기를 출력
# (60000, 28, 28) = 28x28 크기 이미지 60000장
print("훈련 데이터 shape:", x_train.shape)

# 테스트용 이미지 데이터의 크기를 출력
# (10000, 28, 28) = 28x28 크기 이미지 10000장
print("테스트 데이터 shape:", x_test.shape)


# =========================
# 2. 데이터 확인 (샘플 이미지)
# =========================

# 학습 데이터의 첫 번째 이미지를 흑백(cmap="gray")으로 화면에 표시
plt.imshow(x_train[0], cmap="gray")

# 첫 번째 이미지의 정답 라벨을 제목으로 표시
plt.title(f"Label: {y_train[0]}")

# 픽셀 값 색상 범위를 옆에 색 막대로 표시
plt.colorbar()

# 현재까지 설정한 이미지를 화면에 출력
plt.show()


# =========================
# 3. 데이터 전처리
# =========================

# 학습 이미지 데이터를 float32 타입으로 바꾸고 255로 나누어 0~1 범위로 정규화
x_train = x_train.astype("float32") / 255.0

# 테스트 이미지 데이터도 동일하게 정규화
x_test = x_test.astype("float32") / 255.0

# 학습 데이터의 각 28x28 이미지를 1차원 784칸 벡터로 변환
# -1은 데이터 개수(60000)를 자동으로 맞추라는 뜻
x_train = x_train.reshape(-1, 28 * 28)

# 테스트 데이터도 각 이미지를 784칸 벡터로 변환
x_test = x_test.reshape(-1, 28 * 28)

# 전처리 후 학습 데이터의 shape를 출력
# (60000, 784) 형태가 됨
print("변환 후 shape:", x_train.shape)


# =========================
# 4. 모델 구성 (Dense Neural Network)
# =========================

# Sequential 모델 생성
# 층을 위에서 아래로 순서대로 쌓는 가장 기본적인 모델 구조
model = models.Sequential([

    # 첫 번째 Dense 층 추가
    # 입력 크기는 784, 출력 뉴런 수는 128
    # activation="relu"는 음수는 0으로, 양수는 그대로 통과시키는 활성화 함수
    layers.Dense(128, activation="relu", input_shape=(784,)),

    # 두 번째 Dense 층 추가
    # 이전 층의 출력 128개를 받아 64개 뉴런으로 변환
    layers.Dense(64, activation="relu"),

    # 출력층 추가
    # 숫자 0~9 총 10개 클래스를 분류하므로 뉴런 수는 10
    # softmax는 10개 클래스에 대한 확률값으로 변환
    layers.Dense(10, activation="softmax")
])

# 만들어진 모델의 층 구조와 파라미터 수를 출력
model.summary()


# =========================
# 5. 모델 컴파일
# =========================

# 모델의 학습 방법을 설정
model.compile(
    # optimizer="adam" : 가중치를 업데이트하는 최적화 알고리즘
    optimizer="adam",

    # loss="sparse_categorical_crossentropy" :
    # 정답 라벨이 원-핫 인코딩이 아닌 정수(0~9)일 때 쓰는 다중분류 손실 함수
    loss="sparse_categorical_crossentropy",

    # metrics=["accuracy"] :
    # 학습 중 정확도를 함께 출력하도록 설정
    metrics=["accuracy"]
)


# =========================
# 6. 모델 학습
# =========================

# model.fit()으로 모델 학습 시작
history = model.fit(

    # 입력 데이터: 학습 이미지
    x_train,

    # 정답 데이터: 학습 라벨
    y_train,

    # 전체 학습 데이터를 10번 반복 학습
    epochs=10,

    # 한 번에 64개씩 묶어서 학습
    batch_size=64,

    # 학습 데이터의 10%를 검증(validation)용으로 자동 분리
    validation_split=0.1
)


# =========================
# 7. 모델 평가
# =========================

# 학습이 끝난 모델을 테스트 데이터로 평가
# test_loss에는 손실값, test_acc에는 정확도가 저장됨
test_loss, test_acc = model.evaluate(x_test, y_test)

# 테스트 정확도를 출력
print("\n테스트 정확도:", test_acc)


# =========================
# 8. 예측 테스트
# =========================

# 테스트 데이터 전체에 대해 예측 수행
# 결과는 각 이미지마다 10개 클래스 확률 배열로 나옴
pred = model.predict(x_test)

# 첫 번째 테스트 이미지의 예측 결과에서
# 가장 큰 확률을 가진 인덱스를 선택 = 예측 숫자
predicted_label = np.argmax(pred[0])

# 첫 번째 테스트 이미지의 실제 정답 라벨
true_label = y_test[0]

# 예측값과 실제값을 함께 출력
print(f"\n예측값: {predicted_label}, 실제값: {true_label}")


# =========================
# 9. 결과 시각화
# =========================

# 첫 번째 테스트 이미지를 다시 28x28 모양으로 바꿔서 흑백으로 출력
plt.imshow(x_test[0].reshape(28, 28), cmap="gray")

# 제목에 예측값과 실제값을 함께 표시
plt.title(f"Pred: {predicted_label} / True: {true_label}")

# x축, y축 눈금을 숨김
plt.axis("off")

# 이미지를 화면에 출력
plt.show()


# =========================
# 10. 학습 그래프
# =========================

# 전체 그래프 창 크기를 가로 12, 세로 5로 설정
plt.figure(figsize=(12, 5))

# 1행 2열 중 첫 번째 위치에 그래프 선택
plt.subplot(1, 2, 1)

# 학습 정확도 변화를 선 그래프로 그림
plt.plot(history.history["accuracy"], label="Train")

# 검증 정확도 변화를 선 그래프로 그림
plt.plot(history.history["val_accuracy"], label="Validation")

# 그래프 제목 설정
plt.title("Accuracy")

# 범례 표시
plt.legend()

# 1행 2열 중 두 번째 위치에 그래프 선택
plt.subplot(1, 2, 2)

# 학습 손실값 변화를 선 그래프로 그림
plt.plot(history.history["loss"], label="Train")

# 검증 손실값 변화를 선 그래프로 그림
plt.plot(history.history["val_loss"], label="Validation")

# 그래프 제목 설정
plt.title("Loss")

# 범례 표시
plt.legend()

# 그래프를 화면에 출력
plt.show()