# E05. Image Recognition

컴퓨터비전 실습 - 이미지 분류 (MNIST + CIFAR-10)

---

## 📌 과제 개요

본 실습은 두 가지 이미지 분류 문제를 구현한다.

### 1️⃣ MNIST 손글씨 숫자 분류
- Dense Neural Network 사용
- 28x28 흑백 이미지
- 0~9 숫자 분류

### 2️⃣ CIFAR-10 이미지 분류
- CNN (Convolutional Neural Network) 사용
- 32x32 RGB 이미지
- 10개 클래스 분류 (dog 포함)

---

## 🧠 01. MNIST 분류 (Dense Neural Network)

<details>
<summary>🔍 코드 보기</summary>

```python
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
```

</details>

---

### 📊 데이터 확인

![mnist_sample](https://github.com/dlwldks/computer-vision/blob/main/E05_Image%20Recognition/outputs/01_train data.png)

---

### 📈 학습 결과

![mnist_graph](https://github.com/dlwldks/computer-vision/blob/main/E05_Image%20Recognition/outputs/01_loss accuracy.png)

- **Test Accuracy:** `0.9758`

---

### 🔍 예측 결과

![mnist_pred](https://github.com/dlwldks/computer-vision/blob/main/E05_Image%20Recognition/outputs/01_image predict.png)

예측값: 7  
실제값: 7  

---

## 🧠 02. CIFAR-10 CNN 분류

<details>
<summary>🔍 코드 보기</summary>

```python
# D:\computer-vision\E05_Image Recognition\02_cifar10_cnn_dog_classifier.py
# → 현재 파일의 전체 경로 (코드 설명용 주석)

# 운영체제 관련 기능 (파일 존재 확인 등)
import os  # 파일 경로 존재 여부 확인, 시스템 종료 등에 사용

# OpenCV (이미지 읽기, 색상 변환, 리사이즈)
import cv2  # 이미지 처리 라이브러리

# 수치 계산 (배열 처리)
import numpy as np  # 배열 연산 및 argmax 등에 사용

# 그래프 시각화
import matplotlib.pyplot as plt  # 그래프, 이미지 출력

# 딥러닝 프레임워크
import tensorflow as tf  # 전체 딥러닝 엔진

# Keras 모델 구성 요소
from tensorflow.keras import layers, models  # 레이어와 모델 생성 도구

# CIFAR-10 데이터셋 로드용
from tensorflow.keras.datasets import cifar10  # CIFAR-10 데이터셋 불러오기


# =========================
# 1. 경로 설정
# =========================

# 테스트할 이미지 경로 (dog 이미지)
test_image_path = r"D:\computer-vision\E05_Image Recognition\images\dog.jpg"
# → 나중에 모델로 예측할 실제 이미지 파일 경로

# CIFAR-10 데이터셋의 클래스 이름 (총 10개)
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
# → 모델 출력값 index(0~9)를 실제 클래스 이름으로 변환하기 위한 리스트


# =========================
# 2. CIFAR-10 데이터 로드
# =========================

# 학습 데이터와 테스트 데이터 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# → 이미지 데이터 + 정답(label) 동시에 로드

# 데이터 형태 출력 (확인용)
print(f"x_train shape: {x_train.shape}")  
# → (50000, 32, 32, 3) = 32x32 RGB 이미지 50000장

print(f"y_train shape: {y_train.shape}")  
# → (50000, 1) = 각 이미지의 정답 라벨

print(f"x_test shape:  {x_test.shape}")   
# → (10000, 32, 32, 3)

print(f"y_test shape:  {y_test.shape}")   
# → (10000, 1)


# =========================
# 3. 데이터 전처리
# =========================

# 정수(0~255)를 float로 변환 후 255로 나눠서 0~1로 정규화
x_train = x_train.astype("float32") / 255.0  
# → 학습 안정성을 위해 정규화

x_test = x_test.astype("float32") / 255.0  
# → 테스트 데이터도 동일하게 처리


# =========================
# 4. CNN 모델 구성
# =========================

# Sequential 모델 생성 (층을 순서대로 쌓는 구조)
model = models.Sequential([

    # 첫 번째 합성곱 층
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
    # → 32개의 필터로 특징 추출
    # → 입력 이미지 크기: 32x32x3

    # 풀링 층
    layers.MaxPooling2D((2, 2)),
    # → 크기를 절반으로 줄여서 연산량 감소

    # 두 번째 합성곱 층
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    # → 더 많은 필터로 복잡한 특징 학습

    layers.MaxPooling2D((2, 2)),

    # 세 번째 합성곱 층
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    # → 더 깊은 특징 추출

    layers.MaxPooling2D((2, 2)),

    # 2D 데이터를 1D로 변환
    layers.Flatten(),
    # → Dense층 입력을 위해 평탄화

    # 완전 연결층
    layers.Dense(128, activation="relu"),
    # → 특징을 종합해서 학습

    # Dropout
    layers.Dropout(0.3),
    # → 30% 뉴런을 랜덤으로 끄면서 과적합 방지

    # 출력층
    layers.Dense(10, activation="softmax")
    # → 10개 클래스 확률 출력
])

# 모델 구조 출력
model.summary()  
# → 층별 파라미터 확인


# =========================
# 5. 모델 컴파일
# =========================

model.compile(
    optimizer="adam",  
    # → 가중치 업데이트 방식

    loss="sparse_categorical_crossentropy",  
    # → 다중 클래스 분류 손실 함수

    metrics=["accuracy"]  
    # → 정확도 출력
)


# =========================
# 6. 모델 학습
# =========================

history = model.fit(
    x_train, y_train,        
    # → 학습 데이터

    epochs=10,               
    # → 전체 데이터 10번 반복

    batch_size=64,           
    # → 64개씩 나눠서 학습

    validation_split=0.1,    
    # → 10%는 검증 데이터

    verbose=1                
    # → 학습 로그 출력
)


# =========================
# 7. 테스트 세트 평가
# =========================

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
# → 학습 안 한 데이터로 성능 평가

print(f"\n테스트 정확도: {test_acc:.4f}")  
# → 정확도 출력

print(f"테스트 손실: {test_loss:.4f}")  
# → 손실값 출력


# =========================
# 8. 학습 결과 그래프
# =========================

plt.figure(figsize=(12, 5))  
# → 그래프 창 크기 설정

plt.subplot(1, 2, 1)  
# → 1행 2열 중 첫 번째 그래프

plt.plot(history.history["accuracy"], label="Train Accuracy")  
# → 학습 정확도

plt.plot(history.history["val_accuracy"], label="Val Accuracy")  
# → 검증 정확도

plt.title("Accuracy")  
plt.xlabel("Epoch")  
plt.ylabel("Accuracy")  
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(history.history["loss"], label="Train Loss")  
# → 학습 손실

plt.plot(history.history["val_loss"], label="Val Loss")  
# → 검증 손실

plt.title("Loss")  
plt.xlabel("Epoch")  
plt.ylabel("Loss")  
plt.legend()

plt.tight_layout()  
# → 그래프 겹침 방지

plt.show()  
# → 출력


# =========================
# 9. dog.jpg 예측
# =========================

if not os.path.exists(test_image_path):  
# → 파일 존재 여부 확인
    print(f"\n테스트 이미지가 존재하지 않습니다: {test_image_path}")
    raise SystemExit  # 프로그램 종료

img_bgr = cv2.imread(test_image_path)  
# → 이미지 읽기 (BGR 형식)

if img_bgr is None:  
# → 이미지 로드 실패 체크
    print("\n이미지를 불러올 수 없습니다.")
    raise SystemExit

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  
# → RGB로 변환 (matplotlib용)

img_resized = cv2.resize(img_rgb, (32, 32))  
# → 모델 입력 크기에 맞게 변경

img_input = img_resized.astype("float32") / 255.0  
# → 정규화

img_input = np.expand_dims(img_input, axis=0)  
# → (1, 32, 32, 3) 형태로 변환 (배치 추가)


# =========================
# 9-2. 예측 수행
# =========================

pred = model.predict(img_input, verbose=0)  
# → 모델로 예측

pred_index = np.argmax(pred[0])  
# → 가장 높은 확률의 클래스 선택

pred_label = class_names[pred_index]  
# → 클래스 이름 변환

pred_conf = pred[0][pred_index]  
# → 해당 확률 값


# =========================
# 9-3. 결과 출력
# =========================

print("\n=== dog.jpg 예측 결과 ===")
print(f"예측 클래스: {pred_label}")
print(f"신뢰도: {pred_conf:.4f}")

print("\n클래스별 예측 확률:")

for i, prob in enumerate(pred[0]):  
# → 모든 클래스 확률 출력 반복문
    print(f"{class_names[i]:>10s}: {prob:.4f}")


# =========================
# 10. 이미지 시각화
# =========================

plt.figure(figsize=(5, 5))  
# → 출력 창 크기

plt.imshow(img_rgb)  
# → 이미지 출력

plt.title(f"Prediction: {pred_label} ({pred_conf:.4f})")  
# → 예측 결과 표시

plt.axis("off")  
# → 축 제거

plt.show()  
# → 화면 출력
```

</details>

---

### 📈 학습 결과

![cifar_graph](https://github.com/dlwldks/computer-vision/blob/main/E05_Image%20Recognition/outputs/02_loss accuracy.png)

- **Test Accuracy:** `0.7242`
- **Test Loss:** `0.8511`

---

### 🐶 이미지 예측 결과

![dog_pred](https://github.com/dlwldks/computer-vision/blob/main/E05_Image%20Recognition/outputs/02_image predict.png)

예측 클래스: dog  
신뢰도: 0.6103  

---

## 📊 결과 비교

| 모델 | 데이터셋 | 정확도 |
|------|---------|--------|
| Dense NN | MNIST | 97.58% |
| CNN | CIFAR-10 | 72.42% |

---

## 💡 느낀점

- MNIST는 단순한 구조(Dense)로도 높은 성능 달성 가능
- CIFAR-10은 복잡한 이미지 → CNN 필요
- CNN에서도 더 깊은 구조 또는 데이터 증강이 필요할 수 있음

---

## 🚀 결론

- 이미지 분류에서 데이터 복잡도에 따라 모델 선택이 중요
