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
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

</details>

---

### 📊 데이터 확인

![mnist_sample](./images/01_train data.png)

---

### 📈 학습 결과

![mnist_graph](./images/01_loss accuracy.png)

- **Test Accuracy:** `0.9758`

---

### 🔍 예측 결과

![mnist_pred](./images/01_image predict.png)

예측값: 7  
실제값: 7  

---

## 🧠 02. CIFAR-10 CNN 분류

<details>
<summary>🔍 코드 보기</summary>

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])
```

</details>

---

### 📈 학습 결과

![cifar_graph](./images/02_loss accuracy.png)

- **Test Accuracy:** `0.7242`
- **Test Loss:** `0.8511`

---

### 🐶 이미지 예측 결과

![dog_pred](./images/02_image predict.png)

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
