import numpy as np
import pandas as pd

n_d = 5
learning_rate = 0.0001  # 학습률 줄임
cycle = 200 * 500  # 몇 번 반복할 것인지.

Y = np.array([47, 40, 50, 60, 50])
X = np.array([50, 37, 45, 20, 31])

w = np.random.rand()  # 초기값을 랜덤으로 설정
b = np.random.rand()  # 초기값을 랜덤으로 설정

for i in range(cycle):
    result = X.dot(w) + b
    loss = np.sum((result - Y) ** 2) / n_d
    
    w_gradient = -2 * np.dot(X.T, (Y - result)) / n_d  # 평균 제곱 오차를 w에 대해 편미분.
    b_gradient = -2 * np.sum(Y - result) / n_d   # b에 대해 편미분
    
    w -= learning_rate * w_gradient  # 편미분 값에 학습률을 곱해 업데이트
    b -= learning_rate



print("Final Weights (w):", w)  # 최종 결과 출력
print("Final Intercept (b):", b)


print(40*w + b)