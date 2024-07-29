from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#delta 룰 / 시그모이드 함수

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


# feed forward/ err 계산/ 기울기 계산(y(1-y))  -- feed back / 0.1~0.9 , 0.1~ 0.9 (9 * 9) matplot 으로 3차원 결과 확인
weight1 = np.random.rand(2)
b1 = np.random.rand(1)
print("초기 가중치:", weight1,b1)

weight2 = np.random.rand(2)
b2 = np.random.rand(1)
print("초기 가중치:", weight2, b2)

inputs = np.array([[x1, x2] for x1 in np.arange(0.1, 1.0, 0.1) for x2 in np.arange(0.1, 1.0, 0.1)])
and_ = np.array([int(x1 > 0.5 and x2 > 0.5) for x1, x2 in inputs])
or_ = np.array([int(x1 > 0.5 or x2 > 0.5) for x1, x2 in inputs])


learning_rate = 0.1
cycle = 10000

for _ in range(cycle):
    for i in range(len(inputs)):
        x = inputs[i]
        target = and_[i]
        tmp = np.dot(x, weight1) + b1[0]
        output = sigmoid(tmp)
        delta = sigmoid_derivative(output)    


        for j in range(len(weight1)):
            weight1[j] += learning_rate * delta*(target-output) * x[j]
        b1[0] += learning_rate * (target - output)

for _ in range(cycle):
    for i in range(len(inputs)):
        x = inputs[i]
        target = and_[i]
        tmp = np.dot(x, weight1) + b1[0]
        output = sigmoid(tmp)
        delta = sigmoid_derivative(output)    


        for j in range(len(weight1)):
            weight1[j] += learning_rate * delta*(target-output) * x[j]
        b1[0] += learning_rate * (target - output)



def AND(x1,x2):
    tmp = x1*weight1[0] + x2*weight1[1] + b1[0]
    
    #print(tmp)
    return sigmoid(tmp)

def OR(x1,x2):
    tmp = x1*weight2[0] + x2*weight2[1] + b2[0]
    return sigmoid(tmp)
    

print("가중치 AND:", weight1, b1)
print("가중치 OR:", weight2, b2)


fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')

x1 = inputs[:, 0]
x2 = inputs[:, 1]
tmp = np.array([AND(x1[i],x2[i]) for i in range(len(x1))])
print(tmp)
ax.scatter(x1, x2, tmp)

bx = fig.add_subplot(212, projection='3d')
tmp = np.array([OR(x1[i],x2[i]) for i in range(len(x1))])
print(tmp)
bx.scatter(x1, x2, tmp)

plt.show()
