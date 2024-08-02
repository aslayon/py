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



#1. network 프로그래밍. input 을 여러개( 다차원 ) -> net 를 구하는데 반복문이 필요(x1*weight1[0] + x2*weight1[1] + b1[0] ) 이렇게 한 줄로 불가능.
# 또 층이 늘어남에 따라 반복문의 반복문이 필요, 중간층의 에러 를 구하는 법 . err1 = w1 * err  //  err2 = w2 * err  // 출력층에서의 err = (t-y) 타겟이 없을 때는 yw

# exclusive or 구현 해봐라. ///////////////// 출력/은닉/입력 -> 1/2/2
#00 0  
#01 1 
#10 1 
#11 0 
##################################################################2/3/2
#00 10
#01 01
#10 01
#11 10


######################################################################
# T ㄷ ㄱ 모양 3*3 /채워져있으면 0.9 비어있으면 0.1  /////////////////   3/5/9
# test data ㄴ 모양.









def BP(inputs, targets, totalnum, num, learning_rate, cycle):
    weight = []
    bias = []

    # 가중치 및 바이어스 초기화
    for i in range(totalnum-1):
        weight.append(np.random.randn(num[i], num[i+1]) * 0.1)  # 작은 랜덤 값으로 초기화
        bias.append(np.zeros((1, num[i+1])))  # 0으로 초기화

    for _ in range(cycle):
        for i in range(len(inputs)):
            x = inputs[i]
            target = targets[i]

            #순전파
            activations = [x]
            for k in range(totalnum-1):
                net_input = np.dot(activations[k], weight[k]) + bias[k]
                activation = sigmoid(net_input)
                activations.append(activation)

            # 출력 계산
            output = activations[-1]
            error = target - output
            delta = error * sigmoid_derivative(output)

            # 역전파
            deltas = [delta]
            for k in range(totalnum-2, 0, -1):
                delta = np.dot(deltas[-1], weight[k].T) * sigmoid_derivative(activations[k])
                deltas.append(delta)
            deltas.reverse()

            #업데이트
            for k in range(totalnum-1):
                activations_k = activations[k].reshape(-1, 1)
                deltas_k = deltas[k].reshape(1, -1)
                weight[k] += learning_rate * np.dot(activations_k, deltas_k)
                bias[k] += learning_rate * deltas[k]

    return weight, bias


inputs = np.array([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]])

xor_ = np.array([0, 1, 1, 0])


totalnum = 3
num = [2, 2, 1]
learning_rate = 0.1
cycle = 100000




#weights, biases = BP(inputs, xor_, totalnum, num, learning_rate, cycle)


def test(inputs, weight, bias):

    x = inputs
        
    for k in range(totalnum-1):    
        output = np.dot(x, weight[k]) + bias[k]
        x = sigmoid(output)
    return sigmoid(output)

# 결과 출력
#print("가중치:")
#print(weights)
#print("바이어스:")
#print(biases)




#for i in range(4):
#    print("xor ", inputs[i],test(inputs[i],weights,biases))


inputs = np.array([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]])

xor_ = np.array([[0,1], [1,0], [1,0], [0,1]])


def BP2(inputs, targets, totalnum, num, learning_rate, cycle):
    weight = []
    bias = []

    # 가중치 및 바이어스 초기화
    for i in range(totalnum-1):
        weight.append(np.random.randn(num[i], num[i+1]) * 0.1)  # 작은 랜덤 값으로 초기화
        bias.append(np.zeros((1, num[i+1])))  # 0으로 초기화

    for _ in range(cycle):
        for i in range(len(inputs)):
            x = inputs[i]
            target = targets[i]

            # 순전파
            activations = [x]
            for k in range(totalnum-1):
                net_input = np.dot(activations[k], weight[k]) + bias[k]
                activation = sigmoid(net_input)
                #print(activation)
                activations.append(activation)

            # 출력 계산
            output = activations[-1]
            error = target - output
            
            delta = error * sigmoid_derivative(output)

            # 역전파 - 출력층에서 은닉층으로
            deltas = [delta]
            for k in range(totalnum-2, 0, -1):
                delta = np.dot(deltas[-1], weight[k].T) * sigmoid_derivative(activations[k])
                deltas.append(delta)
            deltas.reverse()

            # 가중치 및 바이어스 업데이트
            for k in range(totalnum-1):
                activations_k = activations[k].reshape(-1, 1)
                deltas_k = deltas[k].reshape(1, -1)
                weight[k] += learning_rate * np.dot(activations_k, deltas_k)
                bias[k] += learning_rate * deltas[k]

    return weight, bias

totalnum = 3
num = [2, 3, 2]
learning_rate = 0.1
cycle = 100000
#weights, biases = BP2(inputs, xor_, totalnum, num, learning_rate, cycle)
#for i in range(4):
#    print("xor ", inputs[i],test(inputs[i],weights,biases))

######################################################################
# T ㄷ ㄱ 모양 3*3 /채워져있으면 0.9 비어있으면 0.1  /////////////////   3/5/9
# test data ㄴ 모양.

inputs = np.array([[0.9, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1], [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9]])

xor_ = np.array([[1,0,0], [0,1,0], [0,0,1]])

totalnum = 3
num = [9, 5, 3]
learning_rate = 0.1
cycle = 100000
weights, biases = BP2(inputs, xor_, totalnum, num, learning_rate, cycle)
for i in range(3):
    print("xor ", inputs[i],test(inputs[i],weights,biases))

print("xor ", [0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9],test([0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9],weights,biases))
