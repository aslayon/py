import tensorflow as tf

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
#import pickle
#import os

#mnist 784/400/100/10  .dot  은닉층 h1,2,3  층 마다 w1,w2,w3




#def save_to_file(data, filename):
#    with open(filename, 'wb') as f:
#        pickle.dump(data, f)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def BP2(inputs, targets, layer_num, Node, learning_rate, cycle, tolerance=1e-5, patience=5, save_path='./model_data'):
    weight = []
    bias = []

    # 가중치 및 바이어스 초기화
    for i in range(layer_num-1):
        weight.append(np.random.randn(Node[i], Node[i+1]) * 0.1)
        bias.append(np.zeros((1, Node[i+1])))

    best_weights = [w.copy() for w in weight]
    best_biases = [b.copy() for b in bias]
    best_loss = float('inf')
    epochs_without_improvement = 0

    # 모델 저장 경로 생성
    #if not os.path.exists(save_path):
    #    os.makedirs(save_path)

    for epoch in range(cycle):
        total_loss = 0
        epoch_activations = []  # 각 에포크의 활성화 값을 저장하기 위한 리스트

        for i in range(len(inputs)):
            x = inputs[i]
            target = targets[i]

            # 순전파
            activations = [x]
            for k in range(layer_num-1):
                net_input = np.dot(activations[k], weight[k]) + bias[k] # (1,n) (n,m) -> (1,m)   n 이 아래층
                activation = sigmoid(net_input)
                activations.append(activation)
            
            epoch_activations.append(activations)  # 활성화 값 저장

            # 출력 계산
            output = activations[-1]
            error = target - output
            total_loss += np.sum(np.square(error))

            delta = error * sigmoid_derivative(output)

            # 역전파
            deltas = [delta]
            for k in range(layer_num-2, 0, -1):
                delta = np.dot(deltas[-1], weight[k].T) * sigmoid_derivative(activations[k])
                deltas.append(delta)
            deltas.reverse()

            # 가중치 및 바이어스 업데이트
            for k in range(layer_num-2, -1, -1):
                activations_k = activations[k].reshape(-1, 1) #(n,1)
                deltas_k = deltas[k].reshape(1, -1) #(1,m) 
                weight[k] += learning_rate * np.dot(activations_k, deltas_k) #-> (n,m) 윗층의 노드 수  m, 아래층  m
                bias[k] += learning_rate * deltas[k]

        # 가중치 및 활성화 값을 파일로 저장
        #save_to_file(weight, f"{save_path}/weights_epoch_{epoch+1}.pkl")
        #save_to_file(epoch_activations, f"{save_path}/activations_epoch_{epoch+1}.pkl")

        # 평균 손실 계산
        avg_loss = total_loss / len(inputs)
        print(f"Epoch {epoch+1}/{cycle}, Loss: {avg_loss}")

        # 조기 종료 조건 확인
        if avg_loss < best_loss - tolerance:
            best_loss = avg_loss
            best_weights = [w.copy() for w in weight]
            best_biases = [b.copy() for b in bias]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("fin")
                break

    print("Training complete")
    return best_weights, best_biases

def test(x, weights, biases):
    activations = [x]
    for k in range(len(weights)):
        net_input = np.dot(activations[k], weights[k]) + biases[k]
        activation = sigmoid(net_input)
        activations.append(activation)
    return activations[-1]

def to_categorical_custom(y, num_classes):
    return np.eye(num_classes)[y]

# MNIST 데이터 로딩 및 전처리

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255 #0~255 사이의 입력값을 0과 1 사이로 변경(부동소수점 까지) _입력에 대한 전처리
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255 #0~255 사이의 입력값을 0과 1 사이로 변경(부동소수점 까지)
y_train = to_categorical_custom(y_train, 10)
y_test = to_categorical_custom(y_test, 10)

###########################################################################################################################

totalnum = 4
num = [784, 300,100, 10] 
learning_rate = 0.1
cycle = 2 #------------------------------------------------------------------총 반복 수

# 학습
weights, biases = BP2(x_train, y_train, totalnum, num, learning_rate, cycle)

# 예측 및 정확도 평가

correct_predictions = 0
for i in range(60000):  #
    prediction = test(x_train[i], weights, biases)
    true_label = np.argmax(y_train[i]) #10개의 출력값중 가장 높은 값 반환( 확률 높은 얘)
    predicted_label = np.argmax(prediction)
    if true_label == predicted_label:
        correct_predictions += 1

accuracy = correct_predictions / 60000
print(f"Accuracy on 60000 test samples: {accuracy * 100:.2f}%")


correct_predictions = 0
for i in range(10000):  
    prediction = test(x_test[i], weights, biases)
    true_label = np.argmax(y_test[i])
    predicted_label = np.argmax(prediction)
    if true_label == predicted_label:
        correct_predictions += 1
        #print(true_label)
        #print(predicted_label)

accuracy = correct_predictions / 10000
print(f"Accuracy on 10000 test samples: {accuracy * 100:.2f}%")


#Epoch 1/100, Loss: 0.19193707082687256
#Epoch 2/100, Loss: 0.08843381637251038
#Epoch 3/100, Loss: 0.06412976960445319
#Epoch 4/100, Loss: 0.05026349026340248
#Epoch 5/100, Loss: 0.04119056744120321
#Epoch 6/100, Loss: 0.0346296603930859
#Epoch 7/100, Loss: 0.029690133393363842
#Epoch 8/100, Loss: 0.02579793104981896
#Epoch 9/100, Loss: 0.02255332552310857
#Epoch 10/100, Loss: 0.019763508893447263
#Epoch 11/100, Loss: 0.017381797260902887

#파이썬 -> pytorch 패턴 루프, 패턴 한번에 여러개 넣는 데이터 로더