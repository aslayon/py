from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import math





weight1 = np.random.rand(2)
b1 = np.random.rand(1)
print("초기 가중치:", weight1,b1)

weight2 = np.random.rand(2)
b2 = np.random.rand(1)
print("초기 가중치:", weight2, b2)

inputs = np.array([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]])
and_ = np.array([0, 0, 0, 1])
or_ = np.array([0, 1, 1, 1])


learning_rate = 0.1
cycle = 10000

for _ in range(cycle):
    for i in range(len(inputs)):
        x = inputs[i]
        target = and_[i]
        tmp = np.dot(x, weight1) + b1[0]
        output = 1 if tmp > 0 else 0
        
        for j in range(len(weight1)):
            weight1[j] += learning_rate * (target-output) * x[j]
        b1[0] += learning_rate * (target - output)

for _ in range(cycle):
    for i in range(len(inputs)):
        x = inputs[i]
        target = or_[i]
        tmp = np.dot(x, weight2) + b2[0]
        output = 1 if tmp > 0 else 0
        
        for j in range(len(weight2)):
            weight2[j] += learning_rate * (target-output) * x[j]
        b2[0] += learning_rate * (target - output)



def AND(x1,x2):
    tmp = x1*weight1[0] + x2*weight1[1] + b1[0]
    #print(tmp)
    if tmp>0:
        return 1
    else :
        return 0

def OR(x1,x2):
    tmp = x1*weight2[0] + x2*weight2[1] + b2[0]
    if tmp>0:
        return 1
    else :
        return 0
    

print("가중치 AND:", weight1, b1)
print("가중치 OR:", weight2, b2)


print("AND(0, 0):", AND(0, 0))
print("AND(0, 1):", AND(0, 1))
print("AND(1, 0):", AND(1, 0))
print("AND(1, 1):", AND(1, 1))

print("OR(0, 0):", OR(0, 0))
print("OR(0, 1):", OR(0, 1))
print("OR(1, 0):", OR(1, 0))
print("OR(1, 1):", OR(1, 1))