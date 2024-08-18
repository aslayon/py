import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
print(torch.cuda.is_available())  # True가 출력되면 GPU를 정상적으로 사용할 수 있는 것입니다.
import tensorflow as tf
import numpy as np

def to_categorical_custom(y, num_classes):
    return np.eye(num_classes)[y]

# MNIST 데이터 로딩 및 전처리

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = torch.tensor(x_train.reshape(-1, 28*28).astype('float32') / 255)
x_test = torch.tensor(x_test.reshape(-1, 28*28).astype('float32') / 255)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

###########################################################################################################################

class MyDataset(Dataset): #데이터셋 클래스는 3개의 함수가 필수적으로 구현되어야함.
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    


train_dataset = MyDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)# 데이터, 한번에 처리할 샘플 수 ( 6만개를 60개씩 - > 한번의 사이클에서 6천번)  , 셔플 여부

test_dataset = MyDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False)
#파이썬 -> pytorch 패턴 루프, 패턴 한번에 여러개 넣는 데이터 로더
""""
nn.module 의 원형.
class Module(object):
    def __init__(self, *args, **kwargs):
        # Initialize the module here
        pass

    def forward(self, *input):   -------------------------------------------순방향
        # Override this method to define the forward pass
        raise NotImplementedError

    def __call__(self, *input):
        # This method is used to call the forward method
        return self.forward(*input)

    def parameters(self):
        # Return an iterator over module parameters
        return []

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Return a dictionary containing a whole state of the module
        pass

    def load_state_dict(self, state_dict, strict=True):
        # Load the module's state dict
        pass

    def zero_grad(self):
        # Set gradients of all parameters to zero
        pass

    def train(self, mode=True):
        # Set the module in training or evaluation mode
        pass

    def eval(self):
        # Set the module in evaluation mode
        return self.train(False)
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 300)  #y=x⋅W^T+b 즉 .dot + bias // 하지만 입력값이 없음. -> 빈 배열을 생성하는 느낌. 가중치와 바이어스는 초기화 해줌.
        self.fc2 = nn.Linear(300, 100)  # fc 멤버들. // 가중치, 바이어스, 입력수, 출력수, 순전파(y=x⋅W^T+b)
        self.fc3 = nn.Linear(100, 10)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))  # 첫 번째 레이어 후 시그모이드 적용 (x 를 통해 입력값 전달.)
        x = self.sigmoid(self.fc2(x))  # 두 번째 레이어 후 시그모이드 적용
        x = self.fc3(x)  # 출력 레이어
        return x



# 모델, 손실 함수, 옵티마이저
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 학습
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad() # 기울기 초기화
        outputs = model(inputs) # 모델을 통해 예측값 계산 (순전파)
        loss = criterion(outputs, targets)#손실함수 (표준이라는 뜻)
        loss.backward() #loss.grad 에 기울기가 저장됨.
        optimizer.step() # 가중치 업데이트  weight[k] += learning_rate * np.dot(activations_k, deltas_k) #-> (n,m) 윗층의 노드 수  m, 아래층  m
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# 평가

model.eval()
correct_predictions = 0
with torch.no_grad():
    for inputs, targets in train_loader:
        outputs = model(inputs)
        predicted_labels = outputs.argmax(dim=1)
        correct_predictions += (predicted_labels == targets).sum().item()

accuracy = correct_predictions / len(x_train)
print(f"Accuracy on test samples: {accuracy * 100:.2f}%")

model.eval()
correct_predictions = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted_labels = outputs.argmax(dim=1)
        correct_predictions += (predicted_labels == targets).sum().item()

accuracy = correct_predictions / len(x_test)
print(f"Accuracy on test samples: {accuracy * 100:.2f}%")

