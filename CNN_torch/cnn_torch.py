import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn.functional as F

# 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# MNIST 데이터셋 로드
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)  # Flatten
        x = self.sigmoid(self.fc1(x))  # 은닉층 1
        x = self.sigmoid(self.fc2(x))  # 은닉층 2
        x = self.fc3(x)  # 출력층
        return x

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델, 손실 함수, 최적화 도구 설정
model = CNN().to(device)
criterion = nn.MSELoss()  # MSE Loss로 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)  # SGD로 설정

# 원-핫 인코딩 함수 (디바이스 인식)
def one_hot_encode(labels, num_classes=10, device=device):
    return torch.eye(num_classes, device=device)[labels]

# 학습
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target_one_hot = one_hot_encode(target, device=device)  # 원-핫 인코딩
        optimizer.zero_grad()  # 기울기 초기화
        output = model(data)  # 순전파
        loss = criterion(output, target_one_hot)  # loss 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 테스트
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        target_one_hot = one_hot_encode(target, device=device)  # 원-핫 인코딩
        output = model(data)
        test_loss += criterion(output, target_one_hot).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print(f'Test set: Average loss: {test_loss:.8f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
