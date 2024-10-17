from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
import numpy as np

# Định nghĩa transforms
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Tạo dataset
data_dir = 'brain-tumor-mri-dataset/Testing'
dataset = ImageFolder(root=data_dir, transform=transform)

# In ra tên các lớp để kiểm tra
print("Các lớp trong dataset:", dataset.classes)

# Tạo DataLoader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Tải mô hình ResNet50 pre-trained
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Đóng băng các lớp của mô hình
for param in model.parameters():
    param.requires_grad = False

# Thay thế lớp fully connected cuối cùng
num_classes = len(dataset.classes)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)

# Điều chỉnh lớp đầu vào cho ảnh grayscale
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Chuyển mô hình sang GPU nếu có
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Định nghĩa loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Early stopping parameters
patience = 10
best_loss = float('inf')
counter = 0

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Learning rate scheduling
    scheduler.step(epoch_loss)

    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0

        # Chuyển state_dict thành dictionary Python
        state_dict = model.state_dict()
        model_weights = {name: param.cpu().numpy().tolist() for name, param in state_dict.items()}

        # Lưu trọng số vào file JSON
        with open('best_brain_tumor_classifier_weights.json', 'w') as f:
            json.dump(model_weights, f)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

print("Training complete!")

# Đánh giá mô hình

# Đọc trọng số từ file JSON
with open('best_brain_tumor_classifier_weights.json', 'r') as f:
    model_weights = json.load(f)

# Tạo state dict mới cho mô hình
new_state_dict = {}

# Chuyển đổi dữ liệu JSON thành PyTorch tensors
for name, params in model_weights.items():
    new_state_dict[name] = torch.FloatTensor(params).to(device)

# Load state dict vào mô hình
model.load_state_dict(new_state_dict)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on dataset: {100 * correct / total:.2f}%')


# Ví dụ về cách sử dụng mô hình để dự đoán một ảnh đơn
def predict_single_image(image_path):
    # Load và tiền xử lý ảnh
    image = Image.open(image_path).convert('L')  # Chuyển sang ảnh grayscale
    image = transform(image).unsqueeze(0).to(device)  # Thêm chiều batch và chuyển sang device

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return dataset.classes[predicted.item()]

# Sử dụng hàm để dự đoán
image_path = 'brain-tumor-mri-dataset/Training/pituitary/Tr-pi_0012.jpg'
predicted_class = predict_single_image(image_path)
print(f"Predicted class: {predicted_class}")