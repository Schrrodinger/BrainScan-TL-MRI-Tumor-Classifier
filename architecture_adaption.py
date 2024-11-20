from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import vgg16, VGG16_Weights
import os
import json
import numpy as np

# Define transforms
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Create dataset
data_dir = 'brain-tumor-mri-dataset/Testing'
dataset = ImageFolder(root=data_dir, transform=transform)

# Print class names
print("Classes in the dataset:", dataset.classes)

# Create DataLoader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load pre-trained VGG16 model
weights = VGG16_Weights.IMAGENET1K_V1  # You can also use VGG16_Weights.DEFAULT
model = vgg16(weights=weights)

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Replace classifier layers for the number of classes in your dataset
num_classes = len(dataset.classes)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, num_classes)
)

# Adjust the input layer to accept grayscale input (1 channel)
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Early stopping parameters
patience = 5
best_loss = float('inf')
counter = 0

# Training loop
num_epochs = 20
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
    current_lr = scheduler.optimizer.param_groups[0]['lr']
    print(f'Current learning rate: {current_lr}')

    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0

        # Save model in HDF5-like format (PyTorch .pt or .pth format)
        torch.save(model.state_dict(), 'best_brain_tumor_classifier.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

print("Training complete!")

# Load the best model weights
model.load_state_dict(torch.load('best_brain_tumor_classifier.pth'))
model = model.to(device)

# Evaluate the model
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

# Example of using the model to predict a single image
def predict_single_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return dataset.classes[predicted.item()]

# Use the function to predict
image_path = 'brain-tumor-mri-dataset/Training/pituitary/Tr-pi_0012.jpg'
predicted_class = predict_single_image(image_path)
print(f"Predicted class: {predicted_class}")
