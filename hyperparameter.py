import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import optuna
from torch.utils.data import DataLoader, random_split

def build_model(learning_rate, dropout_rate, dense_units):
    model = models.vgg16(weights='IMAGENET1K_V1')  # Updated to use weights argument, as 'pretrained' is deprecated
    for param in model.features.parameters():
        param.requires_grad = False  # Freeze base model layers

    # Modify classifier layers
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_features, dense_units),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(dense_units, len(train_data.classes)),
        nn.Softmax(dim=1)
    )

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    return model, optimizer, loss_fn

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    dense_units = trial.suggest_int('dense_units', 128, 512, step=64)

    model, optimizer, loss_fn = build_model(learning_rate, dropout_rate, dense_units)
    model = model.to(device)

    num_epochs = 10
    best_val_accuracy = 0
    patience = 3
    counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total

        # Track best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            counter = 0  # Reset early stopping counter
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        print(f"Epoch {epoch + 1}: Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return best_val_accuracy


if __name__ == "__main__":
    # Paths to your directories
    train_dir = 'brain-tumor-mri-dataset/Training'
    val_dir = 'brain-tumor-mri-dataset/Testing'

    # Transformations for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pretrained models
    ])

    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)

    # Split validation data
    val_size = int(0.2 * len(train_data))
    train_size = len(train_data) - val_size
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

    # Create DataLoadersCom
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=6)

    # Print the best hyperparameters
    print("Best hyperparameters found:", study.best_params)
