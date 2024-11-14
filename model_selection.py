import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define paths
base_path = 'brain-tumor-mri-dataset'
train_path = os.path.join(base_path, 'Training')
test_path = os.path.join(base_path, 'Testing')

# Define image size and batch size
img_size = (224, 224)
batch_size = 32  # Increased batch size as we're using more data

# Create ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load the full dataset
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


# Function to create model without pre-trained weights
def create_model(model_type):
    input_tensor = Input(shape=img_size + (3,))

    if model_type == 'ResNet50':
        base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)
    elif model_type == 'VGG16':
        base_model = VGG16(weights=None, include_top=False, input_tensor=input_tensor)
    elif model_type == 'InceptionV3':
        base_model = InceptionV3(weights=None, include_top=False, input_tensor=input_tensor)
    else:
        raise ValueError("Invalid model type")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Added dropout for regularization
    outputs = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to evaluate model
def evaluate_model(model, generator):
    predictions = model.predict(generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = generator.classes

    accuracy = accuracy_score(y_true, y_pred)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=generator.class_indices.keys()))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Calculate false positive and false negative rates for each class
    n_classes = len(generator.class_indices)
    fp_rates = []
    fn_rates = []

    for i in range(n_classes):
        fp = np.sum(cm[:i, i]) + np.sum(cm[i + 1:, i])
        fn = np.sum(cm[i, :i]) + np.sum(cm[i, i + 1:])
        tp = cm[i, i]
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]

        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        fp_rates.append(fp_rate)
        fn_rates.append(fn_rate)

    print("\nFalse Positive Rates:")
    for cls, rate in zip(generator.class_indices.keys(), fp_rates):
        print(f"{cls}: {rate:.4f}")

    print("\nFalse Negative Rates:")
    for cls, rate in zip(generator.class_indices.keys(), fn_rates):
        print(f"{cls}: {rate:.4f}")

    return accuracy, np.mean(fp_rates), np.mean(fn_rates)


# List of models to evaluate
models = ['ResNet50', 'VGG16', 'InceptionV3']

# Evaluate each model
results = {}
for name in models:
    print(f"\nEvaluating {name}...")
    model = create_model(name)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=20,  # Increased number of epochs
        validation_data=test_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(test_generator)
    )

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title(f'{name} - Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(f'{name} - Loss')
    plt.legend()
    plt.savefig(f'{name}_training_history.png')
    plt.close()

    accuracy, mean_fp_rate, mean_fn_rate = evaluate_model(model, test_generator)
    results[name] = (accuracy, mean_fp_rate, mean_fn_rate)

# Print results and select the best model
print("\nModel Performance Summary:")
for name, (accuracy, mean_fp_rate, mean_fn_rate) in results.items():
    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Mean False Positive Rate: {mean_fp_rate:.4f}")
    print(f"  Mean False Negative Rate: {mean_fn_rate:.4f}")

# Select the best model based on accuracy
best_model = max(results, key=lambda x: results[x][0])
print(f"\nBest performing model (based on accuracy): {best_model}")