import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define data directories
base_dir = 'brain-tumor-mri-dataset'  # Update this path to your dataset location
train_dir = os.path.join(base_dir, 'Training')
test_dir = os.path.join(base_dir, 'Testing')

# Image parameters
img_width, img_height = 224, 224
batch_size = 32

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Only rescaling for validation/test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and iterate training data in batches
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load and iterate test data in batches
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Assuming you've already run the model selection script and have determined the best model
best_model_name = 'VGG16'  # Replace with your actual best model name

def create_fine_tuning_model(model_name, num_classes=4):
    if model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    elif model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    else:
        raise ValueError("Invalid model name")

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    return model

# Create and compile the fine-tuning model
fine_tuning_model = create_fine_tuning_model(best_model_name)
fine_tuning_model.compile(optimizer=Adam(learning_rate=0.0001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Fine-tune the model
history = fine_tuning_model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the fine-tuned model
test_loss, test_accuracy = fine_tuning_model.evaluate(test_generator)
print(f"Test accuracy after fine-tuning: {test_accuracy:.4f}")

# Optional: Unfreeze some layers of the base model for further fine-tuning
def unfreeze_model(model):
    # We'll unfreeze the last 30 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-30:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    return model

fine_tuning_model = unfreeze_model(fine_tuning_model)
fine_tuning_model.compile(optimizer=Adam(learning_rate=1e-5),  # Lower learning rate
                          loss='categorical_crossentropy',
                                        metrics=['accuracy'])

# Further fine-tuning
history = fine_tuning_model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Final evaluation
test_loss, test_accuracy = fine_tuning_model.evaluate(test_generator)
print(f"Final test accuracy after fine-tuning: {test_accuracy:.4f}")

# Save the fine-tuned model
fine_tuning_model.save('fine_tuned_brain_tumor_classifier.h5')