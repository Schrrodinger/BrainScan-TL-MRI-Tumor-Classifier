import tensorflow as tf
from keras import Model
from keras.src.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Load the Pre-trained and Fine-tuned Model or Define New Model with Correct Output
model = load_model('fine_tuned_brain_tumor_classifier.h5')

# 2. Print model layers to find the last convolutional layer
print("Model layers and types:")
for layer in model.layers:
    layer_name = layer.name
    layer_type = type(layer)

    # Use getattr to safely access output_shape if it exists
    layer_output_shape = getattr(layer, 'output_shape', 'N/A')

    print(layer_name, layer_output_shape, layer_type)

# Optional: Compile with a lower learning rate for further fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 2. Set Up Data Generators with Augmentation for Training and Validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of the training data as validation
)

train_generator = train_datagen.flow_from_directory(
    'brain-tumor-mri-dataset/Training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # This is the training subset
)

val_generator = train_datagen.flow_from_directory(
    'brain-tumor-mri-dataset/Training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # This is the validation subset
)

# Verify number of classes detected by data generators
print(f"Classes in training generator: {train_generator.num_classes}")
print(f"Classes in validation generator: {val_generator.num_classes}")

# Adjust model output layer if the number of classes does not match
if model.output_shape[-1] != train_generator.num_classes:
    # Add new output layer with the correct number of classes
    x = model.layers[-2].output  # Connect to the second-to-last layer
    output = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Define Callbacks for Early Stopping and Model Checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# 4. Train the Model with Fine-Tuning
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint]
)

# 5. Evaluate the Model on Test Data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'brain-tumor-mri-dataset/Testing',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy:.2f}')

# 6. Visualize Model's Decisions with Grad-CAM (Optional)
def get_gradcam(model, img_array, layer_name="block5_conv3"):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    guided_grads = (tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads)
    weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
    cam = np.ones(conv_outputs.shape[1:3], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[0, :, :, i]
    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max()
    return heatmap

# Load and display the Grad-CAM heatmap on an example image
img_path = 'brain-tumor-mri-dataset/Testing/pituitary/Te-pi_0010.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Unable to load image at {img_path}. Please check the file path or ensure the image exists.")
else:
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    heatmap = get_gradcam(model, img_array)

    plt.imshow(img_resized)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap
    plt.show()

# 7. Save the Model for Deployment
model.save("final_brain_tumor_classifier_model.h5")
