import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.saving import load_model

# Load the model
model_path = "/final_brain_tumor_classifier_model.h5"
model = load_model(model_path)

# Define Grad-CAM function
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

# Dataset path
data_dir = "brain-tumor-mri-dataset/Testing"  # Update to your dataset's root path
output_dir = "model visualization/Grad-CAM heatmap/gradcam_results"
os.makedirs(output_dir, exist_ok=True)

# Process all images in the dataset
for subfolder in os.listdir(data_dir):
    subfolder_path = os.path.join(data_dir, subfolder)
    if os.path.isdir(subfolder_path):  # Ensure it's a folder
        output_subfolder = os.path.join(output_dir, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        for img_name in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_name)
            try:
                # Load and preprocess the image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error: Unable to load image at {img_path}. Skipping.")
                    continue

                img_resized = cv2.resize(img, (224, 224))
                img_array = np.expand_dims(img_resized / 255.0, axis=0)

                # Generate Grad-CAM heatmap
                heatmap = get_gradcam(model, img_array)

                # Save the Grad-CAM visualization
                plt.imshow(img_resized)
                plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap
                plt.axis("off")  # Remove axis for cleaner visualization
                output_path = os.path.join(output_subfolder, img_name)
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close()  # Close the figure to avoid memory issues

                print(f"Saved Grad-CAM for {img_path} to {output_path}")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
