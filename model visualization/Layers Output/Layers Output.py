from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Load model và hình ảnh
model_path = 'final_brain_tumor_classifier_model.h5'
image_path = 'brain-tumor-mri-dataset/Testing/glioma/Te-gl_0010.jpg'
model = load_model(model_path)

# Tiền xử lý hình ảnh
image = load_img(image_path, target_size=(224, 224))  # Resize hình ảnh nếu cần
image_array = img_to_array(image) / 255.0  # Chuẩn hóa về [0, 1]
image_array = np.expand_dims(image_array, axis=0)  # Thêm batch dimension

# Trích xuất đầu ra các layer
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]  # Lấy các layer convolution
feature_extractor = Model(inputs=model.input, outputs=layer_outputs)  # Tạo mô hình trung gian
features = feature_extractor.predict(image_array)

# Hiển thị đầu ra các filter từ một layer cụ thể
def display_all_layers(features, layer_names, num_filters=8):
    """
    Hiển thị đầu ra của tất cả các layer, kèm theo tên layer.

    Args:
        features: Đầu ra của tất cả các layer.
        layer_names: Danh sách tên các layer.
        num_filters: Số filter hiển thị từ mỗi layer (mặc định là 8).
    """
    for layer_index, layer_name in enumerate(layer_names):
        layer_features = features[layer_index]  # Đầu ra của layer hiện tại
        num_filters = min(num_filters, layer_features.shape[-1])  # Giới hạn số filter hiển thị

        # Vẽ output của layer
        plt.figure(figsize=(15, 15))
        plt.suptitle(f"Layer: {layer_name}", fontsize=16, fontweight='bold')  # Tên layer

        for i in range(num_filters):
            plt.subplot(1, num_filters, i + 1)
            plt.imshow(layer_features[0, :, :, i], cmap='viridis')  # Hiển thị filter
            plt.axis('off')
        plt.show()


# Trích xuất tên các layer và đầu ra
layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]  # Lấy các layer convolution
features = feature_extractor.predict(image_array)  # Trích xuất đầu ra các layer

# Hiển thị đầu ra của tất cả các layer
display_all_layers(features, layer_names, num_filters=8)


