# evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,
    balanced_accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, matthews_corrcoef,
    silhouette_score, davies_bouldin_score, average_precision_score, top_k_accuracy_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load dữ liệu và mô hình đã huấn luyện

model = load_model('final_brain_tumor_classifier_model.h5')

test_dir = 'brain-tumor-mri-dataset/Testing'
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
class_labels = list(test_generator.class_indices.keys())


# 2. Classification Metrics
def calculate_classification_metrics(y_true, y_pred, y_pred_prob, class_labels):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("Classification Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC-AUC score (multi-class)
    roc_auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr', average='weighted')
    print("ROC-AUC Score:", roc_auc)


# 3. Regression-like Analysis on Probabilities
def calculate_regression_metrics(y_true, y_pred_prob):
    mse = mean_squared_error(y_true, np.max(y_pred_prob, axis=1))
    mae = mean_absolute_error(y_true, np.max(y_pred_prob, axis=1))
    print("Regression Metrics on Probabilities:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")


# 4. Clustering Metrics
def calculate_clustering_metrics(y_true, y_pred):
    try:
        silhouette = silhouette_score(y_pred.reshape(-1, 1), y_true, metric='euclidean')
        davies_bouldin = davies_bouldin_score(y_pred.reshape(-1, 1), y_true)
        print("Clustering Metrics:")
        print(f"Silhouette Score: {silhouette}")
        print(f"Davies-Bouldin Index: {davies_bouldin}")
    except Exception as e:
        print("Clustering Metrics could not be calculated:", str(e))


# 5. Ranking Metrics
def calculate_ranking_metrics(y_true, y_pred_prob):
    y_true_one_hot = np.eye(len(class_labels))[y_true]

    try:
        avg_precision = average_precision_score(y_true_one_hot, y_pred_prob, average="weighted")
        top_k_acc = top_k_accuracy_score(y_true, y_pred_prob, k=3)
        print("Ranking Metrics:")
        print(f"Average Precision Score: {avg_precision}")
        print(f"Top-3 Accuracy: {top_k_acc}")
    except Exception as e:
        print("Ranking Metrics could not be calculated:", str(e))


# 6. Cross-Domain Metrics
def calculate_cross_domain_metrics(y_true, y_pred):
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    print("Cross-Domain Metrics:")
    print(f"Balanced Accuracy: {balanced_acc}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc}")


# 7. Run all evaluation functions
if __name__ == "__main__":
    calculate_classification_metrics(y_true, y_pred, y_pred_prob, class_labels)
    calculate_classification_metrics(y_true, y_pred, y_pred_prob, class_labels)
    calculate_regression_metrics(y_true, y_pred_prob)
    calculate_clustering_metrics(y_true, y_pred)
    calculate_ranking_metrics(y_true, y_pred_prob)
    calculate_cross_domain_metrics(y_true, y_pred)
# done