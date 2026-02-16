import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  


DATASET_PATH = "content/dataset/Train"

def load_images(dataset_path):
    images = []
    labels = []
    for class_id in sorted(os.listdir(dataset_path)):
        class_folder = os.path.join(dataset_path, class_id)
        if not os.path.isdir(class_folder):
            continue
        for file_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, file_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (32, 32))
                images.append(img)
                labels.append(int(class_id))
    return np.array(images), np.array(labels)


X, y = load_images(DATASET_PATH)


def extract_features(images):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # HOG
        hog_features = hog(gray, orientations=9, pixels_per_cell=(8,8),
                           cells_per_block=(2,2), block_norm='L2-Hys', visualize=False)
        # Color histogram
        hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        # ترکیب ویژگی‌ها
        feature_vector = np.hstack([hog_features, hist])
        features.append(feature_vector)
    return np.array(features)

X_features = extract_features(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print("\nSVM Classification Report:\n")
print(classification_report(y_test, y_pred_svm))


joblib.dump(svm_model, "SVM_model.pkl")
joblib.dump(scaler, "SVM_scaler.pkl")
print("\nModels saved: svm_model.pkl , svm_scaler.pkl")





acc = accuracy_score(y_test, y_pred_svm)

cm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues", cbar=True)
plt.title(f"SVM Confusion Matrix\nAccuracy: {acc:.4f}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()



report = classification_report(y_test, y_pred_svm, output_dict=True)

f1_scores = [report[str(i)]['f1-score'] for i in range(len(np.unique(y_test)))]

plt.figure(figsize=(14,6))
plt.bar(range(len(f1_scores)), f1_scores)
plt.title("F1-Score per Class (SVM)")
plt.xlabel("Class")
plt.ylabel("F1-Score")
plt.ylim(0,1)
plt.show()