# 🚦 Traffic Sign Recognition using Deep Learning (CNN)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GYeo1AGNb_jn4WH3WBA0XvUCpc26DPmC?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Deep Learning project to classify **43 different types** of German traffic signs with over **99% accuracy** on the validation set, implemented using TensorFlow/Keras and OpenCV.

## 📝 Project Overview
This project focuses on the **GTSRB (German Traffic Sign Recognition Benchmark)**. Recognizing traffic signs is a critical component for Autonomous Vehicles and Advanced Driver Assistance Systems (ADAS). 

### Key Features:
* **Dataset:** GTSRB (50,000+ images).
* **Architecture:** Multi-layer Convolutional Neural Network (CNN).
* **Preprocessing:** BGR to RGB conversion, image resizing ($30 \times 30$), and normalization.
* **Tools:** Python, TensorFlow, Keras, OpenCV, Matplotlib, Scikit-learn.

---

## 🏗️ Model Architecture
The model is a sequential CNN designed to extract hierarchical features from traffic sign images:
1.  **Block 1:** 2x Conv2D (32 filters, 5x5) + MaxPool + Dropout (0.25)
2.  **Block 2:** 2x Conv2D (64 filters, 3x3) + MaxPool + Dropout (0.25)
3.  **Fully Connected:** Flatten + Dense (256 units, ReLU) + Dropout (0.5)
4.  **Output:** Dense (43 units, Softmax)

---

## 🚀 Performance & Results
The model achieves state-of-the-art performance within just 10-20 epochs:
-   **Validation Accuracy:** ~99.2%
-   **Loss Function:** Categorical Cross-Entropy
-   **Optimizer:** Adam



## 💻 How to Run
1.  **Open in Google Colab:**
    Upload the `.ipynb` file to your Colab environment.
2.  **Dataset Setup:**
    - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
    - Mount your Google Drive and update the file paths in the notebook.
3.  **Train/Test:**
    Run all cells to preprocess data, train the model, and evaluate results.

---

## 📷 Real-World Testing
The model was tested on random traffic sign images from the internet. It successfully identified signs with high confidence scores (>95%).

---

## 🎓 Author
**Mohammad Amin Horri Farahani**
* Artificial Intelligence Project
* Mentors: Erfan Mohammadpour, Niki Mahdian, Mobin Rozati

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
