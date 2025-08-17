# 🩸 Blood Group Detection using Fingerprint with CNN

This project leverages Convolutional Neural Networks (CNNs) to classify **blood groups** based on **fingerprint images**. It explores the feasibility of biometric patterns in predicting medical attributes using deep learning.

---

## 📌 Problem Statement

Traditional blood group identification requires invasive techniques such as blood sampling. This project aims to utilize **biometric fingerprint images** to non-invasively classify a person’s blood group using a Convolutional Neural Network (CNN).

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy / Pandas
- Matplotlib / Seaborn
- Google Colab / Jupyter Notebook

---

## 🧪 Model Architecture

The CNN model includes:
- Convolutional layers with ReLU activations
- MaxPooling layers
- Dropout for regularization
- Fully connected Dense layers
- Softmax output for multi-class classification

> Training and evaluation were conducted on over **6,000 fingerprint images** labeled with respective blood groups.

---

## 📊 Results

The final model achieves strong performance in multi-class classification. Sample metrics:

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| A+    | 0.91      | 0.89   | 0.90     |
| B+    | 0.90      | 0.92   | 0.91     |
| ...   | ...       | ...    | ...      |

Plots like **accuracy curves** and **confusion matrices** are included in the `results/` directory.

---

## 📁 Project Structure

bloodgrp-detection-using-fingerprint-CNN/
│
├── 📁 results/ # Evaluation plots (accuracy, confusion matrix)
│
├── 📁 models/ # Saved model files (.h5 or .pkl)
│
├── 📁 dataset/ # Dataset notes or sample images (not full dataset)
│ └── README.md # Instructions to access the dataset
│
├── 📄 blood_group_cnn.ipynb # Main Jupyter Notebook with code, training, results
├── 📄 final.pdf # Project report explaining problem, methodology, results
├── 📄 requirements.txt # List of dependencies
├── 📄 .gitignore # Common files and folders to ignore in Git
└── 📄 README.md # Project overview and instructions

## 📔 Jupyter Notebook

All preprocessing, model training, and evaluation steps are combined in a single notebook:

👉 [`blood_group_cnn.ipynb`](./blood_group_cnn.ipynb)

This notebook includes:
- Data loading and augmentation
- CNN model architecture
- Training and validation
- Accuracy plots and confusion matrix

---

## 📥 Dataset

The dataset contains over 8,000 fingerprint images for blood group classification. Due to its large size, it is hosted externally. You can download it from the following link:

[Download Dataset](https://drive.google.com/file/d/1YRDuwJ2LNrWk7IpvVrWYz--VgLVjCmAv/view?usp=drive_link)

Please follow the instructions in the `dataset/README.md` for details on how to set up the dataset.
