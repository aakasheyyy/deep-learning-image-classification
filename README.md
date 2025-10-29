COMPANY NAME: CODTECH IT SOLUTIONS

NAME:AAKASH R

INTERN ID:CT04DR985

DOMAIN:DATA SCIENCE

DURATION: 4 WEEKS

MENTOR:NEELA SANTHOSH

# deep-learning-image-classification
A CNN-based deep learning model that classifies handwritten digits using TensorFlow (MNIST dataset)
# 🧠 Deep Learning Image Classification (MNIST)

A Convolutional Neural Network (CNN) built using TensorFlow to classify handwritten digits from the MNIST dataset.

## 📘 Project Overview
This project trains a CNN model to recognize digits (0–9) using the MNIST dataset.  
It achieves **~99% validation accuracy** and demonstrates how deep learning can interpret visual data.

## 🧩 Features
- Built with TensorFlow & Keras  
- Uses CNN architecture (Conv2D, MaxPooling, Dense)  
- Achieves high accuracy on unseen data  
- Includes visualizations of accuracy & loss  

## 📊 Results
| Metric | Accuracy |
|--------|-----------|
| Training Accuracy | 99.28% |
| Validation Accuracy | 98.97% |

## 📷 Model Visualization
![Sample Output](sample_predictions.png)

## 🚀 How to Run
1. Open the `.ipynb` notebook in Google Colab  
2. Run each cell step by step  
3. Model file `mnist_cnn_model.h5` can be reloaded using:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model("mnist_cnn_model.h5")
📦 Requirements
TensorFlow 2.x

NumPy

Matplotlib

OUTPUT
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/094e6865-6ac3-49a6-82e8-98f6c5fcd10a" />
