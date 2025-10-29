COMPANY NAME: CODTECH IT SOLUTIONS

NAME:AAKASH R

INTERN ID:CT04DR985

DOMAIN:DATA SCIENCE

DURATION: 4 WEEKS

MENTOR:NEELA SANTHOSH

# deep-learning-image-classification
🧠 Deep Learning Image Classification (MNIST)
A Convolutional Neural Network (CNN) built using TensorFlow to classify handwritten digits (0–9) from the MNIST dataset.
This project demonstrates how deep learning can interpret visual patterns with high accuracy.

📘 Project Overview

This project implements a Convolutional Neural Network (CNN) for image classification using the MNIST handwritten digit dataset.
The model is trained to recognize digits from 0 to 9 and achieves nearly 99% validation accuracy.

The dataset contains:

60,000 training images

10,000 testing images

Each image is 28x28 pixels, grayscale.

🧩 Features

✅ Built using TensorFlow and Keras
✅ Implements convolutional and pooling layers
✅ Achieves high accuracy and low loss
✅ Includes result visualizations and saved model file
✅ Runs smoothly on Google Colab (Windows compatible)

🧠 Model Architecture
Layer Type	Output Shape	Parameters
Conv2D (32 filters)	(26, 26, 32)	320
MaxPooling2D	(13, 13, 32)	0
Conv2D (64 filters)	(11, 11, 64)	18,496
MaxPooling2D	(5, 5, 64)	0
Flatten	(1600)	0
Dense (64)	(64)	102,464
Dense (10)	(10)	650

Total Parameters: 121,930
Trainable Parameters: 121,930

📊 Results
Metric	Accuracy
Training Accuracy	99.28%
Validation Accuracy	98.97%
📷 Sample Model Output

Here’s a sample of the model’s predictions on test data 👇

🚀 How to Run the Project

Open in Google Colab
Upload the file mnist_cnn_project.ipynb and open it in Colab.

Run all cells step by step
This will train and evaluate your CNN model.

Reload saved model (optional)

from tensorflow.keras.models import load_model
model = load_model("mnist_cnn_model.h5")


Generate predictions
The notebook includes code to visualize results and save the image as sample_predictions.png.

💾 Files in this Repository
File Name	Description
mnist_cnn_project.ipynb	Google Colab notebook with code
mnist_cnn_model.h5	Saved trained model
sample_predictions.png	Image showing predicted digits
README.md	Project documentation
🧠 Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

💡 Future Improvements

Add model evaluation on custom user-drawn digits

Use Dropout layers to reduce overfitting

Deploy model as a web app using Flask or Streamlit

<img width="668" height="917" alt="Image" src="https://github.com/user-attachments/assets/3247c7e8-2067-48c9-9ea2-f1ff933d6d94" />

