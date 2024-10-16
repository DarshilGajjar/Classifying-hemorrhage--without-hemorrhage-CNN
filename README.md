# Hemorrhage Detection using CNN

## Introduction
In this project, I used a convolutional neural network (CNN) architecture to classify patients suffering from hemorrhage by analyzing head CT images. The dataset consists of two classes: "Normal" for healthy patients and "Hemorrhage" for unhealthy ones.

## Project Structure
|– data/                     # Directory to store datasets
|– models/                   # Directory for storing trained models
|– src/                      # Python scripts and modules
|   |– data_preprocessing.py # Data preprocessing script
|   |– cnn_model.py          # CNN model architecture script
|– README.md                 # Project documentation
|– requirements.txt          # List of dependencies

## Libraries Required
* Python 3.6+
* TensorFlow
* Keras
* NumPy
* Pandas
* Seaborn
* MatplotLib
* Scikit-learn

## Data Processing
We utilized `ImageDataGenerator` from Keras for data augmentation, including rescaling, random zoom, shear transformations, and rotations to enhance the model's robustness. Data was split into training, validation, and testing sets.

## CNN Model Architecture
The CNN model consists of multiple convolutional layers, followed by max-pooling and dropout for regularization. The model ends with a softmax layer for binary classification.

## Results
The model achieved an accuracy of 96% with a training loss of 0.1184.

![Head CT Classification](https://i.imgur.com/O9XSyiP.jpeg)

## Discussion
This model successfully detects hemorrhage in head CT images with high accuracy.


