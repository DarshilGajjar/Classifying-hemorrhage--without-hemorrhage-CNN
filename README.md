# Hemorrhage Detection using Convolutional Neural Networks (CNNs)

## Introduction
In this project, I used a convolutional neural network (CNN) architecture to classify patients suffering from hemorrhage by analyzing head CT images. The dataset consists of two classes: "Normal" for healthy patients and "Hemorrhage" for unhealthy ones.

## Installation
To set up the project, ensure you have the following libraries installed in your Python environment:

```bash
pip install glob
pip install keras
pip install numpy
pip install pandas
pip install seaborn
pip install matplotlib
pip install tensorflow
pip install scikit-learn
```
## Project Structure
The Project is organized into the following directories

```plaintext
project_root/
├── data/             # Contains datasets
├── models/           # Stores trained models
├── notebooks/        # Jupyter notebooks for experimentation and visualization
└── src/              # Python scripts for data preprocessing and CNN model
```

## Usage
Here's how you can use this project

### Preprocessing Data
To prepare the dataset for the CNN model, images are preprocessed using the Keras ImageDataGenerator for data augmentation, which includes operations such as rescaling, rotation, and flipping. The data is split into training, validation, and testing sets to evaluate model performance effectively.

To preprocess the head CT images, run the following command in your terminal:
``` bash
python src/data_preprocessing.py
```

### Training the CNN Model
The CNN architecture consists of multiple convolutional layers, dropout layers for regularization, and max-pooling layers to reduce spatial dimensions. The final output layer uses a softmax activation function for binary classification.

Train the CNN model using the command below:
``` bash
python src/cnn_model.py
```

### Evaluating the Model
After training, we evaluate the performance of the CNN model on the test dataset and make predictions.
``` bash
python src/evaluate_cnn_model.py
```

## Results
After training and evaluating the CNN model, it achieved an accuracy of *96%* in predicting/classifying hemorrhage using head CT images, with a training loss of *0.1184*.

![Head CT Classification](https://i.imgur.com/O9XSyiP.jpeg)

## Discussion
The CNN model successfully classifies and detects hemorrhage from head CT images with high accuracy, demonstrating the potential of deep learning in medical image analysis. Future work may include expanding the dataset and enhancing model complexity for improved performance.