# Customer-churn-prediction-using-ANN
# Neural Network for Binary Classification

## Overview

This project implements a simple neural network using TensorFlow/Keras to perform binary classification. The model is trained on a dataset with 26 features, and the goal is to predict one of two classes (binary classification).

## Objective

The primary objective is to build and evaluate a neural network that can be applied to binary classification problems, where the output class is either 0 or 1. The model will be trained using training data and validated using validation data to monitor performance.

## Project Structure

- **Model**: The model consists of one input layer, one hidden layer with ReLU activation, and one output layer with a sigmoid activation.
- **Training**: The model is trained using the Adam optimizer and binary cross-entropy loss.
- **Evaluation**: The model's performance is evaluated using accuracy and loss metrics, and the results are plotted for both training and validation data.

## Steps Taken

1. **Model Architecture**: 
    - Input layer: 26 input features.
    - Hidden layer: 20 neurons with ReLU activation.
    - Output layer: 1 neuron with sigmoid activation.
    
2. **Compilation**:
    - Optimizer: Adam
    - Loss function: Binary Cross-Entropy
    - Evaluation metrics: Accuracy

3. **Training**: 
    - 100 epochs of training using a training dataset.
    - Validation data is used to monitor model performance during training.

4. **Visualization**: 
    - Training and validation accuracy over epochs.
    - Training and validation loss over epochs.

## Requirements

- Python 3.x
- TensorFlow (for Keras)
- NumPy
- Matplotlib

## Installation

To run this project, ensure you have the necessary Python packages installed. You can install them using pip:

```bash
pip install tensorflow numpy matplotlib
