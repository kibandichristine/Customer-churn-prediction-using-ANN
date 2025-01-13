# Customer-churn-prediction-using- Artificial Neural Networks (ANNs)


## Overview

This project applies Artificial Neural Networks (ANNs) to predict customer churn, a critical task for businesses aiming to improve customer retention. The primary objective is to classify customers into two categories: those likely to churn and those who are not.

## Objective

The primary objective is to build and evaluate a neural network that can be applied to binary classification problems, where the output class is either 0 or 1. The model will be trained using training data and validated using validation data to monitor performance.

## Data Details

The dataset used includes:

Features: 26 input variables representing customer attributes and behaviors.

Target Variable: Churn (binary: Yes/No).

## Key Features

Demographic Information: Gender, Senior Citizen, Dependents, etc.

Service Details: Internet Service, Streaming, Tech Support, etc.

Billing Information: Monthly Charges, Total Charges, Payment Method.

## Data Preprocessing

1) Removed irrelevant columns (e.g., customerID).
2) Handled missing and inconsistent data (e.g., blank TotalCharges rows dropped).
3) Encoded categorical variables using one-hot encoding and binary conversion for "Yes"/"No" responses.
   
## Modeling Approach

### Goals

1.Design a Neural Network:

  . Input Layer: Includes preprocessed features. 
 
  . Hidden Layers: Configured for non-linear feature learning.
 
  . Output Layer: Binary classification (Churn: Yes/No).
        
2. Train the Model:
   
. Loss Function: Binary Cross-Entropy.

. Metric: Accuracy.

3. Evaluate Performance:
   
. Metrics: Accuracy and Loss during training and validation.

## Exploratory Data Analysis (EDA)
1. Tenure vs. Churn: Customers with longer tenures are less likely to churn.
    .Visualized using histograms of churned and non-churned customers.
2. Monthly Charges vs. Churn: High monthly charges are associated with increased churn likelihood.

## Preprocessing Steps

1.Data Cleaning:

  . Dropped rows with blank TotalCharges.
  . Converted TotalCharges to numeric data type.
    
2.Feature Transformation:

  . Replaced categorical entries like "No Internet Service" with "No".
  
  . Converted "Yes"/"No" responses to binary (1/0).
  
  . One-hot encoded multi-category features (InternetService, Contract, PaymentMethod).

## Results
### Data Insights
. Customers with shorter tenures and higher charges show higher churn rates.
. Certain services, like Fiber Optic Internet, have distinct churn patterns.

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
      
5. **Deployment**:
   - deployed using streamlit

## Requirements

- Python 3.x
- TensorFlow (for Keras)
- NumPy
- Matplotlib
- streamlit


