# Product Return Prediction Application Documentation


## Overview
This document provides a detailed description of the **Product Return Prediction Application**, including its functionality, features, and usage.


## Hosted URL
The application is live on Streamlit Cloud:
[https://cu-return.streamlit.app/](https://cu-return.streamlit.app/)


## GitHub Repository
Source code for this project is available at:
[https://github.com/anurag-b-cumu/product-return](https://github.com/anurag-b-cumu/product-return)


## Features
1. **Model Training**:
   - Trains Random Forest and Linear Regression models.
   - Displays performance metrics (accuracy).

2. **Return Probability Prediction**:
   - Predicts the likelihood of a product being returned.
   - Allows product selection from a dropdown menu.

3. **User-Friendly Interface**:
   - Streamlit-powered UI with separate tabs for each model.


## Application Components
### Backend
The backend processes the data and implements machine learning models using `scikit-learn`. The main features include:
- Encoding categorical variables.
- Splitting data for training and testing.
- Generating metrics (accuracy, precision, recall, F1 score).
- Predicting probabilities using trained models.

### Frontend
The frontend is developed with Streamlit and provides:
- Tabs for model selection.
- Dropdown menu for product selection.
- Display of metrics and prediction results.


## Functions and Parameters
### `__init__(self, file: str)`
- **Purpose**: Initializes the class, processes the CSV file, and prepares the data for modeling.
- **Parameters**:
  - `file`: Path to the uploaded CSV file.

### `get_product_list(self)`
- **Purpose**: Retrieves the unique product names from the dataset.
- **Returns**: A list of product names.

### `generate_model_random_forest(self)`
- **Purpose**: Trains a Random Forest classifier on the dataset.
- **Returns**: A dictionary containing the model’s accuracy, precision, recall, and F1 score.

### `predict_product_return_probability(self, product: str)`
- **Purpose**: Predicts the return probability of a specified product using the Random Forest model.
- **Parameters**:
  - `product`: The name of the product.
- **Returns**: The probability of the product being returned.

### `generate_model_linear_regression(self)`
- **Purpose**: Trains a Linear Regression model on the dataset.
- **Returns**: A dictionary containing the model’s accuracy, precision, recall, and F1 score.

### `predict_return_probability_linear(self, product: str)`
- **Purpose**: Predicts the return probability of a specified product using the Linear Regression model.
- **Parameters**:
  - `product`: The name of the product.
- **Returns**: The probability of the product being returned.


## Why Random Forest and Linear Regression?
### Random Forest
- **Reason**: Effective for classification problems with imbalanced datasets.
- **Advantages**:
  - Handles non-linear relationships well.
  - Reduces overfitting through ensemble learning.
- **Usage**:
  - **How it is used**: The Random Forest model is trained using features such as `Product Name`, `Product Price`, `Purchased Item Count`, and `Refund on Return`. During prediction, it calculates the return probability by evaluating the input product details encoded into numeric form and averaged contextual features like price, purchase count, and refund amount.
  - **Training Process**: The dataset is split into training and testing sets. The model learns patterns from the training set by creating multiple decision trees and combining their predictions to reduce overfitting and improve accuracy. The target variable, `Returned`, is binary (0 for not returned, 1 for returned).
  - **Prediction Parameters**:
    - Product Name (encoded)
    - Average Product Price
    - Average Purchased Item Count
    - Average Refund on Return
  - **Input Required**: A valid product name present in the dataset. The model uses the encoded product name along with averaged contextual feature values to predict the return probability.

### Linear Regression
- **Reason**: Useful for predicting continuous probabilities.
- **Advantages**:
  - Simple and interpretable.
  - Efficient for smaller datasets.
- **Usage**:
  - **How it is used**: The Linear Regression model is trained using the same features as the Random Forest model: `Product Name`, `Product Price`, `Purchased Item Count`, and `Refund on Return`. It outputs a continuous probability score between 0 and 1, representing the likelihood of a return.
  - **Training Process**: The dataset is scaled using `StandardScaler` to normalize feature values. The model minimizes the difference between predicted and actual values for the `Returned` column using regression techniques. Predictions are clamped between 0 and 1 to ensure valid probability values.
  - **Prediction Parameters**:
    - Product Name (encoded and scaled)
    - Scaled Product Price
    - Scaled Purchased Item Count
    - Scaled Refund on Return
  - **Input Required**: A valid product name from the dataset. The model transforms the input data into scaled features and predicts the return probability based on learned patterns.
 

## How to Use
1. Access the application at [https://cu-return.streamlit.app/](https://cu-return.streamlit.app/).
2. Navigate to the Random Forest or Linear Regression tab.
3. Train the selected model and view metrics.
4. Select a product to predict its return probability.


## Limitations
- Accuracy depends on the quality and completeness of the data.
- Predictions may vary between models due to differences in algorithms.
- Only supports the predefined CSV structure.
