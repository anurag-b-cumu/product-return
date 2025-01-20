# Product Return Prediction Application Documentation

### Overview
This document provides a detailed description of the **Product Return Prediction Application**, including its functionality, features, and usage.

### Hosted URL
The application is live on Streamlit Cloud:
[https://cu-return.streamlit.app/](https://cu-return.streamlit.app/)

### GitHub Repository
Source code for this project is available at:
[https://github.com/anurag-b-cumu/product-return](https://github.com/anurag-b-cumu/product-return)

### Features
1. **CSV File Upload**:
   - Upload a CSV file containing sales and return data.
   - Ensure the file meets the required structure.

2. **Model Training**:
   - Trains Random Forest and Linear Regression models.
   - Displays performance metrics (accuracy, precision, recall, F1 score).

3. **Return Probability Prediction**:
   - Predicts the likelihood of a product being returned.
   - Allows product selection from a dropdown menu.

4. **User-Friendly Interface**:
   - Streamlit-powered UI with separate tabs for each model.

### Application Components
#### Backend
The backend processes the data and implements machine learning models using `scikit-learn`. The main features include:
- Encoding categorical variables.
- Splitting data for training and testing.
- Generating metrics (accuracy, precision, recall, F1 score).
- Predicting probabilities using trained models.

#### Frontend
The frontend is developed with Streamlit and provides:
- File upload functionality.
- Tabs for model selection.
- Dropdown menu for product selection.
- Display of metrics and prediction results.

### Functions and Parameters
#### `__init__(self, file: str)`
- **Purpose**: Initializes the class, processes the CSV file, and prepares the data for modeling.
- **Parameters**:
  - `file`: Path to the uploaded CSV file.

#### `get_product_list(self)`
- **Purpose**: Retrieves the unique product names from the dataset.
- **Returns**: A list of product names.

#### `generate_model_random_forest(self)`
- **Purpose**: Trains a Random Forest classifier on the dataset.
- **Returns**: A dictionary containing the model’s accuracy, precision, recall, and F1 score.

#### `predict_product_return_probability(self, product: str)`
- **Purpose**: Predicts the return probability of a specified product using the Random Forest model.
- **Parameters**:
  - `product`: The name of the product.
- **Returns**: The probability of the product being returned.

#### `generate_model_linear_regression(self)`
- **Purpose**: Trains a Linear Regression model on the dataset.
- **Returns**: A dictionary containing the model’s accuracy, precision, recall, and F1 score.

#### `predict_return_probability_linear(self, product: str)`
- **Purpose**: Predicts the return probability of a specified product using the Linear Regression model.
- **Parameters**:
  - `product`: The name of the product.
- **Returns**: The probability of the product being returned.

### Why Random Forest and Linear Regression?
#### Random Forest
- **Reason**: Effective for classification problems with imbalanced datasets.
- **Advantages**:
  - Handles non-linear relationships well.
  - Reduces overfitting through ensemble learning.
- **Usage**: Provides a binary classification (returned or not).

#### Linear Regression
- **Reason**: Useful for predicting continuous probabilities.
- **Advantages**:
  - Simple and interpretable.
  - Efficient for smaller datasets.
- **Usage**: Provides a probability value between 0 and 1 for returns.

### Required Parameters for Output
1. **CSV File**:
   - Contains the following columns:
     - `Date`: Date of transaction (DD/MM/YYYY).
     - `Product Name`: Name of the product.
     - `Product Price`: Price of the product.
     - `Purchased Item Count`: Quantity purchased.
     - `Refunded item count`: Number of items refunded.
     - `Refund on Return`: Refund amount.

2. **Selected Product**:
   - The product name must match one in the uploaded dataset.

### Prerequisites
- Python 3.7+
- Required Python libraries:
  - pandas
  - numpy
  - scikit-learn
  - streamlit

### Installation
To run the app locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/anurag-b-cumu/product-return.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run main.py
   ```

### How to Use
1. Access the application at [https://cu-return.streamlit.app/](https://cu-return.streamlit.app/).
2. Upload a CSV file with sales data.
3. Navigate to the Random Forest or Linear Regression tab.
4. Train the selected model and view metrics.
5. Select a product to predict its return probability.

### Limitations
- Accuracy depends on the quality and completeness of the data.
- Predictions may vary between models due to differences in algorithms.
- Only supports the predefined CSV structure.

### Future Enhancements
- Incorporate additional machine learning models.
- Add data visualization for insights.
- Support for saving and loading trained models.

---

For feedback or issues, contact the developer at [developer@example.com](mailto:developer@example.com).
