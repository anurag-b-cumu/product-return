# README

## Product Return Prediction Application

### Overview
The **Product Return Prediction Application** is a web-based tool designed to predict the likelihood of product returns based on historical sales data. The app utilizes machine learning models (Random Forest and Linear Regression) to generate predictions and metrics, offering insights into product return trends.

### Hosted URL
The application is hosted on **Streamlit Cloud** and can be accessed at:
[https://cu-return.streamlit.app/](https://cu-return.streamlit.app/)

### GitHub Repository
Source code for this project is available at:
[https://github.com/anurag-b-cumu/product-return](https://github.com/anurag-b-cumu/product-return)

### Features
- Upload and process CSV files with sales data.
- Train and evaluate machine learning models (Random Forest and Linear Regression).
- Predict return probabilities for individual products.
- Interactive UI for streamlined user experience.

### Prerequisites
Ensure you have the following libraries installed locally if running the application:
- **Python 3.7+**
- pandas
- numpy
- scikit-learn
- streamlit

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/anurag-b-cumu/product-return.git
   ```
2. Navigate to the project directory:
   ```bash
   cd product-return
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application locally:
   ```bash
   streamlit run main.py
   ```

### Usage
1. Open the app in your browser.
2. Upload a CSV file with the required structure.
3. Train models and view their performance metrics.
4. Select a product to predict its return probability.

### CSV Requirements
The uploaded file must include the following columns:
- `Date`: Date of transaction (DD/MM/YYYY).
- `Product Name`: Name of the product.
- `Product Price`: Price of the product.
- `Purchased Item Count`: Quantity purchased.
- `Refunded item count`: Number of items refunded.
- `Refund on Return`: Refund amount.

### Example Output
- **Random Forest**
  - Accuracy: 85.34%
  - Product Return Probability: 65.23%

- **Linear Regression**
  - Accuracy: 70.56%
  - Product Return Probability: 62.34%

### Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

### License
This project is licensed under the MIT License.
