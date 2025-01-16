from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np


class Prediction: 
    def __init__(self, file:str):
        self.data = pd.read_csv(file)
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y')
        self.data['Returned'] = (self.data['Refunded item count'] > 0).astype(int)

        self.label_encoders = {}
        for column in ['Product Name']:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])
            self.label_encoders[column] = le

        features = ['Product Name', 'Product Price', 'Purchased Item Count', 'Refund on Return']
        self.X = self.data[features]
        self.y = self.data['Returned']

    def get_product_list(self):
        return self.label_encoders['Product Name'].inverse_transform(self.data['Product Name'].unique()).tolist()

    def generate_model_random_forest(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42, stratify=self.y)

        self.model_rf = RandomForestClassifier(random_state=42)
        self.model_rf.fit(X_train, y_train)

        y_pred = self.model_rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    
    def predict_product_return_probability(self, product):
        try:
            encoded_product_name = self.label_encoders['Product Name'].transform([product])[0]
        except ValueError:
            return "Error: Provided product name or category does not exist in the training data."
        
        avg_price = self.X['Product Price'].mean()
        avg_purchased_count = self.X['Purchased Item Count'].mean()
        avg_refund_on_return = self.X['Refund on Return'].mean()
        
        input_data = pd.DataFrame({
            'Product Name': [encoded_product_name],
            'Product Price': [avg_price],
            'Purchased Item Count': [avg_purchased_count],
            'Refund on Return': [avg_refund_on_return]
        })
        
        probabilities = self.model_rf.predict_proba(input_data)
        return probabilities[0][1]
    
    def generate_model_linear_regression(self):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, self.y, test_size=0.3, random_state=42)

        self.model_lr = LinearRegression()
        self.model_lr.fit(X_train, y_train)

        y_pred = self.model_lr.predict(X_test)
    
        y_pred = [max(0, min(1, pred)) for pred in y_pred]
        
        y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
        
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

    def predict_return_probability_linear(self, product):
        try:
            encoded_product_name = self.label_encoders['Product Name'].transform([product])[0]
        except ValueError:
            return "Error: Provided product name or category does not exist in the training data."
        
        avg_price = self.data['Product Price'].mean()
        avg_purchased_count = self.data['Purchased Item Count'].mean()
        avg_refund_on_return = self.data['Refund on Return'].mean()
        
        input_features = np.array([[encoded_product_name, avg_price, avg_purchased_count, avg_refund_on_return]])
        input_scaled = self.scaler.transform(input_features)
        
        return_prob = self.model_lr.predict(input_scaled)[0]
        return max(0, min(1, return_prob))
