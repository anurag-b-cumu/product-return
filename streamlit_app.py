import streamlit as st
from backend import Prediction


prediction = None

st.title("Product Return Prediction")



st.header("Upload the CSV file")
# uploaded_file = st.file_uploader("Choose a file", type=["csv"])
uploaded_file = "dataset.csv"


if uploaded_file:
    prediction = Prediction(uploaded_file)
    product_list = prediction.get_product_list()
    input_product = None
    
    # random_forest, linear_regression = st.tabs(["Random Forest", "Linear Regression"])
    # with random_forest:
    #     model_accuracy = prediction.generate_model_random_forest()
    #     st.success(f"Model is ready to Use with {model_accuracy['Accuracy']*100:.2f}% accuracy")

    #     st.subheader("Select the product to predict the return probability")
    #     input_product = st.selectbox("Select the product", product_list if len(product_list) > 0 else ["Document Do Not Contain Any Product Name"], key="random_forest")

    #     if input_product != "Document Do Not Contain Any Product Name":
    #         prediction_result = prediction.predict_product_return_probability(input_product)
    #         st.header("Prediction Result")
    #         st.subheader(f'Product Return Probability: {prediction_result*100:.2f}%')

    # with linear_regression:
    model_accuracy = prediction.generate_model_linear_regression()
    st.success(f"Model is ready to Use with {model_accuracy['Accuracy']*100:.2f}% accuracy")

    st.subheader("Select the product to predict the return probability")
    input_product = st.selectbox("Select the product", product_list if len(product_list) > 0 else ["Document Do Not Contain Any Product Name"], key="linear_regression")

    if input_product != "Document Do Not Contain Any Product Name":
        prediction_result = prediction.predict_return_probability_linear(input_product)
        st.header("Prediction Result")
        st.subheader(f'Product Return Probability: {prediction_result*100:.2f}%')