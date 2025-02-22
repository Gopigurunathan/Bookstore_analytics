import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the trained ANN model
#load the trained model
model=load_model(r'/home/ubuntu/Bookstore_analytics/churn_prediction_ann_model.h5')

#load the encoder and scaler


with open(r'/home/ubuntu/Bookstore_analytics/one_hot_shipping.pkl','rb') as file:
    label_encoder_shipping=pickle.load(file)
    
with open(r'/home/ubuntu/Bookstore_analytics\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)




# Title of the app
st.title('Customer Churn Prediction Dashboard')

# Sidebar for user input
st.sidebar.header('Customer Details')

total_orders = st.sidebar.number_input('Total Orders', min_value=0)
days_since_last_order = st.sidebar.number_input('Days Since Last Order', min_value=0)
average_order_value = st.sidebar.number_input('Average Order Value', min_value=0.0)
completed_orders = st.sidebar.number_input('Completed Orders', min_value=0)
canceled_orders = st.sidebar.number_input('Canceled Orders', min_value=0)
address_status_changed = st.sidebar.selectbox('Address Status Changed', [0, 1])
shipping_method_preference = st.sidebar.selectbox('Shipping Method Preference', label_encoder_shipping.categories_[0])



# Calculate additional features
customer_lifetime_value = total_orders * average_order_value
order_frequency = days_since_last_order / total_orders if total_orders > 0 else 0


# Prepare the input data
input_data = pd.DataFrame({
    'total_orders': [total_orders],
    'days_since_last_order': [days_since_last_order],
    'average_order_value': [average_order_value],
    'completed_orders': [completed_orders],
    'canceled_orders': [canceled_orders],
    'address_status_changed': [address_status_changed],
    'shipping_method_preference':'Express'
    
    
})

# Convert shipping method to one-hot encoded format
# One-hot encode 'shipping'
shipping_encoded = label_encoder_shipping.transform(input_data[['shipping_method_preference']]).toarray()
shipping_encoded_df = pd.DataFrame(shipping_encoded, columns=label_encoder_shipping.get_feature_names_out(['shipping_method_preference']))

# Concatenate one-hot encoded shipping method
input_data = pd.concat([input_data.drop('shipping_method_preference', axis=1), shipping_encoded_df], axis=1)




# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Initialize prediction_prob with None
prediction_prob = None

# Predict button
if st.sidebar.button('Predict Churn'):
    # Make prediction
    prediction = model.predict(input_data_scaled)  # Make sure you use `input_data_scaled`
    prediction_prob = prediction[0][0]

# Display results only if prediction_prob is set
if prediction_prob is not None:
    st.write(f'Churn Probability: {prediction_prob:.2f}')

    st.subheader('Prediction Result')
    if prediction_prob > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')



   
