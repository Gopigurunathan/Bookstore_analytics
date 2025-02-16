import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

# Load the trained ANN model
model = tf.keras.models.load_model(r'C:\Users\nkn05\OneDrive\Desktop\DL_proj\.venv\model_training\churn_prediction_ann_model.h5')




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
shipping_method_preference = st.sidebar.selectbox('Shipping Method Preference', ['International', 'Express', 'Priority', 'Standard'])


# Convert shipping method to one-hot encoded format
shipping_method_mapping = {'International': [1, 0, 0], 'Express': [0, 1, 0], 'Priority': [0, 0, 1], 'Standard': [0, 0, 0]}
shipping_method_encoded = shipping_method_mapping[shipping_method_preference]

# Calculate additional features
customer_lifetime_value = total_orders * average_order_value
order_frequency = days_since_last_order / total_orders if total_orders > 0 else 0

# Prepare input data
input_data = np.array([[total_orders, days_since_last_order, average_order_value, completed_orders, canceled_orders, address_status_changed] + shipping_method_encoded ])

# Predict button
if st.sidebar.button('Predict Churn'):
    # Make prediction
    prediction_prob = model.predict(input_data)
    prediction = (prediction_prob > 0.5).astype(int)
    
    # Display result
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error('**Prediction:** Churn')
    else:
        st.success('**Prediction:** No Churn')
    
    # Show prediction probability
    st.write(f'**Churn Probability:** {prediction_prob[0][0]:.2f}')