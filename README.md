# Neural Networks for the Publishing Industry: Enhancing Customer Experience and Sales

## Overview

This project applies Artificial Neural Networks (ANNs) to the publishing industry to analyze customer behavior, recommend books, and forecast demand. The aim is to improve customer experience and optimize sales strategies through data-driven insights.

## Skills Acquired

Data Cleaning and Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering for Neural Networks

Building and Training Artificial Neural Networks (ANNs)

Model Evaluation and Optimization

MySQL Usage and Integration

Deploying Deep Learning Models using AWS and Streamlit

Documentation and Reporting

## Domain

Publishing Industry (E-commerce/Bookstore Analytics)

## Problem Statement

Leverage neural networks to analyze and predict customer behavior, recommend books, and forecast demand in the publishing industry, improving both customer experience and sales.

## Business Use Cases

Customer Churn Prediction: Identify customers likely to stop purchasing and implement retention strategies.

Personalized Book Recommendations: Provide targeted recommendations to enhance customer satisfaction.

Demand Forecasting: Predict future book sales for inventory optimization.

## Approach

1. Data Understanding & Preprocessing

Import SQL data into PostgreSQL.

Normalize tables and handle missing data.

Convert raw data into a machine learning-friendly format.

2. Exploratory Data Analysis (EDA)

Generate statistical summaries and identify trends.

Visualize relationships between customers, orders, and books.

3. Feature Engineering

Transform and encode categorical features (e.g., genres, locations).

Aggregate customer behavior metrics (e.g., purchase frequency).

Create temporal features for forecasting.

4. Model Development

Built ANN models for:

Churn Prediction: Predict whether a customer will churn.

Genre Prediction: Classify books into genres using metadata.

Demand Forecasting: Use time-series data to predict future book sales.

5. Model Evaluation

Metrics used:

Churn Prediction: Accuracy, Precision, Recall, F1-Score.

Genre Prediction: Accuracy, Confusion Matrix, Precision, Recall, F1-Score.

Demand Forecasting: MAE, RMSE.

6. Deployment

Deploy trained models on AWS EC2.

Create an interactive frontend using Streamlit to:

Display recommendations.

Provide churn predictions.

Visualize demand forecasts.

## Results

A functional system for predictions, recommendations, and demand forecasting.

Insights into customer behavior and sales patterns.

Improved operational efficiency in inventory management.

## Technical Stack

Database: PostgreSQL, MySQL

Programming: Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras

Visualization: Matplotlib, Seaborn

Deployment: AWS EC2, Streamlit

Version Control: Git, GitHub

## Dataset

### Tables and Descriptions

Author Table: Metadata about authors.

Publisher Table: Metadata about publishers.

Book Table: Metadata about books.

Customer Table: Information about customers.

Orders Table: Details of customer orders.

Order History Table: Status updates on orders.

Shipping Table: Information on shipping methods.

### Project Deliverables

Cleaned and preprocessed dataset

EDA report with visualizations

Feature engineering code and descriptions

Predictive models with code and explanations

Model evaluation report

Insights and recommendations report

AWS Deployment with nohup

Source code and documentation






