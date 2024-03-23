import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Function to preprocess input data
def preprocess_input(data):
    # Preprocess input data
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('reg', 'Regular')

    for i in range(len(data)):
        if pd.isna(data.loc[i,'Outlet_Size']):
            if (data.loc[i,'Outlet_Type']=='Grocery Store') or (data.loc[i,'Outlet_Type']=='Supermarket Type1') :
                data.loc[i, 'Outlet_Size'] = 'Small'
            elif (data.loc[i,'Outlet_Type']=='Supermarket Type2') or (data.loc[i,'Outlet_Type']=='Supermarket Type3') :
                data.loc[i, 'Outlet_Size'] = 'Medium'

    data['Item_Type_Category'] = data['Item_Identifier'].apply(lambda x: x[0:2])
    data['Item_Type_Category'] = data['Item_Type_Category'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
    data.loc[data['Item_Type_Category']=='Non-Consumable','Item_Fat_Content'] = "Non-Edible"

    Item_Type_Mean = data.pivot_table(columns='Item_Type', values='Item_Weight', aggfunc=lambda x:x.mean())
    for i in range(len(data)):
        if pd.isna(data.loc[i, 'Item_Weight']):
            item = data.loc[i, 'Item_Type']
            data.at[i, 'Item_Weight'] = Item_Type_Mean[item]

    Item_Visibility_Mean = data[['Item_Type_Category', 'Item_Visibility']].groupby(['Item_Type_Category'], as_index=False).mean()
    for i in range(len(data)):
        if data.loc[i, 'Item_Visibility']==0:
            cat =  data.loc[i, 'Item_Type_Category']
            m = Item_Visibility_Mean.loc[Item_Visibility_Mean['Item_Type_Category'] == cat]['Item_Visibility']
            data.at[i, 'Item_Visibility'] = m

    if 'Outlet_Establishment_Year' in data.columns:
        data['Operation_Years'] = 2013-data['Outlet_Establishment_Year']
        data = data.drop(['Outlet_Establishment_Year'], axis=1)
    else:
        st.warning("Warning: 'Outlet_Establishment_Year' column not found in input data.")

    lb = LabelEncoder()
    data['Outlet'] = lb.fit_transform(data['Outlet_Identifier'])
    var = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size', 'Item_Type_Category']
    for item in var:
        data[item] = lb.fit_transform(data[item])

    data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size', 'Item_Type_Category'])

    return data

# Function to predict using the models
def predict_sales(input_data):
    # Load models and scaler
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_RNN = joblib.load(os.path.join(current_dir, 'model_RNN.pkl'))
    model_LinReg = joblib.load(os.path.join(current_dir, 'model_LinReg.pkl'))
    model_RidgeReg = joblib.load(os.path.join(current_dir, 'model_RidgeReg.pkl'))
    scaler = joblib.load(os.path.join(current_dir, 'min_max_scaler.pkl'))

    # Preprocess input data
    input_data_processed = preprocess_input(input_data)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data_processed)

    # Reshape for RNN model
    x_input = np.reshape(input_data_scaled, (input_data_scaled.shape[0], input_data_scaled.shape[1], 1))

    # Predict using models
    predictions_RNN = model_RNN.predict(x_input)
    predictions_LinReg = model_LinReg.predict(input_data_scaled)
    predictions_RidgeReg = model_RidgeReg.predict(input_data_scaled)

    return predictions_RNN, predictions_LinReg, predictions_RidgeReg

# Main function for Streamlit app
def main():
    st.title("BigMart Sales Prediction Web App")

    st.sidebar.header("Input Features")

    # User input for data
    user_input = {}

    user_input['Item_Identifier'] = st.sidebar.text_input("Item Identifier")
    user_input['Item_Weight'] = st.sidebar.number_input
