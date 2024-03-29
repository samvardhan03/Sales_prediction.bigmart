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

    data['Operation_Years'] = 2013-data['Outlet_Establishment_Year']
    data = data.drop(['Item_Type', 'Outlet_Establishment_Year', 'Item_Identifier'], axis=1)

    lb = LabelEncoder()
    data['Outlet'] = lb.fit_transform(data['Outlet_Identifier'])
    var = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size', 'Item_Type_Category']
    for item in var:
        data[item] = lb.fit_transform(data[item])

    data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size', 'Item_Type_Category'])

    return data

# Function to predict using the models
def predict_sales(input_data, model_RNN, model_LinReg, model_RidgeReg, scaler):
    # Preprocess input data
    input_data_processed = preprocess_input(input_data)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data_processed)

    # Predict using models
    predictions_RNN = model_RNN.predict(input_data_scaled)
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
    user_input['Item_Weight'] = st.sidebar.number_input("Item Weight", min_value=0.0, max_value=500.0)
    user_input['Item_Visibility'] = st.sidebar.number_input("Item Visibility", min_value=0.0, max_value=1.0)
    user_input['Item_MRP'] = st.sidebar.number_input("Item MRP", min_value=0.0)
    user_input['Outlet_Identifier'] = st.sidebar.selectbox("Outlet Identifier", ["OUT010", "OUT013", "OUT017", "OUT018", "OUT019", "OUT027", "OUT035", "OUT045", "OUT046", "OUT049"])
    user_input['Outlet_Size'] = st.sidebar.selectbox("Outlet Size", ["Small", "Medium", "High"])
    user_input['Outlet_Location_Type'] = st.sidebar.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
    user_input['Outlet_Type'] = st.sidebar.selectbox("Outlet Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])
    user_input['Item_Fat_Content'] = st.sidebar.selectbox("Item Fat Content", ["Low Fat", "Regular", "Non-Edible"])
    user_input['Item_Type'] = st.sidebar.selectbox("Item Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned", "Breads", "Starchy Foods", "Others", "Seafood"])

    if st.sidebar.button("Predict"):
        input_data = pd.DataFrame(user_input, index=[0])
        input_data_processed = preprocess_input(input_data)

        # Load models and scaler
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_RNN = joblib.load(os.path.join(current_dir, 'model_RNN.pkl'))
        model_LinReg = joblib.load(os.path.join(current_dir, 'model_LinReg.pkl'))
        model_RidgeReg = joblib.load(os.path.join(current_dir, 'model_RidgeReg.pkl'))
        scaler = joblib.load(os.path.join(current_dir, 'min_max_scaler.pkl'))

        input_data_scaled = scaler.transform(input_data_processed)

        # Predict using models
        predictions_RNN, predictions_LinReg, predictions_RidgeReg = predict_sales(input_data_scaled, model_RNN, model_LinReg, model_RidgeReg, scaler)

        st.write("Predictions:")
        st.write("RNN:", predictions_RNN[0])
        st.write("Linear Regression:", predictions_LinReg[0])
        st.write("Ridge Regression:", predictions_RidgeReg[0])

if __name__ == "__main__":
    main()
