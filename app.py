import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load and preprocess dataset
df = pd.read_csv('salesData.csv')

# Handle missing values
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
mode_outlet_size = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode()[0])
missing = df['Outlet_Size'].isnull()
df.loc[missing, 'Outlet_Size'] = df.loc[missing, 'Outlet_Type'].apply(lambda x: mode_outlet_size[x][0])

# Clean and encode
df.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)
encoder = LabelEncoder()
for col in ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']:
    df[col] = encoder.fit_transform(df[col])

df.drop(['Item_Identifier'], axis=1, inplace=True)

# Log transform
df['Item_Visibility'] = np.log1p(df['Item_Visibility'])
df['Item_Outlet_Sales'] = np.log1p(df['Item_Outlet_Sales'])

# Split features and target
X = df.drop('Item_Outlet_Sales', axis=1)
y = df['Item_Outlet_Sales']

# Train model
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
model = XGBRegressor()
model.fit(x_train, y_train)

# Streamlit UI
st.set_page_config(page_title="🛒 Sales Prediction App", layout="centered")
st.title("📈 BigMart Sales Prediction App")
st.markdown("Predict sales for a product at a particular outlet using XGBoost Regression.")

# Input fields
item_weight = st.number_input("Item Weight", min_value=0.0, value=9.3)
item_visibility = st.number_input("Item Visibility", min_value=0.0, value=1217.0)
item_mrp = st.number_input("Item MRP", min_value=0.0, value=249.8)
item_fat_content = st.number_input("Item Fat Content (Encoded)", min_value=0, max_value=10, value=1)
item_type = st.number_input("Item Type (Encoded)", min_value=0, max_value=20, value=10)
outlet_identifier = st.number_input("Outlet Identifier (Encoded)", min_value=0, max_value=10, value=3)
est_year = st.number_input("Outlet Establishment Year", min_value=1985, max_value=2025, value=2000)
outlet_size = st.number_input("Outlet Size (Encoded)", min_value=0, max_value=5, value=1)
outlet_location = st.number_input("Outlet Location Type (Encoded)", min_value=0, max_value=5, value=1)
outlet_type = st.number_input("Outlet Type (Encoded)", min_value=0, max_value=5, value=1)

# Predict button
if st.button("📊 Predict Sales"):
    input_data = np.array([item_weight, item_visibility, item_mrp,
                           item_fat_content, item_type, outlet_identifier,
                           est_year, outlet_size, outlet_location, outlet_type])

    input_data[1] = np.log1p(input_data[1])  # Log transform visibility
    input_reshaped = input_data.reshape(1, -1)
    prediction = model.predict(input_reshaped)
    output = np.expm1(prediction)  # Inverse log1p

    st.success(f"🛍️ Predicted Sales: ₹ {round(output[0], 2)}")

# To run the app, use the command: streamlit run app.py