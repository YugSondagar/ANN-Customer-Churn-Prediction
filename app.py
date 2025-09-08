import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle


# Load the trained model & encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_geography.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit UI
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction App")
st.markdown("Use this app to **predict the probability of customer churn** based on their details.")

# Sidebar Information
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This app uses a **Deep Learning model** to predict churn probability.  
- Trained on customer banking dataset.  
- Uses **LabelEncoder** for Gender and **OneHotEncoder** for Geography.  
- Scales features using **StandardScaler**.  
""")


# Input Form
st.header("📝 Customer Details")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    tenure = st.slider('📅 Tenure (Years)', 0, 10, 3)
    num_of_products = st.slider('🛒 Number of Products', 1, 4, 1)

with col2:
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900, value=650)
    balance = st.number_input('💰 Balance', min_value=0.0, step=100.0)
    estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, step=100.0)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
    is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

# Prepare Input Data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Merge encoded features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale features
input_data_scaled = scaler.transform(input_data)


prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]


st.header("🔮 Prediction Result")

st.progress(float(prediction_proba))

if prediction_proba > 0.5:
    st.error(f"⚠️ Churn Probability: **{prediction_proba:.2f}**\n\nThe customer is **likely to churn**.")
else:
    st.success(f"✅ Churn Probability: **{prediction_proba:.2f}**\n\nThe customer is **not likely to churn**.")


with st.expander("🔍 View Processed Input Data"):
    st.dataframe(input_data)
