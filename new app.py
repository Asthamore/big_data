#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model.keras")

# Cache the patient data load
@st.cache_data
def load_patient_data():
    return pd.read_csv("dummy_patient_data.csv")

# Home page
def home():
    st.title("🧠 Brain Tumor Detection App")
    st.markdown("Welcome! This app helps detect brain tumors from MRI scans and provides patient data insights.")

# Classification page
def classify():
    st.title("📷 Tumor Classification")

    uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).resize((128, 128))
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 128, 128, 3)

        prediction = model.predict(img_array)[0][0]
        label = "Tumor Detected" if prediction > 0.5 else "No Tumor Detected"
        st.subheader(f"🧾 Prediction: **{label}**")
        st.write(f"🧪 Confidence: `{prediction:.2f}`")

# Dashboard page
def dashboard():
    st.title("📊 Patient Dashboard")
    df = load_patient_data()

    # Convert Diagnosis to binary numeric (Yes=1, No=0)
    df['Diagnosis'] = df['Diagnosis'].map({'Yes': 1, 'No': 0})

    # Sidebar Filters
    st.sidebar.write("### Filters")
    gender_filter = st.sidebar.selectbox("Select Gender", options=['All', 'Male', 'Female'])
    region_filter = st.sidebar.selectbox("Select Region", options=['All'] + df['Region'].unique().tolist())
    age_filter = st.sidebar.slider("Select Age Range", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=(int(df['Age'].min()), int(df['Age'].max())))

    # Apply filters
    filtered_df = df.copy()
    if gender_filter != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]
    if region_filter != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == region_filter]
    filtered_df = filtered_df[(filtered_df['Age'] >= age_filter[0]) & (filtered_df['Age'] <= age_filter[1])]

    st.write("### 👀 First Look at the Data")
    st.dataframe(filtered_df.head())

    # --- Gender vs Tumor Presence ---
    st.write("### 🧑‍🤝‍🧑 Gender vs Tumor Presence (Bar Chart)")
    gender_tumor = filtered_df.groupby(['Gender', 'Diagnosis']).size().unstack().fillna(0)
    st.bar_chart(gender_tumor)

    # --- Age vs Tumor Rate ---
    st.write("### 📈 Age vs Tumor Rate (Line Chart)")
    age_tumor = filtered_df.groupby('Age')['Diagnosis'].mean()
    if not age_tumor.empty:
        st.line_chart(age_tumor)
    else:
        st.warning("No data available for selected filters to plot the line chart.")

    # --- Tumor Distribution by Region ---
    st.write("### 🥧 Tumor Distribution by Region (Pie Chart)")
    region_tumor = filtered_df[filtered_df['Diagnosis'] == 1]['Region'].value_counts()
    if not region_tumor.empty:
        fig1, ax1 = plt.subplots()
        ax1.pie(region_tumor, labels=region_tumor.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
    else:
        st.warning("No tumor cases found for selected filters to plot the pie chart.")

    # --- Heatmap ---
    st.write("### 🌡️ Heatmap: Region vs Age vs Tumor Count")
    heat_df = filtered_df[filtered_df['Diagnosis'] == 1].groupby(['Region', 'Age']).size().unstack().fillna(0)
    if not heat_df.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(heat_df, cmap='coolwarm', annot=True, fmt='.0f')
        st.pyplot(fig2)
    else:
        st.warning("No tumor data available for heatmap generation.")

# Navigation
pages = {
    "Home": home,
    "Classification": classify,
    "Dashboard": dashboard
}

st.sidebar.title("🔍 Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()


# In[ ]:




