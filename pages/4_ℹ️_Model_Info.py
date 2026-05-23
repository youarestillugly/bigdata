import streamlit as st
from pyspark.sql import SparkSession

# ---------------- PAGE ----------------
st.set_page_config(page_title="Model Info", page_icon="ℹ")

st.title("ℹ Model Information Dashboard")

st.markdown("### ✈ Airline Passenger Satisfaction ML Model")

# ---------------- BASIC INFO ----------------
st.info("""
This project predicts whether an airline passenger is satisfied or not
based on service quality and travel experience.
""")

col1, col2, col3 = st.columns(3)

col1.metric("Algorithm", "GBT (Gradient Boosted Trees)")
col2.metric("Task", "Binary Classification")
col3.metric("Framework", "PySpark MLlib")

st.success("✅ Model successfully trained on airline passenger dataset")

# ---------------- HOW IT WORKS ----------------
st.markdown("## 🧠 How the Model Works")

st.markdown("""
1. Input passenger service experience  
2. Data is converted into feature vector  
3. GBT model processes decision trees  
4. Output: Satisfied (1) or Not Satisfied (0)  
""")

# ---------------- FEATURES ----------------
st.markdown("## 📊 Features Used in Training")

features = [
    "Age",
    "Flight Distance",
    "Inflight Wifi Service",
    "Departure/Arrival Time Convenience",
    "Ease of Online Booking",
    "Gate Location",
    "Food and Drink",
    "Online Boarding",
    "Seat Comfort",
    "Inflight Entertainment",
    "On-board Service",
    "Leg Room Service",
    "Baggage Handling",
    "Check-in Service",
    "Inflight Service",
    "Cleanliness",
    "Departure Delay (Minutes)",
    "Arrival Delay (Minutes)"
]

st.write(features)

# ---------------- MODEL INSIGHT ----------------
st.markdown("## 📌 Model Insight")

st.info("""
Gradient Boosted Trees combines multiple weak decision trees to form a strong learner.
It improves accuracy by reducing errors step by step.
""")

# ---------------- WHY THIS MODEL ----------------
st.markdown("## ⭐ Why GBT was used?")

st.markdown("""
- Handles non-linear relationships well  
- Works great with tabular data  
- High accuracy for classification problems  
- Robust to feature interactions  
""")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🚀 Built using PySpark + Streamlit | Airline ML Project")