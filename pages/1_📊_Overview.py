import streamlit as st
from pyspark.sql import SparkSession
import pandas as pd

# ---------------- TITLE ----------------
st.title("📊 Airline Passenger Overview Dashboard")

st.markdown("""
### ✈ Airline Passenger Satisfaction Analysis
This dashboard provides a summary of passenger satisfaction based on service quality, travel experience, and delay factors.
""")

# ---------------- LOAD DATA ----------------
spark = SparkSession.builder.appName("Overview").getOrCreate()

df = spark.read.csv("cleaned_airline_dataset", header=True, inferSchema=True)
df_pd = df.toPandas()

# Convert label safely
df_pd["label"] = df_pd["label"].astype(int)

# ---------------- KPIS ----------------
total = len(df_pd)
satisfied = int(df_pd["label"].sum())
not_satisfied = total - satisfied
satisfaction_rate = round((satisfied / total) * 100, 2)

avg_delay = round(df_pd["Arrival_Delay_in_Minutes"].mean(), 2)
avg_distance = round(df_pd["Flight_Distance"].mean(), 2)

# ---------------- KPI DISPLAY ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Passengers", total)
col2.metric("Satisfied", satisfied)
col3.metric("Not Satisfied", not_satisfied)
col4.metric("Satisfaction Rate", f"{satisfaction_rate}%")

st.markdown("---")

# ---------------- INSIGHT TEXT ----------------
st.subheader("📌 Key Insights")

st.write(f"""
- The dataset contains **{total} passengers**
- Average flight distance is **{avg_distance} km**
- Average arrival delay is **{avg_delay} minutes**
- Satisfaction rate is **{satisfaction_rate}%**

👉 This shows how service quality and delays influence passenger satisfaction.
""")

# ---------------- CHART SECTION ----------------
st.subheader("📊 Satisfaction Distribution")

st.bar_chart(df_pd["label"].value_counts())

st.markdown("---")

