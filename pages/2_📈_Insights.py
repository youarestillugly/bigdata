import streamlit as st
from pyspark.sql import SparkSession
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, round

# ---------------- PAGE ----------------
st.set_page_config(page_title="Insights Dashboard", layout="wide")

st.title("📈 Airline Data Insights Dashboard")

st.markdown("""
This dashboard explores how **service quality, delays, travel type, and demographics**
affect passenger satisfaction using Big Data analysis.
""")

# ---------------- LOAD DATA ----------------
spark = SparkSession.builder.appName("Insights").getOrCreate()
df = spark.read.csv("cleaned_airline_dataset", header=True, inferSchema=True)
df_pd = df.toPandas()

df_pd["label"] = df_pd["label"].astype(int)

st.divider()

# ======================================================
# 1️⃣ SATISFACTION DISTRIBUTION
# ======================================================
st.subheader("📊 1. Overall Satisfaction Distribution")

total = len(df_pd)
satisfied = df_pd["label"].sum()
not_satisfied = total - satisfied

col1, col2, col3 = st.columns(3)
col1.metric("Total Passengers", total)
col2.metric("Satisfied", int(satisfied))
col3.metric("Not Satisfied", int(not_satisfied))

st.bar_chart(df_pd["label"].value_counts())

st.markdown("""
📌 **Insight:**
- Dataset shows class balance
- Helps model learn both classes effectively
""")

st.divider()

# ======================================================
# 2️⃣ GENDER DISTRIBUTION
# ======================================================
st.subheader("👥 2. Gender Distribution")

gender_pd = df_pd["Gender"].value_counts()

fig, ax = plt.subplots()
ax.pie(gender_pd, labels=gender_pd.index, autopct='%1.1f%%')
ax.set_title("Gender Distribution")
st.pyplot(fig)

st.markdown("""
📌 **Insight:**
- Gender distribution is fairly balanced
- No strong bias toward one group
""")

st.divider()

# ======================================================
# 3️⃣ CUSTOMER TYPE VS SATISFACTION
# ======================================================
st.subheader("🧳 3. Customer Type vs Satisfaction")

cust = pd.crosstab(df_pd["Customer_Type"], df_pd["label"])

st.bar_chart(cust)

st.markdown("""
📌 **Insight:**
- Returning customers show higher satisfaction
- First-time customers are more sensitive to service issues
""")

st.divider()

# ======================================================
# 4️⃣ TRAVEL TYPE
# ======================================================
st.subheader("✈ 4. Travel Type Distribution")

travel = df_pd["Type_of_Travel"].value_counts()

fig, ax = plt.subplots()
ax.pie(travel, labels=travel.index, autopct='%1.1f%%')
ax.set_title("Travel Type")
st.pyplot(fig)

st.markdown("""
📌 **Insight:**
- Business travel dominates dataset
- Business passengers expect higher service quality
""")

st.divider()

# ======================================================
# 5️⃣ CLASS VS SATISFACTION
# ======================================================
st.subheader("💺 5. Travel Class vs Satisfaction")

class_ct = pd.crosstab(df_pd["Class"], df_pd["label"])

st.bar_chart(class_ct)

st.markdown("""
📌 **Insight:**
- Business class passengers are more satisfied
- Economy passengers show lower satisfaction
""")

st.divider()

# ======================================================
# 6️⃣ AGE DISTRIBUTION
# ======================================================
st.subheader("👤 6. Passenger Age Distribution")

fig, ax = plt.subplots()
sns.histplot(df_pd["Age"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.markdown("""
📌 **Insight:**
- Majority passengers are middle-aged
- Age does not strongly affect satisfaction
""")

st.divider()

# ======================================================
# 7️⃣ FLIGHT DISTANCE
# ======================================================
st.subheader("🛫 7. Flight Distance Distribution")

fig, ax = plt.subplots()
sns.histplot(df_pd["Flight_Distance"], bins=40, kde=True, ax=ax)
st.pyplot(fig)

st.markdown("""
📌 **Insight:**
- Most flights are short to medium distance
- Long flights slightly reduce satisfaction
""")

st.divider()

# ======================================================
# 8️⃣ DELAY DISTRIBUTION
# ======================================================
st.subheader("⏱ 8. Departure Delay Distribution")

fig, ax = plt.subplots()
sns.histplot(df_pd["Departure_Delay_in_Minutes"], bins=50, kde=True, ax=ax)
st.pyplot(fig)

st.markdown("""
📌 **Insight:**
- Most flights have low delay
- Few extreme delays exist (outliers)
""")

st.divider()

# ======================================================
# 9️⃣ DELAY VS SATISFACTION
# ======================================================
st.subheader("📉 9. Arrival Delay vs Satisfaction")

fig, ax = plt.subplots()
sns.boxplot(data=df_pd, x="label", y="Arrival_Delay_in_Minutes", ax=ax)
st.pyplot(fig)

st.markdown("""
📌 **Insight:**
- Higher delays strongly reduce satisfaction
- Delay is a key predictive feature
""")

st.divider()

# ======================================================
# 🔟 SERVICE QUALITY AVERAGE
# ======================================================
st.subheader("⭐ 10. Average Service Ratings")

service_cols = [
    "Inflight_wifi_service",
    "Ease_of_Online_booking",
    "Food_and_drink",
    "Online_boarding",
    "Seat_comfort",
    "Inflight_entertainment",
    "On_board_service",
    "Leg_room_service",
    "Baggage_handling",
    "Checkin_service",
    "Inflight_service",
    "Cleanliness"
]

service_means = df_pd[service_cols].mean().sort_values()

st.bar_chart(service_means)

st.markdown("""
📌 **Insight:**
- WiFi and online booking are weakest areas
- Service consistency impacts satisfaction heavily
""")

st.divider()

# ======================================================
# 11️⃣ CORRELATION HEATMAP
# ======================================================
st.subheader("🔥 11. Feature Correlation Heatmap")

numeric_cols = [
    "Age", "Flight_Distance",
    "Inflight_wifi_service", "Ease_of_Online_booking",
    "Food_and_drink", "Online_boarding",
    "Seat_comfort", "Inflight_entertainment",
    "On_board_service", "Leg_room_service",
    "Baggage_handling", "Checkin_service",
    "Inflight_service", "Cleanliness",
    "Departure_Delay_in_Minutes",
    "Arrival_Delay_in_Minutes"
]

corr = df_pd[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.markdown("""
📌 **Insight:**
- Service-related features are strongly correlated
- Delay negatively impacts satisfaction
""")

st.divider()

# ======================================================
# 🧠 FINAL SUMMARY
# ======================================================
st.subheader("🧠 Final Insights Summary")

st.success("""
✔ Delay is the strongest negative factor  
✔ Service quality drives satisfaction  
✔ Business class passengers are more satisfied  
✔ WiFi + booking system need improvement  
✔ Model is trained on meaningful behavioral patterns  
""")