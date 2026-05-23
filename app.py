import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# ---------------- PAGE ----------------
st.set_page_config(page_title="Airline AI", layout="centered")

st.title("✈ Airline Passenger Satisfaction Prediction")

st.markdown("### Enter passenger experience details below 👇")

# ---------------- INPUTS ----------------
age = st.number_input("Age", 1, 100, 25)
flight_distance = st.number_input("Flight Distance", 0, 5000, 1000)

wifi = st.slider("Inflight Wifi Service", 0, 5, 3)
dep_arr = st.slider("Departure/Arrival Time Convenience", 0, 5, 3)
online_booking = st.slider("Ease of Online Booking", 0, 5, 3)
gate_location = st.slider("Gate Location", 0, 5, 3)
food = st.slider("Food and Drink", 0, 5, 3)
online_boarding = st.slider("Online Boarding", 0, 5, 3)
seat_comfort = st.slider("Seat Comfort", 0, 5, 3)
entertainment = st.slider("Inflight Entertainment", 0, 5, 3)
onboard_service = st.slider("Onboard Service", 0, 5, 3)
leg_room = st.slider("Leg Room Service", 0, 5, 3)
baggage = st.slider("Baggage Handling", 0, 5, 3)
checkin = st.slider("Check-in Service", 0, 5, 3)
inflight_service = st.slider("Inflight Service", 0, 5, 3)
cleanliness = st.slider("Cleanliness", 0, 5, 3)

departure_delay = st.number_input("Departure Delay (min)", 0, 1000, 0)
arrival_delay = st.number_input("Arrival Delay (min)", 0, 1000, 0)

# ---------------- SPARK ----------------
spark = SparkSession.builder.appName("StreamlitApp").getOrCreate()

# ---------------- MODEL ----------------
model = PipelineModel.load("gbt_pipeline_model")

# ---------------- FEATURE ORDER (MUST MATCH TRAINING) ----------------
feature_cols = [
    "Age",
    "Flight_Distance",
    "Inflight_wifi_service",
    "Departure_Arrival_time_convenient",
    "Ease_of_Online_booking",
    "Gate_location",
    "Food_and_drink",
    "Online_boarding",
    "Seat_comfort",
    "Inflight_entertainment",
    "On_board_service",
    "Leg_room_service",
    "Baggage_handling",
    "Checkin_service",
    "Inflight_service",
    "Cleanliness",
    "Departure_Delay_in_Minutes",
    "Arrival_Delay_in_Minutes"
]

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Satisfaction"):

    with st.spinner("Analyzing passenger experience..."):

        data = spark.createDataFrame([(
            age,
            flight_distance,
            wifi,
            dep_arr,
            online_booking,
            gate_location,
            food,
            online_boarding,
            seat_comfort,
            entertainment,
            onboard_service,
            leg_room,
            baggage,
            checkin,
            inflight_service,
            cleanliness,
            departure_delay,
            arrival_delay
        )], feature_cols)

        result = model.transform(data)

        prediction = result.select("prediction").collect()[0][0]

        # probability (important for GBT)
        if "probability" in result.columns:
            prob = result.select("probability").collect()[0][0][1]
        else:
            prob = None

    # ---------------- OUTPUT UI ----------------
    st.markdown("## 🎯 Result")

    if prob is not None:
        st.progress(float(prob))

    if prediction == 1:
        st.success(f"😊 Passenger is SATISFIED")
        if prob:
            st.info(f"Confidence: {prob:.2f}")
    else:
        st.error(f"😡 Passenger is NOT SATISFIED")
        if prob:
            st.info(f"Confidence: {prob:.2f}")