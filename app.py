import streamlit as st
import pandas as pd
import joblib 
import datetime
import matplotlib.pyplot as plt

model = joblib.load("Rainfall_prediction_model.pkl")
st.set_page_config(
    page_title="India Rainfall Prediction",
    page_icon="🌧️",
    layout="centered"
)
st.title("India Weather Rainfall Prediction")
st.write("Predict whether rainfall is expected based on weather and geographical conditions.")

state = st.selectbox(
    "State",
    [
        "Andhra Pradesh",
        "Telangana",
        "Tamil Nadu",
        "Karnataka",
        "Kerala",
        "Maharashtra",
        "Odisha",
        "West Bengal",
        "Gujarat",
        "Rajasthan"
    ]
)
district = st.text_input(
    "District",
    "Hyderabad"
)
station_name = st.text_input(
    "Weather Station",
    "Hyderabad"
)
season = st.selectbox(
    "Season",
    [
        "Winter",
        "Summer",
        "Monsoon",
        "Post-Monsoon"
    ]
)
month = st.selectbox(
    "Month",
    [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]
)
avg_temp = st.number_input(
    "Average Temperature (°C)",
    value=25.0
)
min_temp = st.number_input(
    "Minimum Temperature (°C)",
    value=20.0
)
max_temp = st.number_input(
    "Maximum Temperature (°C)",
    value=30.0
)
wind_speed = st.number_input(
    "Wind Speed",
    value=10.0
)
air_pressure = st.number_input(
    "Air Pressure",
    value=1005.0
)
elevation = st.number_input(
    "Elevation",
    value=500.0
)
latitude = st.number_input(
    "Latitude",
    value=17.38
)
longitude = st.number_input(
    "Longitude",
    value=78.48
)
year = st.number_input(
    "Year",
    min_value=1900,
    max_value=2100,
    value=2026
)
day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=1
)

date = st.date_input("Date", datetime.date(2019, 7, 6))
st.write("Date", date)


def rain_effect():
    st.markdown("""
    <style>
    .rain-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 999999;
        overflow: hidden;
    }

    .rain-drop {
        position: absolute;
        top: -50px;
        width: 2px;
        height: 25px;
        background: #4da6ff;
        opacity: 0.7;
        animation: fall 0.8s linear infinite;
    }

    @keyframes fall {
        0% {
            transform: translateY(-50px);
        }
        100% {
            transform: translateY(100vh);
        }
    }

    .drop1 { left: 5%; animation-delay: 0s; }
    .drop2 { left: 10%; animation-delay: .2s; }
    .drop3 { left: 15%; animation-delay: .4s; }
    .drop4 { left: 20%; animation-delay: .1s; }
    .drop5 { left: 25%; animation-delay: .5s; }
    .drop6 { left: 30%; animation-delay: .3s; }
    .drop7 { left: 35%; animation-delay: .7s; }
    .drop8 { left: 40%; animation-delay: .2s; }
    .drop9 { left: 45%; animation-delay: .6s; }
    .drop10 { left: 50%; animation-delay: .1s; }
    .drop11 { left: 55%; animation-delay: .4s; }
    .drop12 { left: 60%; animation-delay: .8s; }
    .drop13 { left: 65%; animation-delay: .3s; }
    .drop14 { left: 70%; animation-delay: .5s; }
    .drop15 { left: 75%; animation-delay: .2s; }
    .drop16 { left: 80%; animation-delay: .6s; }
    .drop17 { left: 85%; animation-delay: .1s; }
    .drop18 { left: 90%; animation-delay: .4s; }
    .drop19 { left: 95%; animation-delay: .7s; }
    </style>

    <div class="rain-container">
        <div class="rain-drop drop1"></div>
        <div class="rain-drop drop2"></div>
        <div class="rain-drop drop3"></div>
        <div class="rain-drop drop4"></div>
        <div class="rain-drop drop5"></div>
        <div class="rain-drop drop6"></div>
        <div class="rain-drop drop7"></div>
        <div class="rain-drop drop8"></div>
        <div class="rain-drop drop9"></div>
        <div class="rain-drop drop10"></div>
        <div class="rain-drop drop11"></div>
        <div class="rain-drop drop12"></div>
        <div class="rain-drop drop13"></div>
        <div class="rain-drop drop14"></div>
        <div class="rain-drop drop15"></div>
        <div class="rain-drop drop16"></div>
        <div class="rain-drop drop17"></div>
        <div class="rain-drop drop18"></div>
        <div class="rain-drop drop19"></div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_excel("eda_data.xlsx")

data = load_data()

# ==========================================================
# 📊 EXPLORATORY DATA ANALYSIS
# ==========================================================

st.header("📊 Exploratory Data Analysis")

st.write(
    "Explore the rainfall dataset using statistics and visualizations."
)

# ----------------------------------------------------------
# Dataset Overview
# ----------------------------------------------------------

st.subheader("📋 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{len(data):,}")

with col2:
    st.metric("Total Features", len(data.columns))

with col3:
    st.metric("Missing Values", f"{data.isnull().sum().sum():,}")


# ----------------------------------------------------------
# Dataset Preview
# ----------------------------------------------------------

st.subheader("🔍 Dataset Preview")

st.dataframe(
    data.head(10),
    use_container_width=True
)


# ----------------------------------------------------------
# Missing Values
# ----------------------------------------------------------

st.subheader("❓ Missing Values")

missing = data.isnull().sum()

missing = missing[missing > 0].sort_values(
    ascending=False
)

if len(missing) > 0:
    st.bar_chart(missing)
else:
    st.success("No missing values found.")


# ----------------------------------------------------------
# Rain vs No Rain
# ----------------------------------------------------------

st.subheader("🌧️ Rain vs No Rain")

data["Rain_Status"] = data["rainfall"].apply(
    lambda x: "Rain" if x > 0 else "No Rain"
)

rain_counts = data["Rain_Status"].value_counts()

fig, ax = plt.subplots()

ax.bar(
    rain_counts.index,
    rain_counts.values
)

ax.set_xlabel("Rainfall Status")
ax.set_ylabel("Number of Records")
ax.set_title("Rain vs No Rain")

st.pyplot(fig)


# ----------------------------------------------------------
# Rainfall by Season
# ----------------------------------------------------------

st.subheader("🌦️ Rainfall by Season")

season_data = pd.crosstab(
    data["season"],
    data["Rain_Status"]
)

st.bar_chart(season_data)


# ----------------------------------------------------------
# Rainfall by Month
# ----------------------------------------------------------

st.subheader("📅 Rainfall by Month")

month_data = pd.crosstab(
    data["month"],
    data["Rain_Status"]
)

st.bar_chart(month_data)


# ----------------------------------------------------------
# Average Temperature
# ----------------------------------------------------------

st.subheader("🌡️ Temperature Analysis")

temperature_data = data.groupby(
    "Rain_Status"
)["avg_temp"].mean()

st.bar_chart(temperature_data)


# ----------------------------------------------------------
# Wind Speed
# ----------------------------------------------------------

st.subheader("💨 Wind Speed Analysis")

wind_data = data.groupby(
    "Rain_Status"
)["wind_speed"].mean()

st.bar_chart(wind_data)


# ----------------------------------------------------------
# Air Pressure
# ----------------------------------------------------------

st.subheader("🌬️ Air Pressure Analysis")

pressure_data = data.groupby(
    "Rain_Status"
)["air_pressure"].mean()

st.bar_chart(pressure_data)

if st.button("🔮 Predict Rainfall"):
    input_data = pd.DataFrame({
        "month": [month],
        "season": [season],
        "station_name": [station_name],
        "state": [state],
        "district": [district],
        "avg_temp": [avg_temp],
        "min_temp": [min_temp],
        "max_temp": [max_temp],
        "wind_speed": [wind_speed],
        "air_pressure": [air_pressure],
        "elevation": [elevation],
        "latitude": [latitude],
        "longitude": [longitude],
        "year": [year],
        "day": [day],
        "date":[date]
    })

    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("🌧️ Rain Expected")
        #show rain only prediction is rain
        rain_effect()
        st.write("The model predicts that rainfall is likely")
    else:
        st.info("☀️ No Rain Expected")
        "The model predicts that rainfall is unlikely"
        st.snow()
