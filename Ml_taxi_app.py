import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="PragyanAI Taxi Predictor", layout="wide")

st.title("PragyanAI Taxi Fare Prediction App (End-to-End ML)")

@st.cache_data
def load_data():
    # Loading the standard Seaborn taxi dataset from online source
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"
    df = pd.read_csv(url)
    
    # Feature Engineering: Extract hour from pickup time
    df['pickup'] = pd.to_datetime(df['pickup'])
    df['hour'] = df['pickup'].dt.hour
    
    # Select relevant columns and drop rows with missing values
    df = df[['distance', 'passengers', 'hour', 'fare']].dropna()
    return df

df = load_data()

st.subheader("PragyanAI Dataset Preview")
st.dataframe(df.head()) # Using dataframe for a cleaner look

# --- Model Training Section ---

# Using all 3 features mentioned in your UI
X = df[['distance', 'passengers', 'hour']]
y = df['fare']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Display Metrics in Columns
col1, col2 = st.columns(2)
with col1:
    st.metric("R2 Score", f"{r2:.2f}")
with col2:
    st.metric("RMSE", f"${rmse:.2f}")

# --- User Input Section ---

st.divider()
st.subheader("Enter Trip Details")

c1, c2, c3 = st.columns(3)

with c1:
    distance = st.number_input("Distance (miles)", min_value=0.0, value=5.0, step=0.1)
with c2:
    passengers = st.number_input("Number of Passengers", min_value=1, max_value=6, value=1)
with c3:
    hour = st.slider("Hour of Day (0-23)", 0, 23, 12)

if st.button("Predict Fare", type="primary"):
    # Create input array matching the 3 features used in training
    input_data = np.array([[distance, passengers, hour]])
    prediction = model.predict(input_data)
    
    # Ensure prediction isn't negative (Linear regression quirk)
    final_fare = max(0, prediction[0])
    
    st.success(f"### Estimated Fare: ${final_fare:.2f}")
