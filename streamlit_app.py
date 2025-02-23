import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import requests
import plotly.express as px

# Custom CSS for a beautiful design
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db;
        color: black;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        border: 1px solid #3498db;
        border-radius: 5px;
        padding: 10px;
    }
    .stHeader {
        color: #3498db;
    }
    .stMarkdown {
        color: #2c3e50;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stExpander {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.title("⚽ Ultimate Football Match Prediction and Bet Analysis ⚽")

# Load enhanced dataset
@st.cache_data
def load_enhanced_data():
    # Example enhanced dataset with more features
    data = pd.DataFrame({
        'home_team': ['Team A', 'Team B', 'Team C', 'Team A', 'Team B', 'Team C'],
        'away_team': ['Team B', 'Team C', 'Team A', 'Team C', 'Team A', 'Team B'],
        'home_form': [5, 3, 2, 4, 1, 3],  # Home team form (last 5 matches)
        'away_form': [2, 4, 3, 1, 5, 2],  # Away team form (last 5 matches)
        'head_to_head': [3, 1, 2, 3, 0, 1],  # Head-to-head record
        'home_goals': [2, 1, 0, 3, 2, 1],
        'away_goals': [1, 2, 1, 0, 3, 2],
        'result': ['Win', 'Lose', 'Draw', 'Win', 'Lose', 'Draw']  # Target variable
    })
    return data

data = load_enhanced_data()

# Train an advanced model (XGBoost)
def train_advanced_model(data):
    features = pd.get_dummies(data[['home_team', 'away_team', 'home_form', 'away_form', 'head_to_head']])
    target = data['result']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    # Evaluate model
    y_pred = model.predict(X_test)
    st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    return model

model = train_advanced_model(data)

# User inputs for match prediction
st.sidebar.header("Match Prediction")
home_team = st.sidebar.selectbox("Home Team", data['home_team'].unique())
away_team = st.sidebar.selectbox("Away Team", data['away_team'].unique())

# Predict match outcome
if st.sidebar.button("Predict Match Outcome"):
    input_data = pd.get_dummies(pd.DataFrame({'home_team': [home_team], 'away_team': [away_team]}))
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Outcome: {prediction}")

# Bookie selection and bet code analysis
st.sidebar.header("Bet Code Analysis")
bookie = st.sidebar.selectbox("Select Bookie", ["SportyBet", "Betway", "1xBet", "BetKing"])
bet_code = st.sidebar.text_input("Enter Bet Code (e.g., LCQXR5):")

# Function to decode booking code
def decode_booking_code(bookie, bet_code):
    if bookie == "SportyBet":
        # Example: SportyBet code format (LCQXR5)
        # Simulate decoding logic
        bets = []
        for char in bet_code:
            if char == "L":
                bets.append("1X2: Home Win")
            elif char == "C":
                bets.append("Over 2.5 Goals")
            elif char == "Q":
                bets.append("Both Teams to Score (BTTS)")
            elif char == "X":
                bets.append("Correct Score")
            else:
                bets.append(f"Unknown Bet: {char}")
        st.info(f"Decoded Bets: {bets}")
        return bets
    elif bookie == "Betway":
        # Example: Betway code format (123456)
        # Simulate decoding logic
        bets = ["1X2: Home Win", "Over 2.5 Goals", "Both Teams to Score (BTTS)"]
        st.info(f"Decoded Bets: {bets}")
        return bets
    elif bookie == "1xBet":
        # Example: 1xBet code format (ABC123)
        # Simulate decoding logic
        bets = ["1X2: Home Win", "Correct Score", "Double Chance"]
        st.info(f"Decoded Bets: {bets}")
        return bets
    elif bookie == "BetKing":
        # Example: BetKing code format (BK12345)
        # Simulate decoding logic
        bets = ["1X2: Home Win", "Over 2.5 Goals", "Both Teams to Score (BTTS)"]
        st.info(f"Decoded Bets: {bets}")
        return bets
    else:
        st.error("Invalid Bookie Selected.")
        return []

# Function to calculate probability of winning
def calculate_probability(bets):
    total_probability = 1.0
    for bet in bets:
        if bet == "1X2: Home Win":
            input_data = pd.get_dummies(pd.DataFrame({'home_team': [home_team], 'away_team': [away_team]}))
            input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
            probabilities = model.predict_proba(input_data)[0]
            st.write(f"1X2 Probabilities: Home Win = {probabilities[0]:.2f}, Draw = {probabilities[1]:.2f}, Away Win = {probabilities[2]:.2f}")
            total_probability *= max(probabilities)
        elif bet == "Over 2.5 Goals":
            st.info(f"Over 2.5 Goals")
            st.write(f"Probability of Over 2.5: {0.6:.2f}")
            total_probability *= 0.6  # Example probability
        elif bet == "Both Teams to Score (BTTS)":
            st.info("Both Teams to Score (BTTS)")
            st.write(f"Probability of BTTS: {0.5:.2f}")
            total_probability *= 0.5  # Example probability
        elif bet == "Correct Score":
            st.info("Correct Score Analysis")
            st.write(f"Most Likely Score: 2-1 (Probability: {0.3:.2f})")
            total_probability *= 0.3  # Example probability
        elif bet == "Double Chance":
            st.info("Double Chance (1X)")
            st.write(f"Probability of Double Chance: {0.7:.2f}")
            total_probability *= 0.7  # Example probability
        else:
            st.error(f"Unsupported Bet Type: {bet}")
    return total_probability

# Analyze bet code
if st.sidebar.button("Analyze Bet Code"):
    if bet_code:
        bets = decode_booking_code(bookie, bet_code)
        if bets:
            total_probability = calculate_probability(bets)
            st.success(f"Total Probability of Winning: {total_probability:.2f}")
    else:
        st.warning("Please enter a bet code.")

# Additional Betting Tips
st.sidebar.header("Additional Betting Tips")
if st.sidebar.button("Generate Betting Tips"):
    st.info("Betting Tips:")
    st.write("1. **Home Team to Win Either Half**: High probability based on form.")
    st.write("2. **Double Chance (1X)**: Safe bet if home team is strong.")
    st.write("3. **Both Teams to Score (BTTS)**: Likely based on head-to-head stats.")
    st.write("4. **Draw No Bet (DNB)**: Eliminates the draw option for safer betting.")
    st.write("5. **Handicap Betting**: Useful for matches with a clear favorite.")

# Upset Detection
st.sidebar.header("Upset Detection")
if st.sidebar.button("Check for Potential Upset"):
    # Simulate upset detection logic
    upset_probability = 0.2  # Example probability
    if upset_probability > 0.15:
        st.warning(f"Potential Upset Detected! Probability: {upset_probability:.2f}")
    else:
        st.success("No Significant Upset Detected.")

# Fetch live match data from an API
def fetch_live_data():
    api_key = "YOUR_FOOTBALL_DATA_API_KEY"  # Replace with your API key
    url = f"https://api.football-data.org/v4/matches"
    headers = {"X-Auth-Token": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch live data.")
        return None

# Display live match data
if st.sidebar.button("Fetch Live Matches"):
    live_data = fetch_live_data()
    if live_data:
        st.subheader("⚽ Live Matches")
        for match in live_data['matches']:
            st.write(f"{match['homeTeam']['name']} vs {match['awayTeam']['name']} - {match['score']['fullTime']['homeTeam']}-{match['score']['fullTime']['awayTeam']}")

# Display historical data
st.subheader("📊 Historical Match Data")
st.write(data)

# Advanced Visualizations
st.subheader("📈 Advanced Visualizations")
fig = px.bar(data, x='home_team', y='home_goals', color='result', title="Home Team Goals by Result")
st.plotly_chart(fig)

# Footer
st.markdown(
    """
    <div style="text-align: center; padding: 20px; background-color: #3498db; color: white; border-radius: 10px;">
        <p>© 2023 Football Prediction App. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)
