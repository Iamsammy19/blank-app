import streamlit as st
import pandas as pd
import plotly.express as px

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        border: 1px solid #cccccc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Header
st.title("⚽ Football Match Prediction App ⚽")
st.header("Analyze and Predict Football Matches")

# Sidebar
st.sidebar.header("User Input")
team_name = st.sidebar.text_input("Enter Team Name")

# Main Content
if team_name:
    st.success(f"Analyzing data for {team_name}...")

# Example Data
df = pd.DataFrame({
    "Match": ["Match 1", "Match 2", "Match 3"],
    "Goals": [2, 3, 1]
})

# Display Data
st.subheader("Match Data")
st.dataframe(df)

# Interactive Chart
st.subheader("Goals Scored")
fig = px.bar(df, x="Match", y="Goals", color="Match")
st.plotly_chart(fig)

# Footer
st.markdown(
    """
    <div style="text-align: center; padding: 10px; background-color: #f0f2f6;">
        <p>© 2023 Football Prediction App. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)