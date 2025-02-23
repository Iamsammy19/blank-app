import streamlit as st
import pandas as pd
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
        color: white;
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
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Header
st.title("⚽ Football Match Prediction App ⚽")
st.header("Analyze and Predict Football Matches with Ease")

# Sidebar for user input
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
st.subheader("📊 Match Data")
st.dataframe(df.style.applymap(lambda x: f"background-color: #3498db; color: white;"))

# Interactive Chart
st.subheader("📈 Goals Scored")
fig = px.bar(df, x="Match", y="Goals", color="Match", title="Goals Scored in Matches")
st.plotly_chart(fig)

# Additional Features
st.subheader("🌟 Additional Features")
with st.expander("Click to view tips"):
    st.write("1. **Home Team to Win Either Half**: High probability based on form.")
    st.write("2. **Double Chance (1X)**: Safe bet if home team is strong.")
    st.write("3. **Both Teams to Score (BTTS)**: Likely based on head-to-head stats.")

# Footer
st.markdown(
    """
    <div style="text-align: center; padding: 20px; background-color: #3498db; color: white; border-radius: 10px;">
        <p>© 2023 Football Prediction App. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)
