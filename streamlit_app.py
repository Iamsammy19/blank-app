import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import sqlite3
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import os

# --- CSS and Title ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 5px;
        padding: 10px;
        color: #333333;
    }
    .stHeader .stMarkdown {
        color: #007BFF;
    }
    .stMarkdown {
        color: #555555;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        background-color: #ffffff;
        padding: 10px;
    }
    .stExpander {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #eeeeee;
        padding: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #e6e9ef;
        color: #333333;
    }
    .sidebar .st-ef {
        color: #007BFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚽ Ultimate Football Match Prediction and Bet Analysis ⚽")

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    conn = sqlite3.connect('soccer.sqlite')
    matches_df = pd.read_sql_query("SELECT * FROM Match;", conn)
    team_df = pd.read_sql_query("SELECT * FROM Team;", conn)
    conn.close()

    # Feature Engineering
    matches_df['HomeWin'] = np.where(matches_df['home_team_goal'] > matches_df['away_team_goal'], 1, 0)
    matches_df['AwayWin'] = np.where(matches_df['home_team_goal'] < matches_df['away_team_goal'], 1, 0)
    matches_df['Draw'] = np.where(matches_df['home_team_goal'] == matches_df['away_team_goal'], 1, 0)
    matches_df['result'] = matches_df[['HomeWin', 'AwayWin', 'Draw']].idxmax(axis=1).str.replace('Win', '')
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    matches_df = matches_df.sort_values(by='date')

    # Handle missing values
    matches_df.fillna(matches_df.mean(), inplace=True)

    return matches_df

matches_df = load_and_preprocess_data()

# --- Model Training ---
@st.cache_resource
def train_model(features, target, model_name='XGBoost'):
    X = features.copy()
    y = target.copy()

    # Time-Based Split
    time_split = TimeSeriesSplit(n_splits=5)
    train_index, test_index = next(time_split.split(X))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Model Selection
    if model_name == 'XGBoost':
        model = XGBClassifier(random_state=42, objective='multi:softmax', eval_metric='mlogloss', use_label_encoder=False)
    elif model_name == 'LightGBM':
        model = LGBMClassifier(random_state=42, objective='multiclass')
    elif model_name == 'CatBoost':
        model = CatBoostClassifier(random_state=42, objective='MultiClass', verbose=0)
    else:
        st.error("Invalid model name selected.")
        return None, None, None

    # Train Model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, model.predict_proba(X_test))

    st.write(f"Model Accuracy ({model_name}): {accuracy:.2f}")
    st.write(f"Log-loss ({model_name}): {logloss:.2f}")

    return model, X.columns

# --- Sidebar for User Inputs ---
st.sidebar.header("Match Prediction")
model_choice = st.sidebar.selectbox("Select Model", ['XGBoost', 'LightGBM', 'CatBoost'])
model, feature_names = train_model(matches_df.drop(columns=['result']), matches_df['result'], model_choice)

if model is not None:
    st.subheader("Feature Importance")
    if model_choice == 'XGBoost':
        importance = model.feature_importances_
    elif model_choice == 'LightGBM':
        importance = model.feature_importances_
    elif model_choice == 'CatBoost':
        importance = model.get_feature_importance()

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    st.dataframe(feature_importance_df)

# --- Footer ---
st.markdown(
    """
    <div style="text-align: center; padding: 20px; background-color: #007BFF; color: white; border-radius: 8px;">
        <p>© 2025 Football Prediction App. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)