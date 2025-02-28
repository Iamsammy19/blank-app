import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, TimeSeriesSplit
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
import time

# --- CSS and Title ---
st.markdown(
    """
    <style>
    /* ... (same CSS as before) ... */
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚽ Ultimate Football Match Prediction and Bet Analysis ⚽")

# --- Enhanced Data Loading and Feature Engineering ---
@st.cache_data
def load_and_preprocess_data():
    conn = sqlite3.connect('soccer.sqlite')
    matches_df = pd.read_sql_query("SELECT * FROM Match;", conn)
    team_df = pd.read_sql_query("SELECT * FROM Team;", conn)
    conn.close()

    # --- Basic Result Feature ---
    matches_df['HomeWin'] = np.where(matches_df['home_team_goal'] > matches_df['away_team_goal'], 1, 0)
    matches_df['AwayWin'] = np.where(matches_df['home_team_goal'] < matches_df['away_team_goal'], 1, 0)
    matches_df['Draw'] = np.where(matches_df['home_team_goal'] == matches_df['away_team_goal'], 1, 0)
    matches_df['result'] = matches_df[['HomeWin', 'AwayWin', 'Draw']].idxmax(axis=1).str.replace('Win', '')
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    matches_df = matches_df.sort_values(by='date')

    # --- Advanced Feature Engineering ---
    def calculate_form(team_id, is_home, date, df, window=10):
        team_matches = df[((df['home_team_api_id'] == team_id) & (is_home)) | ((df['away_team_api_id'] == team_id) & (not is_home))]
        team_matches = team_matches[team_matches['date'] < date].sort_values(by='date', ascending=False).head(window)
        if team_matches.empty:
            return pd.Series([0] * 12)

        wins, losses, draws, goals_scored, goals_conceded, shots_on_target, shots_off_target, fouls_committed, corners, possession = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for index, match in team_matches.iterrows():
            if (match['home_team_api_id'] == team_id and match['HomeWin'] == 1) or (match['away_team_api_id'] == team_id and match['AwayWin'] == 1): wins += 1
            elif (match['home_team_api_id'] == team_id and match['AwayWin'] == 1) or (match['away_team_api_id'] == team_id and match['HomeWin'] == 1): losses += 1
            else: draws += 1
            if match['home_team_api_id'] == team_id:
                goals_scored += match['home_team_goal']; goals_conceded += match['away_team_goal']
                shots_on_target += match['shoton'] if pd.notna(match['shoton']) else 0; shots_off_target += match['shotoff'] if pd.notna(match['shotoff']) else 0
                fouls_committed += match['foulcommit'] if pd.notna(match['foulcommit']) else 0; corners += match['corner'] if pd.notna(match['corner']) else 0
                possession += match['possession'] if pd.notna(match['possession']) else 0
            else:
                goals_scored += match['away_team_goal']; goals_conceded += match['home_team_goal']
                shots_on_target += match['shoton'] if pd.notna(match['shoton']) else 0; shots_off_target += match['shotoff'] if pd.notna(match['shotoff']) else 0
                fouls_committed += match['foulcommit'] if pd.notna(match['foulcommit']) else 0; corners += match['corner'] if pd.notna(match['corner']) else 0
                possession += match['possession'] if pd.notna(match['possession']) else 0

        return pd.Series([wins / window, losses / window, draws / window, goals_scored / window, goals_conceded / window, shots_on_target / window, shots_off_target / window, fouls_committed / window, corners / window, possession / window, goals_scored - goals_conceded, (shots_on_target + shots_off_target) / window])

    matches_df[['home_team_win_rate_form', 'home_team_loss_rate_form', 'home_team_draw_rate_form', 'home_team_avg_goals_scored_form', 'home_team_avg_goals_conceded_form', 'home_team_avg_shots_on_target_form', 'home_team_avg_shots_off_target_form', 'home_team_avg_fouls_committed_form', 'home_team_avg_corners_form', 'home_team_avg_possession_form', 'home_team_goal_difference_form', 'home_team_avg_total_shots_form']] = matches_df.apply(lambda row: calculate_form(row['home_team_api_id'], True, row['date'], matches_df), axis=1)
    matches_df[['away_team_win_rate_form', 'away_team_loss_rate_form', 'away_team_draw_rate_form', 'away_team_avg_goals_scored_form', 'away_team_avg_goals_conceded_form', 'away_team_avg_shots_on_target_form', 'away_team_avg_shots_off_target_form', 'away_team_avg_fouls_committed_form', 'away_team_avg_corners_form', 'away_team_avg_possession_form', 'away_team_goal_difference_form', 'away_team_avg_total_shots_form']] = matches_df.apply(lambda row: calculate_form(row['away_team_api_id'], False, row['date'], matches_df), axis=1)

    def calculate_h2h_stats(home_team_id, away_team_id, date, df, window=10):
        h2h_matches = df[((df['home_team_api_id'] == home_team_id) & (df['away_team_api_id'] == away_team_id)) | ((df['home_team_api_id'] == away_team_id) & (df['away_team_api_id'] == home_team_id))]
        h2h_matches = h2h_matches[h2h_matches['date'] < date].sort_values(by='date', ascending=False).head(window)
        if h2h_matches.empty:
            return pd.Series([0] * 3)

        home_wins, goal_diff, total_goals = 0, 0, 0
        for index, match in h2h_matches.iterrows():
            if ((match['home_team_api_id'] == home_team_id) & (match['HomeWin'] == 1)) or ((match['away_team_api_id'] == home_team_id) & (match['AwayWin'] == 1)): home_wins += 1
            if (match['home_team_api_id'] == home_team_id): goal_diff += (match['home_team_goal'] - match['away_team_goal']); total_goals += (match['home_team_goal'] + match['away_team_goal'])
            else: goal_diff += (match['away_team_goal'] - match['home_team_goal']); total_goals += (match['home_team_goal'] + match['away_team_goal'])

        return pd.Series([home_wins / len(h2h_matches), goal_diff / len(h2h_matches), total_goals / len(h2h_matches)])

    matches_df[['h2h_win_rate', 'h2h_avg_goal_diff', 'h2h_avg_goals']] = matches_df.apply(lambda row: calculate_h2h_stats(row['home_team_api_id'], row['away_team_api_id'], row['date'], matches_df), axis=1)

    odds_features = ['B365H', 'B365D', 'B365A', 'BSH', 'BSD', 'BSA', 'BWH', 'BWD', 'BWA', 'GBH', 'GBD', 'GBA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'WWH', 'WWD', 'WWA']
    for feature in odds_features:
        if feature in matches_df.columns:
            matches_df[feature] = pd.to_numeric(matches_df[feature], errors='coerce')

    matches_df['avg_home_odds'] = matches_df[['B365H', 'BSH', 'BWH', 'GBH', 'IWH', 'LBH', 'PSH', 'SJH', 'VCH', 'WWH']].mean(axis=1, skipna=True)
    matches_df['avg_draw_odds'] = matches_df[['B365D', 'BSD', 'BWD', 'GBD', 'IWD', 'LBD', 'PSD', 'SJD', 'VCD', 'WWD']].mean(axis=1, skipna=True)
    matches_df['avg_away_odds'] = matches_df[['B365A', 'BSA', 'BWA', 'GBA', 'IWA', 'LBA', 'PSA', 'SJA', 'VCA', 'WWA']].mean(axis=1, skipna=True)

    matches_df['form_advantage'] = matches_df['home_team_win_rate_form'] - matches_df['away_team_win_rate_form']
    matches_df['attack_vs_defense'] = matches_df['home_team_avg_goals_scored_form'] * matches_df['away_team_avg_goals_conceded_form']
    matches_df['h2h_form_interaction'] = matches_df['h2h_win_rate'] * matches_df['form_advantage']
    matches_df['odds_ratio_home_away'] = matches_df['avg_away_odds'] / matches_df['avg_home_odds']

    feature_columns = [
        'season', 'stage', 'home_team_api_id', 'away_team_api_id',
        'home_team_win_rate_form', 'home_team_loss_rate_form', 'home_team_draw_rate_form', 'home_team_avg_goals_scored_form', 'home_team_avg_goals_conceded_form',
        'home_team_avg_shots_on_target_form', 'home_team_avg_shots_off_target_form', 'home_team_avg_fouls_committed_form', 'home_team_avg_corners_form', 'home_team_avg_possession_form', 'home_team_goal_difference_form', 'home_team_avg_total_shots_form',
        'away_team_win_rate_form', 'away_team_loss_rate_form', 'away_team_draw_rate_form', 'away_team_avg_goals_scored_form', 'away_team_avg_goals_conceded_form',
        'away_team_avg_shots_on_target_form', 'away_team_avg_shots_off_target_form', 'away_team_avg_fouls_committed_form', 'away_team_avg_corners_form', 'away_team_avg_possession_form', 'away_team_goal_difference_form', 'away_team_avg_total_shots_form',
        'h2h_win_rate', 'h2h_avg_goal_diff', 'h2h_avg_goals',
        'avg_home_odds', 'avg_draw_odds', 'avg_away_odds',
        'form_advantage', 'attack_vs_defense', 'h2h_form_interaction', 'odds_ratio_home_away'
    ]

    features_df = matches_df[feature_columns].copy()
    target_series = matches_df['result']
    features_encoded_df = pd.get_dummies(features_df, columns=['season', 'stage'])

    numerical_cols = features_df.select_dtypes(include=np.number).columns.tolist()
    numerical_cols_to_scale = [col for col in numerical_cols if col not in ['home_team_api_id', 'away_team_api_id']]
    scaler = StandardScaler()
    if numerical_cols_to_scale:
        features_df[numerical_cols_to_scale] = scaler.fit_transform(features_df[numerical_cols_to_scale])
    features_encoded_df = pd.get_dummies(features_df, columns=['season', 'stage'])
    features_encoded_df.fillna(features_encoded_df.mean(), inplace=True)

    return features_encoded_df, target_series

features, target = load_and_preprocess_data()

# --- Model Training ---
@st.cache_resource
def train_model(features, target, selected_leagues, model_name='XGBoost'): # Added selected_leagues
    X = features.copy()
    y = target.copy()

    if selected_leagues: # Filter data by selected leagues if leagues are chosen
        X_filtered = X[X['league'].isin(selected_leagues)] # Assuming 'league' column exists in features
        y_filtered = target[X_filtered.index] # Filter target accordingly
        X, y = X_filtered, y_filtered # Update X and y to filtered versions
        st.info(f"Model trained on leagues: {', '.join(selected_leagues)}") # Inform user about leagues used

    time_split = TimeSeriesSplit(n_splits=5)
    train_index, test_index = next(time_split.split(X))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    param_grids = {
        'XGBoost': {
            'n_estimators': [200, 400, 600, 800], 'max_depth': [4, 5, 6, 7, 8], 'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9], 'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1], 'reg_lambda': [1, 1.5, 2, 2.5]
        },
        'LightGBM': {
            'n_estimators': [200, 400, 600, 800], 'max_depth': [-1, 5, 7, 9], 'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'num_leaves': [31, 64, 128], 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8]
        },
        'CatBoost': {
            'iterations': [200, 400, 600], 'depth': [4, 5, 6], 'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5], 'verbose': [0]
        }
    }

    if model_name == 'XGBoost': model_class = XGBClassifier(random_state=42, objective='multi:softmax', eval_metric='mlogloss', use_label_encoder=False); param_grid = param_grids['XGBoost']
    elif model_name == 'LightGBM': model_class = LGBMClassifier(random_state=42, objective='multiclass'); param_grid = param_grids['LightGBM']
    elif model_name == 'CatBoost': model_class = CatBoostClassifier(random_state=42, objective='MultiClass', verbose=0); param_grid = param_grids['CatBoost']
    else: st.error("Invalid model name selected."); return None, None, None

    cv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(model_class, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)

    with st.spinner(f'Training {model_name} model...'):
        grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, calibrated_model.predict_proba(X_test))

    st.write(f"Model Accuracy ({model_name}): {accuracy:.2f}")
    st.write(f"Log-loss (Calibrated {model_name}): {logloss:.2f}")
    st.write(f"Best Hyperparameters ({model_name}): {grid_search.best_params_}")

    if model_name == 'XGBoost': importance = best_model.feature_importances_
    elif model_name == 'LightGBM': importance = best_model.feature_importances_
    elif model_name == 'CatBoost': importance = best_model.get_feature_importance(grid_search.best_estimator_.get_params())

    feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return calibrated_model, features.columns, feature_importance_df

# --- Function to Display Feature Importance Chart ---
def display_feature_importance_chart(feature_importance_df, model_name):
    fig = px.bar(feature_importance_df, x='Importance', y='Feature', title=f'Feature Importance for {model_name} Model', orientation='h')
    st.plotly_chart(fig)

# --- Web Scraping Function (Enhanced) ---
def scrape_football_data(selected_leagues_seasons):
    base_url = "http://www.football-data.co.uk/data.php"

    st.info(f"Starting web scraping from: {base_url}. Downloading selected leagues and seasons...")

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        league_tables = soup.find_all('table', {'class': 'tablesorter'})
        if not league_tables:
            st.error("Could not find league tables on the page. Website structure might have changed. Please inspect the website's HTML and update the CSS selectors in the code.")
            return

        download_dir = 'football_data_scraped'
        os.makedirs(download_dir, exist_ok=True)
        downloaded_count = 0
        total_downloads = len(selected_leagues_seasons)
        progress_bar = st.progress(0)

        conn_scrape = sqlite3.connect('football_data_scraped.sqlite') # Database for scraped data

        for table in league_tables:
            league_name_element = table.find_previous_sibling('h4')
            if league_name_element:
                league_name = league_name_element.text.strip()
            else:
                continue

            links = table.find_all('a', href=True)
            for link in links:
                if link['href'].endswith('.csv'):
                    season = link.text.strip()

                    if (league_name, season) in selected_leagues_seasons:
                        csv_link = link['href']
                        csv_url = "http://www.football-data.co.uk/" + csv_link
                        csv_filename = csv_url.split('/')[-1]
                        filepath = os.path.join(download_dir, csv_filename)

                        try:
                            csv_response = requests.get(csv_url, stream=True)
                            csv_response.raise_for_status()

                            with open(filepath, 'wb') as f:
                                for chunk in csv_response.iter_content(chunk_size=8192):
                                    f.write(chunk)

                            st.write(f"Downloaded: {csv_filename} ({league_name}, {season})")

                            # --- Load CSV to DataFrame and Store in SQLite ---
                            try:
                                df_scrape = pd.read_csv(filepath)
                                table_name = f"{league_name.replace(' ', '_')}_{season.replace('-', '_')}" # Sanitize table name
                                df_scrape.to_sql(table_name, conn_scrape, if_exists='replace', index=False) # Store in SQLite
                                st.write(f"Stored data for {csv_filename} in SQLite database.")
                            except Exception as db_error:
                                st.error(f"Error storing {csv_filename} in SQLite database: {db_error}")


                            downloaded_count += 1
                            progress_percent = int((downloaded_count / total_downloads) * 100)
                            progress_bar.progress(progress_percent)
                            time.sleep(1)

                        except requests.exceptions.RequestException as e:
                            st.error(f"Download failed for {csv_filename} ({league_name}, {season}): {e}")
        conn_scrape.close() # Close SQLite connection after scraping

        progress_bar.empty()
        if downloaded_count > 0:
            st.success(f"Successfully downloaded and stored {downloaded_count} CSV files to '{download_dir}' directory and 'football_data_scraped.sqlite' database!")
        else:
            st.info("No CSV files were downloaded based on your selections.")

    except requests.exceptions.RequestException as e:
        st.error(f"Web scraping failed due to a network error: {e}")
    except Exception as e:
        st.error(f"An error occurred during web scraping: {e}")

# --- Load Scraped Data from SQLite ---
@st.cache_data
def load_scraped_data_from_sqlite():
    conn_scrape = sqlite3.connect('football_data_scraped.sqlite')
    table_names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn_scrape)
    table_names = table_names['name'].tolist() # Extract table names to list
    scraped_data = {}
    for table_name in table_names:
        try:
            scraped_data[table_name] = pd.read_sql_query(f"SELECT * FROM {table_name};", conn_scrape)
        except Exception as e:
            st.error(f"Error loading table {table_name} from SQLite: {e}")
    conn_scrape.close()
    return scraped_data

# --- Sidebar for User Inputs ---
st.sidebar.header("Match Prediction")

# Model Selection and League for Training
model_choice = st.sidebar.selectbox("Select Model", ['XGBoost', 'LightGBM', 'CatBoost'])

# --- Extract League Options from Data ---
league_options_from_data = features['season'].unique().tolist() # Use 'season' as proxy for leagues in soccer.sqlite
selected_leagues_train = st.sidebar.multiselect("Leagues for Training (Optional, defaults to all)", league_options_from_data, default=league_options_from_data) # Multi-select for leagues

model, feature_names, feature_importance_df = train_model(features, target, selected_leagues_train, model_choice) # Pass selected_leagues to train_model

if model is not None:
    st.subheader("Feature Importance")
    display_feature_importance_chart(feature_importance_df, model_choice)
    st.dataframe(feature_importance_df)

    team_ids_str = [str(int(team_id)) for team_id in pd.concat([features[['home_team_api_id', 'away_team_api_id']].stack()]).unique() if pd.notna(team_id)]
    home_team_api_id = st.sidebar.selectbox("Home Team API ID", team_ids_str)
    away_team_api_id = st.sidebar.selectbox("Away Team API ID", team_ids_str)
    season_predict = st.sidebar.selectbox("Season for Prediction", features['season'].unique()) # Season selection for prediction
    stage = st.sidebar.selectbox("Stage", features['stage'].unique())

    if st.sidebar.button("Predict Match Outcome"):
        input_data = pd.DataFrame({
            'season': [season_predict], 'stage': [stage], 'home_team_api_id': [int(home_team_api_id)], 'away_team_api_id': [int(away_team_api_id)],
            'home_team_win_rate_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[0]], 'home_team_loss_rate_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[1]],
            'home_team_draw_rate_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[2]], 'home_team_avg_goals_scored_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[3]],
            'home_team_avg_goals_conceded_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[4]], 'home_team_avg_shots_on_target_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[5]],
            'home_team_avg_shots_off_target_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[6]], 'home_team_avg_fouls_committed_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[7]],
            'home_team_avg_corners_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[8]], 'home_team_avg_possession_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[9]],
            'home_team_goal_difference_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[10]], 'home_team_avg_total_shots_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[11]],
            'away_team_win_rate_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[0]], 'away_team_loss_rate_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[1]],
            'away_team_draw_rate_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[2]], 'away_team_avg_goals_scored_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[3]],
            'away_team_avg_goals_conceded_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[4]], 'away_team_avg_shots_on_target_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[5]],
            'away_team_avg_shots_off_target_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[6]], 'away_team_avg_fouls_committed_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[7]],
            'away_team_avg_corners_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[8]], 'away_team_avg_possession_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[9]],
            'away_team_goal_difference_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[10]], 'away_team_avg_total_shots_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[11]],
            'h2h_win_rate': [calculate_h2h_stats(int(home_team_api_id), int(away_team_api_id), pd.to_datetime('today'), matches_df).iloc[0]], 'h2h_avg_goal_diff': [calculate_h2h_stats(int(home_team_api_id), int(away_team_api_id), pd.to_datetime('today'), matches_df).iloc[1]],
            'h2h_avg_goals': [calculate_h2h_stats(int(home_team_api_id), int(away_team_api_id), pd.to_datetime('today'), matches_df).iloc[2]], 'avg_home_odds': [matches_df['avg_home_odds'].mean()],
            'avg_draw_odds': [matches_df['avg_draw_odds'].mean()], 'avg_away_odds': [matches_df['avg_away_odds'].mean()], 'form_advantage': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[0] - calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[0]],
            'attack_vs_defense': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[3] * calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[4]],
            'h2h_form_interaction': [calculate_h2h_stats(int(home_team_api_id), int(away_team_api_id), pd.to_datetime('today'), matches_df).iloc[0] * (calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[0] - calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[0])],
            'odds_ratio_home_away': [matches_df['avg_away_odds'].mean() / matches_df['avg_home_odds'].mean()]
        })
        input_data = pd.get_dummies(input_data, columns=['season', 'stage'])
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        prediction = model.predict(input_data)
        predicted_outcome = prediction[0]
        probabilities = model.predict_proba(input_data)[0]

        st.success(f"Predicted Outcome: {predicted_outcome}")
        st.write(f"Probabilities: Home Win = {probabilities[0]:.2f}, Draw = {probabilities[1]:.2f}, Away Win = {probabilities[2]:.2f}")

    # --- Bet Code Analysis ---
    st.sidebar.header("Bet Code Analysis")
    bookie = st.sidebar.selectbox("Select Bookie", ["SportyBet", "Betway", "1xBet", "BetKing", "Bet365", "William Hill"]) # More bookies
    bet_code = st.sidebar.text_input("Enter Bet Code (e.g., LCQXR5):")

    def decode_booking_code(bookie, bet_code):
        bookie_decoders = {
            "SportyBet": {"L": "1X2: Home Win", "C": "Over 2.5 Goals", "Q": "Both Teams to Score (BTTS)", "X": "Correct Score"},
            "Betway": {"1": "1X2: Home Win", "2": "Over 2.5 Goals", "3": "Both Teams to Score (BTTS)"},
            "1xBet": {"A": "1X2: Home Win", "B": "Correct Score", "C": "Double Chance"},
            "BetKing": {"B1": "1X2: Home Win", "O25": "Over 2.5 Goals", "BTTS": "Both Teams to Score (BTTS)"},
            "Bet365": { # Example Bet365 codes - these are illustrative, actual codes might differ
                "H": "1X2: Home Win", "O": "Over 2.5 Goals", "B": "Both Teams to Score (BTTS)", "U": "Under 2.5 Goals", "AH1": "Asian Handicap -1 Home"
            },
            "William Hill": { # Example William Hill codes - illustrative
                "W": "1X2: Home Win", "G": "Over 2.5 Goals", "B": "Both Teams to Score (BTTS)", "C9": "Corners Over 9.5"
            }
        }
        if bookie not in bookie_decoders: st.error("Invalid Bookie Selected."); return []

        decoder = bookie_decoders[bookie]; bets = []
        for char in bet_code:
            if char in decoder: bets.append(decoder[char])
            else: st.warning(f"Unknown Bet Code Character: {char} for {bookie}"); bets.append(f"Unknown Bet: {char}")
        if bets: st.info("Decoded Bets:")
        for bet in bets: st.write(f"- {bet}") # Bullet points for decoded bets
        return bets


    def calculate_probability(bets, home_team_api_id, away_team_api_id, season, stage, model, feature_names):
        total_probability = 1.0
        for bet in bets:
            if bet == "1X2: Home Win":
                try:
                    input_data = pd.DataFrame({
                        'season': [season], 'stage': [stage], 'home_team_api_id': [int(home_team_api_id)], 'away_team_api_id': [int(away_team_api_id)],
                        'home_team_win_rate_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[0]], 'home_team_loss_rate_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[1]],
                        'home_team_draw_rate_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[2]], 'home_team_avg_goals_scored_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[3]],
                        'home_team_avg_goals_conceded_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[4]], 'home_team_avg_shots_on_target_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[5]],
                        'home_team_avg_shots_off_target_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[6]], 'home_team_avg_fouls_committed_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[7]],
                        'home_team_avg_corners_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[8]], 'home_team_avg_possession_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[9]],
                        'home_team_goal_difference_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[10]], 'home_team_avg_total_shots_form': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[11]],
                        'away_team_win_rate_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[0]], 'away_team_loss_rate_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[1]],
                        'away_team_draw_rate_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[2]], 'away_team_avg_goals_scored_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[3]],
                        'away_team_avg_goals_conceded_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[4]], 'away_team_avg_shots_on_target_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[5]],
                        'away_team_avg_shots_off_target_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[6]], 'away_team_avg_fouls_committed_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[7]],
                        'away_team_avg_corners_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[8]], 'away_team_avg_possession_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[9]],
                        'away_team_goal_difference_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[10]], 'away_team_avg_total_shots_form': [calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[11]],
                        'h2h_win_rate': [calculate_h2h_stats(int(home_team_api_id), int(away_team_api_id), pd.to_datetime('today'), matches_df).iloc[0]], 'h2h_avg_goal_diff': [calculate_h2h_stats(int(home_team_api_id), int(away_team_api_id), pd.to_datetime('today'), matches_df).iloc[1]],
                        'h2h_avg_goals': [calculate_h2h_stats(int(home_team_api_id), int(away_team_api_id), pd.to_datetime('today'), matches_df).iloc[2]], 'avg_home_odds': [matches_df['avg_home_odds'].mean()],
                        'avg_draw_odds': [matches_df['avg_draw_odds'].mean()], 'avg_away_odds': [matches_df['avg_away_odds'].mean()], 'form_advantage': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[0] - calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[0]],
                        'attack_vs_defense': [calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[3] * calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[4]],
                        'h2h_form_interaction': [calculate_h2h_stats(int(home_team_api_id), int(away_team_api_id), pd.to_datetime('today'), matches_df).iloc[0] * (calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[0] - calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[0])],
                        'odds_ratio_home_away': [matches_df['avg_away_odds'].mean() / matches_df['avg_home_odds'].mean()]
                    })
                    input_data = pd.get_dummies(input_data, columns=['season', 'stage'])
                    input_data = input_data.reindex(columns=feature_names, fill_value=0)
                    probabilities = model.predict_proba(input_data)[0]; home_win_prob = probabilities[0]
                    st.write(f"1X2 Probabilities: Home Win = {probabilities[0]:.2f}, Draw = {probabilities[1]:.2f}, Away Win = {probabilities[2]:.2f}")
                    total_probability *= home_win_prob
                except Exception as e: st.error(f"Error calculating 1X2 probability: {e}"); return 0

            elif bet == "Over 2.5 Goals":
                avg_goals_form_home = calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[3]
                avg_goals_form_away = calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[3]
                avg_h2h_goals = calculate_h2h_stats(int(home_team_api_id), int(away_team_api_id), pd.to_datetime('today'), matches_df).iloc[2]
                over_2_5_prob_placeholder = min(0.8, (avg_goals_form_home + avg_goals_form_away + avg_h2h_goals) / 5.0)
                st.info(f"Over 2.5 Goals - Probability Placeholder (Form & H2H based)")
                st.write(f"Probability of Over 2.5: {over_2_5_prob_placeholder:.2f}")
                total_probability *= over_2_5_prob_placeholder

            elif bet == "Under 2.5 Goals": # New bet type - Under 2.5 Goals
                under_2_5_prob_placeholder = 1.0 - min(0.8, (avg_goals_form_home + avg_goals_form_away + avg_h2h_goals) / 5.0) # Inverse of Over 2.5 placeholder
                st.info(f"Under 2.5 Goals - Probability Placeholder (Inverse of Over 2.5)")
                st.write(f"Probability of Under 2.5 Goals: {under_2_5_prob_placeholder:.2f}")
                total_probability *= under_2_5_prob_placeholder

            elif bet == "Both Teams to Score (BTTS)":
                avg_goals_scored_home_form = calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[3]
                avg_goals_conceded_away_form = calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[4]
                avg_goals_scored_away_form = calculate_form(int(away_team_api_id), False, pd.to_datetime('today'), matches_df).iloc[3]
                avg_goals_conceded_home_form = calculate_form(int(home_team_api_id), True, pd.to_datetime('today'), matches_df).iloc[4]
                btts_prob_placeholder = min(0.7, (avg_goals_scored_home_form + avg_goals_scored_away_form - (avg_goals_conceded_home_form + avg_goals_conceded_away_form) + 2) / 4.0 )
                st.info("Both Teams to Score (BTTS) - Probability Placeholder (Form based)")
                st.write(f"Probability of BTTS: {btts_prob_placeholder:.2f}")
                total_probability *= btts_prob_placeholder

            elif bet == "Correct Score": st.info("Correct Score Analysis - Probability Placeholder (Generic)"); st.write(f"Most Likely Score: 2-1 (Probability: {0.3:.2f}) - Generic Placeholder."); total_probability *= 0.3
            elif bet == "Double Chance": st.info("Double Chance (1X) - Probability Placeholder (Generic)"); st.write(f"Probability of Double Chance: {0.7:.2f} - Generic Placeholder."); total_probability *= 0.7
            elif bet == "Asian Handicap -1 Home": # New bet type - Asian Handicap
                st.info("Asian Handicap -1 Home - Probability Placeholder (Generic)"); st.write(f"Probability of Asian Handicap -1 Home: {0.5:.2f} - Generic Placeholder."); total_probability *= 0.5 # Example placeholder
            elif bet == "Corners Over 9.5": # New bet type - Corners
                 st.info("Corners Over 9.5 - Probability Placeholder (Generic)"); st.write(f"Probability of Corners Over 9.5: {0.4:.2f} - Generic Placeholder."); total_probability *= 0.4 # Example placeholder
            else: st.error(f"Unsupported Bet Type: {bet}")
        return total_probability

    if st.sidebar.button("Analyze Bet Code"):
        if bet_code:
            bets = decode_booking_code(bookie, bet_code)
            if bets:
                total_probability = calculate_probability(bets, home_team_api_id, away_team_api_id, season_predict, stage, model, feature_names) # Use season_predict
                if total_probability > 0: st.success(f"Total Probability of Winning: {total_probability:.2f}")
        else: st.warning("Please enter a bet code.")

    # --- Web Scraping Section in Sidebar ---
    st.sidebar.header("Scrape Football Data")
    st.sidebar.subheader("Select Leagues and Seasons to Download")

    @st.cache_data(ttl=3600)
    def get_league_season_options():
        base_url = "http://www.football-data.co.uk/data.php"
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            league_tables = soup.find_all('table', {'class': 'tablesorter'})
            league_seasons_options = []
            for table in league_tables:
                league_name_element = table.find_previous_sibling('h4')
                if league_name_element:
                    league_name = league_name_element.text.strip()
                    links = table.find_all('a', href=True)
                    seasons = [link.text.strip() for link in links if link['href'].endswith('.csv')]
                    league_seasons_options.append({'league': league_name, 'seasons': seasons})
            return league_seasons_options
        except Exception as e:
            st.error(f"Error fetching league/season options: {e}")
            return []

    league_season_options = get_league_season_options()

    if league_season_options:
        selected_leagues_seasons = []
        for league_option in league_season_options:
            league_name = league_option['league']
            seasons = league_option['seasons']
            if seasons:
                expanded_league = st.sidebar.expander(league_name, expanded=False)
                with expanded_league:
                    selected_seasons_for_league = [season for season in seasons if st.checkbox(season, key=f"{league_name}-{season}")]
                    selected_leagues_seasons.extend([(league_name, season) for season in selected_seasons_for_league])

        if st.sidebar.button("Download Selected Data"):
            if selected_leagues_seasons: scrape_football_data(selected_leagues_seasons)
            else: st.warning("Please select at least one league and season to download.")
    else: st.sidebar.error("Could not fetch league and season options. Check website or try again later.")

    # --- Data Visualizations ---
    st.sidebar.header("Data Visualizations")
    if st.sidebar.checkbox("Show League Result Distribution"):
        selected_league_viz = st.sidebar.selectbox("Select League for Visualization", features['season'].unique()) # Use 'season' as league proxy
        league_matches_df = matches_df[matches_df['season'] == selected_league_viz] # Filter by selected league
        if not league_matches_df.empty:
            fig_league_results = px.histogram(league_matches_df, x='result', color='season', title=f"Match Result Distribution in {selected_league_viz} Season")
            st.plotly_chart(fig_league_results)
        else: st.info("No data available for the selected league for visualization.")

    if st.sidebar.checkbox("Show Feature Distributions"):
        feature_to_visualize = st.sidebar.selectbox("Select Feature", ['form_advantage', 'h2h_win_rate', 'avg_home_odds']) # Example features
        fig_feature_dist = px.histogram(features, x=feature_to_visualize, title=f"Distribution of {feature_to_visualize}")
        st.plotly_chart(fig_feature_dist)

    # --- About Section ---
    about_expander = st.sidebar.expander("About This App") # About section in sidebar
    with about_expander:
        st.write("This Football Match Prediction and Bet Analysis App is designed to provide insights into football match outcomes and betting strategies.")
        st.subheader("Data Sources:")
        st.write("- Historical match data from `soccer.sqlite` database (European Soccer Database).")
        st.write("- Option to scrape more recent data from [Football-Data.co.uk](http://www.football-data.co.uk).")
        st.subheader("Features:")
        st.write("- Advanced features engineered from match history: team form, head-to-head statistics, betting odds.")
        st.write("- Machine learning models (XGBoost, LightGBM, CatBoost) trained on historical data to predict match outcomes.")
        st.subheader("Bet Code Analysis:")
        st.write("- Decodes bet codes from various bookmakers (SportyBet, Betway, 1xBet, BetKing, Bet365, William Hill).")
        st.write("- Provides probability estimations for decoded bet types (1X2, Over/Under 2.5 Goals, BTTS, Correct Score, Double Chance, Asian Handicap, Corners). **Note:** Probabilities for some bet types are placeholders and for illustrative purposes.")
        st.subheader("Limitations:")
        st.write("- Model accuracy is based on historical data and feature engineering. Future match outcomes are inherently uncertain.")
        st.write("- Bet code analysis probabilities for non-1X2 markets are simplified placeholders and should not be taken as precise predictions.")
        st.write("- Web scraping depends on the structure of Football-Data.co.uk and might break if the website changes.")
        st.subheader("Usage Guidance:")
        st.write("- Use the app as a tool for informational and entertainment purposes only. Betting involves financial risk. Bet responsibly.")
        st.write("- Explore feature importance charts to understand which factors influence model predictions.")
        st.write("- Be aware of the limitations of the bet code analysis probabilities.")

    # --- Footer ---
    st.markdown(
        """
        <div style="text-align: center; padding: 20px; background-color: #007BFF; color: white; border-radius: 8px;">
            <p>© 2025 Football Prediction App. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True
    )