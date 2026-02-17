# dashboards/app.py
"""
CricMind: Real-Time Win Probability Dashboard
Run: streamlit run dashboards/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title = "CricMind - Cricket Analytics",
    page_icon  = "🏏",
    layout     = "wide"
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    """Load saved ML model"""
    model_path = 'models/win_predictor.pkl'
    if Path(model_path).exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_data():
    """Load cricket data"""
    hist_path = 'data/processed/cricsheet_matches.csv'
    live_path = 'data/processed/matches_dataframe.csv'

    hist_df = pd.read_csv(hist_path) if Path(hist_path).exists() else pd.DataFrame()
    live_df = pd.read_csv(live_path) if Path(live_path).exists() else pd.DataFrame()

    return hist_df, live_df

# ============================================================
# HEADER
# ============================================================
st.title("🏏 CricMind: Cricket Analytics Platform")
st.markdown("**Real-Time Win Probability | Historical Analysis | ML Predictions**")
st.markdown("---")

# Load everything
package      = load_model()
hist_df, live_df = load_data()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Cricket_bat.svg/200px-Cricket_bat.svg.png",
    width=100
)
st.sidebar.title("🏏 CricMind")
st.sidebar.markdown("Cricket Analytics Platform")

page = st.sidebar.radio(
    "Navigation",
    ["🎯 Win Predictor",
     "📊 Historical Analysis",
     "📈 Team Statistics",
     "🔴 Live Matches"]
)

# ============================================================
# PAGE 1: WIN PREDICTOR
# ============================================================
if page == "🎯 Win Predictor":

    st.header("🎯 Win Probability Predictor")
    st.markdown("Select teams and conditions to predict win probability!")

    if package is None:
        st.error("❌ Model not found! Run ml_model.py first.")
    else:
        model    = package['model']
        scaler   = package['scaler']
        le_team  = package['le_team']
        le_venue = package['le_venue']

        # Get team list
        teams = sorted(list(le_team.classes_))
        venues = sorted(list(le_venue.classes_))

        # Input columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔵 Team 1")
            team1 = st.selectbox("Select Team 1", teams, key='team1')

        with col2:
            st.subheader("🔴 Team 2")
            team2_options = [t for t in teams if t != team1]
            team2 = st.selectbox("Select Team 2", team2_options, key='team2')

        # Match conditions
        st.subheader("⚙️ Match Conditions")
        col3, col4, col5 = st.columns(3)

        with col3:
            toss_winner = st.selectbox(
                "Toss Winner",
                [team1, team2]
            )

        with col4:
            toss_decision = st.selectbox(
                "Toss Decision",
                ["bat", "field"]
            )

        with col5:
            venue = st.selectbox(
                "Venue",
                venues
            )

        # Predict button
        if st.button("🎯 PREDICT WIN PROBABILITY", type="primary"):

            try:
                # Feature engineering
                toss_won  = 1 if toss_winner == team1 else 0
                bat_first = int(
                    (toss_won == 1 and toss_decision == 'bat') or
                    (toss_won == 0 and toss_decision == 'field')
                )

                # Encode
                team1_enc = le_team.transform([team1])[0]
                team2_enc = le_team.transform([team2])[0]
                venue_enc = le_venue.transform([venue])[0]

                # Feature vector
                # Home advantage calculate karo
                home_adv = int(
                    any(w.lower() in venue.lower()
                        for w in team1.split() if len(w) > 3)
                )

                # Feature vector - 6 features (home_advantage added!)
                features = pd.DataFrame({
                    'toss_won'      : [toss_won],
                    'bat_first'     : [bat_first],
                    'team1_encoded' : [team1_enc],
                    'team2_encoded' : [team2_enc],
                    'venue_encoded' : [venue_enc],
                    'home_advantage': [home_adv]   # ← NEW!
                })

                # Scale + predict
                features_scaled = scaler.transform(features)
                prob = model.predict_proba(features_scaled)[0]

                team1_prob = prob[1] * 100
                team2_prob = prob[0] * 100

                # Show results
                st.markdown("---")
                st.subheader("🏆 Prediction Result")

                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.metric(
                        label = f"🔵 {team1}",
                        value = f"{team1_prob:.1f}%",
                        delta = f"+{team1_prob - 50:.1f}% vs baseline"
                            if team1_prob > 50 else
                            f"{team1_prob - 50:.1f}% vs baseline"
                    )

                with res_col2:
                    st.metric(
                        label = f"🔴 {team2}",
                        value = f"{team2_prob:.1f}%",
                        delta = f"+{team2_prob - 50:.1f}% vs baseline"
                            if team2_prob > 50 else
                            f"{team2_prob - 50:.1f}% vs baseline"
                    )

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode  = "gauge+number+delta",
                    value = team1_prob,
                    title = {'text': f"{team1} Win Probability"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis' : {'range': [0, 100]},
                        'bar'  : {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40],  'color': "red"},
                            {'range': [40, 60], 'color': "yellow"},
                            {'range': [60, 100],'color': "green"}
                        ],
                        'threshold': {
                            'line' : {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Winner announcement
                winner = team1 if team1_prob > team2_prob else team2
                st.success(f"🏆 Predicted Winner: **{winner}**")

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ============================================================
# PAGE 2: HISTORICAL ANALYSIS
# ============================================================
elif page == "📊 Historical Analysis":

    st.header("📊 Historical IPL Analysis")

    if hist_df.empty:
        st.error("No historical data found!")
    else:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Matches", len(hist_df))
        col2.metric("Seasons", hist_df['season'].nunique())
        col3.metric("Venues", hist_df['venue'].nunique())
        col4.metric("Teams", hist_df['team1'].nunique())

        st.markdown("---")

        # Charts row 1
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("📊 Matches Per Season")
            season_counts = hist_df['season'].value_counts().sort_index()
            fig = px.bar(
                x = season_counts.index,
                y = season_counts.values,
                labels = {'x': 'Season', 'y': 'Matches'},
                color  = season_counts.values,
                color_continuous_scale = 'Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            st.subheader("🎯 Toss Decision")
            toss_counts = hist_df['toss_decision'].value_counts()
            fig = px.pie(
                values = toss_counts.values,
                names  = toss_counts.index,
                title  = "Bat vs Field after Toss"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Charts row 2
        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            st.subheader("🏆 Most Wins")
            wins = hist_df[
                hist_df['winner'] != 'Unknown'
            ]['winner'].value_counts().head(10)

            fig = px.bar(
                x      = wins.values,
                y      = wins.index,
                orientation = 'h',
                labels = {'x': 'Wins', 'y': 'Team'},
                color  = wins.values,
                color_continuous_scale = 'Greens'
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart_col4:
            st.subheader("🏟️ Top Venues")
            venues = hist_df['venue'].value_counts().head(8)
            fig    = px.bar(
                x      = venues.values,
                y      = venues.index,
                orientation = 'h',
                labels = {'x': 'Matches', 'y': 'Venue'},
                color  = venues.values,
                color_continuous_scale = 'Oranges'
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 3: TEAM STATISTICS
# ============================================================
elif page == "📈 Team Statistics":

    st.header("📈 Team Performance Statistics")

    if hist_df.empty:
        st.error("No data found!")
    else:
        # Team selector
        all_teams = sorted(list(set(
            list(hist_df['team1'].unique()) +
            list(hist_df['team2'].unique())
        )))
        all_teams = [t for t in all_teams if t != 'Unknown']

        selected_team = st.selectbox("Select Team", all_teams)

        if selected_team:
            # Filter matches
            team_matches = hist_df[
                (hist_df['team1'] == selected_team) |
                (hist_df['team2'] == selected_team)
            ]

            team_wins = hist_df[
                hist_df['winner'] == selected_team
            ]

            # Stats
            total     = len(team_matches)
            wins      = len(team_wins)
            losses    = total - wins
            win_rate  = (wins/total*100) if total > 0 else 0

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Matches", total)
            m2.metric("Wins", wins)
            m3.metric("Losses", losses)
            m4.metric("Win Rate", f"{win_rate:.1f}%")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                # Win/Loss pie
                fig = px.pie(
                    values = [wins, losses],
                    names  = ['Wins', 'Losses'],
                    title  = f"{selected_team} - Win/Loss",
                    color_discrete_sequence = ['#4CAF50', '#F44336']
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Season performance
                season_wins = hist_df[
                    hist_df['winner'] == selected_team
                ]['season'].value_counts().sort_index()

                fig = px.line(
                    x     = season_wins.index,
                    y     = season_wins.values,
                    title = f"{selected_team} - Wins Per Season",
                    markers = True,
                    labels = {'x': 'Season', 'y': 'Wins'}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Recent matches
            st.subheader("📋 Recent Matches")
            recent = team_matches.tail(10)[
                ['date', 'team1', 'team2', 'venue', 'winner']
            ]
            st.dataframe(recent, use_container_width=True)


# ============================================================
# PAGE 4: LIVE MATCHES
# ============================================================
elif page == "🔴 Live Matches":

    st.header("🔴 Live & Recent Matches")

    if live_df.empty:
        st.warning("No live data. Run first_api_call.py first!")
    else:
        st.metric("Total Matches Found", len(live_df))
        st.markdown("---")

        # Show matches
        if 'match_type' in live_df.columns:
            match_types = ['All'] + list(live_df['match_type'].unique())
            selected_type = st.selectbox("Filter by Type", match_types)

            if selected_type != 'All':
                filtered = live_df[live_df['match_type'] == selected_type]
            else:
                filtered = live_df

            st.dataframe(filtered, use_container_width=True)
        else:
            st.dataframe(live_df, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "**CricMind** | Built with Python, Streamlit, "
    "Scikit-learn, AWS | 🏏"
)