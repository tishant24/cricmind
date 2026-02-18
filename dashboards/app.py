# dashboards/app.py
"""
CricMind Dashboard - Works with Advanced Order-Independent Model
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="CricMind Analytics",
    page_icon="🏏",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load ML model"""
    try:
        with open('models/win_predictor.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        st.error("Model not found!")
        return None

@st.cache_data
def load_data():
    """Load data"""
    try:
        return pd.read_csv('data/processed/cricsheet_matches.csv')
    except:
        st.error("Data not found!")
        return None

# Sidebar
st.sidebar.title("🏏 CricMind")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Win Predictor", "Historical Analysis", "Team Statistics", "Live Matches"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Advanced ML-Powered Analytics")

# ==================================================
# PAGE 1: WIN PREDICTOR
# ==================================================
if page == "Win Predictor":
    st.title("🎯 Win Probability Predictor")
    st.markdown("**Order-Independent ML Model** with Team Ratings & H2H Records")
    st.markdown("---")
    
    model_pkg = load_model()
    
    if model_pkg:
        model = model_pkg['model']
        scaler = model_pkg['scaler']
        le_venue = model_pkg['le_venue']
        team_ratings = model_pkg['team_ratings']
        h2h_dict = model_pkg['h2h_dict']
        
        # Get teams and venues
        teams = sorted(team_ratings.keys())
        venues = sorted(le_venue.classes_)
        
        # Input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            team1 = st.selectbox(
                "Team 1", teams,
                index=teams.index('Mumbai Indians') if 'Mumbai Indians' in teams else 0
            )
        
        with col2:
            team2 = st.selectbox("Team 2", [t for t in teams if t != team1])
        
        with col3:
            venue = st.selectbox(
                "Venue", venues,
                index=venues.index('Wankhede Stadium') if 'Wankhede Stadium' in venues else 0
            )
        
        col4, col5 = st.columns(2)
        
        with col4:
            toss_winner = st.selectbox("Toss Winner", [team1, team2])
        
        with col5:
            toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
        
        if st.button("🔮 Predict Winner", type="primary"):
            
            # Get team ratings
            t1_rating = team_ratings.get(team1, 0.5)
            t2_rating = team_ratings.get(team2, 0.5)
            
            # Calculate features
            rating_diff = t1_rating - t2_rating
            stronger_rating = max(t1_rating, t2_rating)
            weaker_rating = min(t1_rating, t2_rating)
            team1_stronger = int(t1_rating >= t2_rating)
            
            # H2H
            h2h_adv = h2h_dict.get((team1, team2), 0)
            
            # Match conditions
            toss_won = int(toss_winner == team1)
            bat_first = int(
                (toss_won == 1 and toss_decision == 'bat') or
                (toss_won == 0 and toss_decision == 'field')
            )
            
            try:
                v_enc = le_venue.transform([venue])[0]
            except:
                v_enc = 0
            
            home_adv = int(any(
                w.lower() in venue.lower()
                for w in team1.split() if len(w) > 3
            ))
            
            # Create features
            features = pd.DataFrame({
                'toss_won': [toss_won],
                'bat_first': [bat_first],
                'venue_encoded': [v_enc],
                'home_advantage': [home_adv],
                'rating_difference': [rating_diff],
                'stronger_team_rating': [stronger_rating],
                'weaker_team_rating': [weaker_rating],
                'team1_is_stronger': [team1_stronger],
                'h2h_advantage': [h2h_adv],
            })
            
            # Predict
            scaled = scaler.transform(features)
            prob = model.predict_proba(scaled)[0]
            
            team1_prob = prob[1] * 100
            team2_prob = prob[0] * 100
            
            # Display
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    team1,
                    f"{team1_prob:.1f}%",
                    delta=f"{team1_prob - 50:.1f}%" if team1_prob > 50 else None
                )
                st.caption(f"Team Rating: {t1_rating*100:.1f}%")
            
            with col2:
                st.metric(
                    team2,
                    f"{team2_prob:.1f}%",
                    delta=f"{team2_prob - 50:.1f}%" if team2_prob > 50 else None
                )
                st.caption(f"Team Rating: {t2_rating*100:.1f}%")
            
            # H2H info
            if h2h_adv != 0:
                st.info(f"📊 Head-to-Head: {team1 if h2h_adv > 0 else team2} has historical advantage")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=team1_prob,
                title={'text': f"{team1} Win Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Winner
            if team1_prob > team2_prob:
                st.success(f"🏆 Predicted Winner: **{team1}**")
            else:
                st.success(f"🏆 Predicted Winner: **{team2}**")

# ==================================================
# PAGE 2: HISTORICAL ANALYSIS
# ==================================================
elif page == "Historical Analysis":
    st.title("📊 Historical Analysis")
    
    df = load_data()
    
    if df is not None:
        df = df[df['winner'] != 'Unknown']
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Matches", len(df))
        with col2:
            st.metric("Seasons", df['season'].nunique())
        with col3:
            st.metric("Venues", df['venue'].nunique())
        with col4:
            st.metric("Teams", pd.concat([df['team1'], df['team2']]).nunique())
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            season_counts = df['season'].value_counts().sort_index()
            fig = px.bar(
                x=season_counts.index,
                y=season_counts.values,
                title="Matches Per Season",
                labels={'x': 'Season', 'y': 'Matches'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            toss_counts = df['toss_decision'].value_counts()
            fig = px.pie(
                values=toss_counts.values,
                names=toss_counts.index,
                title="Toss Decision"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top teams
        st.subheader("🏆 Top 10 Teams")
        top_teams = df['winner'].value_counts().head(10)
        fig = px.bar(
            x=top_teams.values,
            y=top_teams.index,
            orientation='h',
            labels={'x': 'Wins', 'y': 'Team'}
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# PAGE 3: TEAM STATISTICS
# ==================================================
elif page == "Team Statistics":
    st.title("🏆 Team Statistics")
    
    df = load_data()
    
    if df is not None:
        df = df[df['winner'] != 'Unknown']
        
        # Team selector
        all_teams = sorted(pd.concat([df['team1'], df['team2']]).unique())
        selected_team = st.selectbox("Select Team", all_teams)
        
        # Filter
        team_matches = df[(df['team1'] == selected_team) | (df['team2'] == selected_team)]
        team_wins = len(team_matches[team_matches['winner'] == selected_team])
        team_losses = len(team_matches) - team_wins
        win_rate = (team_wins / len(team_matches) * 100) if len(team_matches) > 0 else 0
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Matches", len(team_matches))
        with col2:
            st.metric("Wins", team_wins)
        with col3:
            st.metric("Losses", team_losses)
        with col4:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=[team_wins, team_losses],
                names=['Wins', 'Losses'],
                title="Win/Loss",
                color_discrete_sequence=['#3FB950', '#F78166']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Season performance
            season_stats = []
            for season in sorted(team_matches['season'].unique()):
                season_data = team_matches[team_matches['season'] == season]
                season_wins = len(season_data[season_data['winner'] == selected_team])
                season_stats.append({
                    'Season': season,
                    'Wins': season_wins,
                    'Matches': len(season_data)
                })
            
            season_df = pd.DataFrame(season_stats)
            fig = px.line(
                season_df,
                x='Season',
                y='Wins',
                title="Season Performance",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

# ==================================================
# PAGE 4: LIVE MATCHES
# ==================================================
else:
    st.title("📺 Live Matches")
    st.info("Live match integration coming soon!")
    
    st.subheader("Mock Data")
    mock = pd.DataFrame({
        'Match': ['MI vs CSK', 'RCB vs KKR', 'DC vs RR'],
        'Venue': ['Wankhede', 'Chinnaswamy', 'Kotla'],
        'Status': ['Completed', 'Live', 'Upcoming'],
        'Winner': ['MI', 'In Progress', 'TBD']
    })
    st.dataframe(mock, use_container_width=True)
