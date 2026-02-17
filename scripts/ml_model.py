# scripts/ml_model.py
"""
CricMind: ML Win Probability Predictor
Using Logistic Regression with:
- Feature Engineering
- Venue Name Standardization
- Normalization (StandardScaler)
- Regularization (L1, L2)
- Cross Validation
- Model Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from dotenv import load_dotenv

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

load_dotenv()

print("="*70)
print("CRICMIND: ML WIN PROBABILITY PREDICTOR")
print("="*70)
print("Using: Logistic Regression + Normalization + Regularization")
print("="*70)


# ============================================================
# VENUE STANDARDIZATION MAP
# Same venue - alag alag names fix karo!
# ============================================================
VENUE_MAP = {

    # ── Premadasa (Sri Lanka) ──────────────────────────────
    'r premadasa stadium'               : 'Premadasa Stadium',
    'r. premadasa stadium'              : 'Premadasa Stadium',
    'r.premadasa stadium'               : 'Premadasa Stadium',
    'premadasa stadium'                 : 'Premadasa Stadium',
    'm premadasa stadium'               : 'Premadasa Stadium',
    'mahinda rajapaksa international'   : 'Premadasa Stadium',

    # ── Wankhede (Mumbai) ─────────────────────────────────
    'wankhede stadium'                  : 'Wankhede Stadium',
    'wankhede stadium, mumbai'          : 'Wankhede Stadium',

    # ── Eden Gardens (Kolkata) ────────────────────────────
    'eden gardens'                      : 'Eden Gardens',
    'eden gardens, kolkata'             : 'Eden Gardens',

    # ── Chinnaswamy (Bangalore) ───────────────────────────
    'm chinnaswamy stadium'             : 'Chinnaswamy Stadium',
    'm. chinnaswamy stadium'            : 'Chinnaswamy Stadium',
    'chinnaswamy stadium'               : 'Chinnaswamy Stadium',
    'royal challengers bangalore'       : 'Chinnaswamy Stadium',

    # ── Chepauk (Chennai) ─────────────────────────────────
    'ma chidambaram stadium'            : 'Chepauk Stadium',
    'm.a. chidambaram stadium'          : 'Chepauk Stadium',
    'chepauk stadium'                   : 'Chepauk Stadium',
    'chepauk, chennai'                  : 'Chepauk Stadium',

    # ── Kotla (Delhi) ─────────────────────────────────────
    'feroz shah kotla'                  : 'Arun Jaitley Stadium',
    'arun jaitley stadium'              : 'Arun Jaitley Stadium',
    'feroz shah kotla ground'           : 'Arun Jaitley Stadium',
    'delhi'                             : 'Arun Jaitley Stadium',

    # ── Rajiv Gandhi (Hyderabad) ──────────────────────────
    'rajiv gandhi international stadium': 'Rajiv Gandhi Stadium',
    'rajiv gandhi intl. cricket stadium': 'Rajiv Gandhi Stadium',
    'rajiv gandhi international'        : 'Rajiv Gandhi Stadium',
    'rajiv gandhi cricket stadium'      : 'Rajiv Gandhi Stadium',

    # ── Narendra Modi (Ahmedabad) ─────────────────────────
    'narendra modi stadium'             : 'Narendra Modi Stadium',
    'sardar patel stadium'              : 'Narendra Modi Stadium',
    'motera stadium'                    : 'Narendra Modi Stadium',

    # ── PCA (Mohali) ──────────────────────────────────────
    'punjab cricket association stadium': 'PCA Stadium Mohali',
    'punjab cricket association is bindra stadium': 'PCA Stadium Mohali',
    'pca stadium'                       : 'PCA Stadium Mohali',
    'is bindra stadium'                 : 'PCA Stadium Mohali',

    # ── Sawai Man Singh (Jaipur) ──────────────────────────
    'sawai mansingh stadium'            : 'SMS Stadium Jaipur',
    'sawai man singh stadium'           : 'SMS Stadium Jaipur',
    'sms stadium'                       : 'SMS Stadium Jaipur',

    # ── Brabourne (Mumbai) ────────────────────────────────
    'brabourne stadium'                 : 'Brabourne Stadium',
    'brabourne stadium, mumbai'         : 'Brabourne Stadium',

    # ── DY Patil (Mumbai) ─────────────────────────────────
    'dy patil sports academy'           : 'DY Patil Stadium',
    'd.y. patil sports academy'         : 'DY Patil Stadium',
    'dy patil stadium'                  : 'DY Patil Stadium',

    # ── Newlands (Cape Town) ──────────────────────────────
    'newlands'                          : 'Newlands Cape Town',
    'newlands, cape town'               : 'Newlands Cape Town',

    # ── MCG (Melbourne) ───────────────────────────────────
    'melbourne cricket ground'          : 'MCG Melbourne',
    'mcg'                               : 'MCG Melbourne',

    # ── SCG (Sydney) ──────────────────────────────────────
    'sydney cricket ground'             : 'SCG Sydney',
    'scg'                               : 'SCG Sydney',

    # ── Lords (London) ────────────────────────────────────
    "lord's"                            : "Lord's London",
    "lord's cricket ground"             : "Lord's London",
}


def standardize_venues(df):
    """
    Fix venue name inconsistencies

    Problem:
    'M Premadasa Stadium'  ← Same venue
    'R Premadasa Stadium'  ← Different spelling
    'Premadasa Stadium'    ← Again different!

    Solution:
    All → 'Premadasa Stadium' ✅
    """

    print("\n🏟️  VENUE STANDARDIZATION")
    print("-"*70)

    original_venues = df['venue'].nunique()

    # Lowercase → lookup → replace
    def clean_venue(venue):
        venue_lower = str(venue).lower().strip()
        return VENUE_MAP.get(venue_lower, venue)

    df['venue'] = df['venue'].apply(clean_venue)

    cleaned_venues = df['venue'].nunique()

    print(f"   Before: {original_venues} unique venues")
    print(f"   After : {cleaned_venues} unique venues")
    print(f"   Merged: {original_venues - cleaned_venues} duplicate venues!")

    print(f"\n   Top 10 Venues (after cleaning):")
    top_venues = df['venue'].value_counts().head(10)
    for venue, count in top_venues.items():
        print(f"   • {venue}: {count} matches")

    return df


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
def load_data():
    """Load cricket data from CSV"""

    print("\nSTEP 1: LOADING DATA")
    print("-"*70)

    csv_path = 'data/processed/cricsheet_matches.csv'

    if not Path(csv_path).exists():
        print(f"File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} matches")
    print(f"   Columns: {list(df.columns)}")

    print(f"\n Sample Data:")
    print(df[['team1', 'team2', 'venue',
              'toss_winner', 'toss_decision', 'winner']].head())

    return df


# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """Create features for ML model"""

    print("\n⚙️  STEP 2: FEATURE ENGINEERING")
    print("-"*70)

    data = df.copy()

    # ── Fix venue names first! ────────────────────────────
    data = standardize_venues(data)

    # ── Remove unknowns ───────────────────────────────────
    data = data[data['winner']        != 'Unknown']
    data = data[data['toss_winner']   != 'Unknown']
    data = data[data['toss_decision'] != 'Unknown']
    data = data[data['team1']         != 'Unknown']
    data = data[data['team2']         != 'Unknown']
    data = data.dropna(subset=['winner', 'team1', 'team2'])

    print(f"\n   After cleaning: {len(data)} matches")

    # ── Feature 1: toss_won ───────────────────────────────
    data['toss_won'] = (
        data['toss_winner'] == data['team1']
    ).astype(int)

    print(f"\n✅ Feature 1: toss_won")
    print(f"   {data['toss_won'].value_counts().to_dict()}")

    # ── Feature 2: bat_first ──────────────────────────────
    data['bat_first'] = (
        ((data['toss_won'] == 1) & (data['toss_decision'] == 'bat')) |
        ((data['toss_won'] == 0) & (data['toss_decision'] == 'field'))
    ).astype(int)

    print(f"\n✅ Feature 2: bat_first")
    print(f"   {data['bat_first'].value_counts().to_dict()}")

    # ── Feature 3 & 4: team encoding ─────────────────────
    le_team   = LabelEncoder()
    all_teams = pd.concat([data['team1'], data['team2']])
    le_team.fit(all_teams)

    data['team1_encoded'] = le_team.transform(data['team1'])
    data['team2_encoded'] = le_team.transform(data['team2'])

    print(f"\n✅ Feature 3 & 4: team1_encoded, team2_encoded")
    print(f"   Total unique teams: {len(le_team.classes_)}")

    # ── Feature 5: venue encoding ─────────────────────────
    le_venue = LabelEncoder()
    data['venue_encoded'] = le_venue.fit_transform(data['venue'])

    print(f"\n✅ Feature 5: venue_encoded")
    print(f"   Total unique venues: {len(le_venue.classes_)}")

    # ── Feature 6: home_advantage ────────────────────────
    # Simple home advantage - team name in venue name?
    data['home_advantage'] = data.apply(
        lambda row: 1 if (
            any(word.lower() in str(row['venue']).lower()
                for word in str(row['team1']).split()
                if len(word) > 3)
        ) else 0,
        axis=1
    )

    print(f"\n✅ Feature 6: home_advantage")
    print(f"   {data['home_advantage'].value_counts().to_dict()}")

    # ── TARGET: team1_won ─────────────────────────────────
    data['team1_won'] = (
        data['winner'] == data['team1']
    ).astype(int)

    win_rate = data['team1_won'].mean() * 100
    print(f"\n✅ Target: team1_won")
    print(f"   Team1 win rate: {win_rate:.1f}%")
    print(f"   {data['team1_won'].value_counts().to_dict()}")

    return data, le_team, le_venue


# ============================================================
# STEP 3: PREPARE FEATURES
# ============================================================
def prepare_features(data):
    """Select features and target"""

    print("\n STEP 3: PREPARING FEATURES")
    print("-"*70)

    feature_columns = [
        'toss_won',       # Toss win?
        'bat_first',      # Batting first?
        'team1_encoded',  # Team1 number
        'team2_encoded',  # Team2 number
        'venue_encoded',  # Venue number
        'home_advantage', # Home ground?
    ]

    X = data[feature_columns]
    y = data['team1_won']

    print(f" Features : {feature_columns}")
    print(f"   X shape  : {X.shape}")
    print(f"   y shape  : {y.shape}")

    print(f"\n📋 Feature Matrix (first 5 rows):")
    print(X.head())

    return X, y, feature_columns


# ============================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================
def split_data(X, y):
    """Split data"""

    print("\n  STEP 4: TRAIN-TEST SPLIT")
    print("-"*70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        random_state = 42,
        stratify     = y
    )

    print(f"   Total   : {len(X)} samples")
    print(f"   Train   : {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
    print(f"   Test    : {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")
    print(f"   Team1 wins in train: {y_train.sum()}")
    print(f"   Team2 wins in train: {(y_train==0).sum()}")

    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 5: NORMALIZATION
# ============================================================
def normalize_features(X_train, X_test):
    """StandardScaler normalization"""

    print("\n STEP 5: NORMALIZATION (StandardScaler)")
    print("-"*70)
    print("Formula: x_scaled = (x - mean) / std_deviation")

    print(f"\nBefore:")
    print(f"   Mean: {X_train.mean().round(2).to_dict()}")
    print(f"   Std : {X_train.std().round(2).to_dict()}")

    scaler = StandardScaler()

    # IMPORTANT: Fit only on train!
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"\nAfter:")
    print(f"   Mean ≈ 0 ")
    print(f"   Std  ≈ 1 ")
    print(f"\n Data leakage prevented!")
    print(f"   (Fit on TRAIN only, then transform both)")

    return X_train_scaled, X_test_scaled, scaler


# ============================================================
# STEP 6: TRAIN MODELS
# ============================================================
def train_models(X_train_scaled, X_test_scaled,
                 y_train, y_test):
    """Train with different regularization"""

    print("\n STEP 6: TRAINING WITH REGULARIZATION")
    print("-"*70)
    print("L1 (Lasso) : Some weights → 0 (feature selection!)")
    print("L2 (Ridge) : All weights small")
    print("C = 1/λ    : Small C = Strong regularization")

    models_config = [
        {
            'name'   : 'L2 (C=0.01) Strong Regularization',
            'penalty': 'l2',
            'C'      : 0.01,
            'solver' : 'lbfgs'
        },
        {
            'name'   : 'L2 (C=0.1)',
            'penalty': 'l2',
            'C'      : 0.1,
            'solver' : 'lbfgs'
        },
        {
            'name'   : 'L2 (C=1.0) Default',
            'penalty': 'l2',
            'C'      : 1.0,
            'solver' : 'lbfgs'
        },
        {
            'name'   : 'L2 (C=10.0) Weak Regularization',
            'penalty': 'l2',
            'C'      : 10.0,
            'solver' : 'lbfgs'
        },
        {
            'name'   : 'L1 (C=0.1)',
            'penalty': 'l1',
            'C'      : 0.1,
            'solver' : 'liblinear'
        },
        {
            'name'   : 'L1 (C=1.0)',
            'penalty': 'l1',
            'C'      : 1.0,
            'solver' : 'liblinear'
        },
    ]

    results      = []
    best_model   = None
    best_accuracy = 0

    for config in models_config:
        print(f"\n Training: {config['name']}")

        model = LogisticRegression(
            penalty     = config['penalty'],
            C           = config['C'],
            solver      = config['solver'],
            max_iter    = 1000,
            random_state= 42
        )

        model.fit(X_train_scaled, y_train)

        y_pred      = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc  = roc_auc_score(y_test, y_pred_prob)

        cv_scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=5, scoring='accuracy'
        )

        print(f"   Accuracy : {accuracy*100:.2f}%")
        print(f"   ROC-AUC  : {roc_auc:.4f}")
        print(f"   CV Score : {cv_scores.mean()*100:.2f}%"
              f" ± {cv_scores.std()*100:.2f}%")

        results.append({
            'name'    : config['name'],
            'penalty' : config['penalty'],
            'C'       : config['C'],
            'accuracy': accuracy,
            'roc_auc' : roc_auc,
            'cv_mean' : cv_scores.mean(),
            'cv_std'  : cv_scores.std(),
            'model'   : model
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model    = model

    return results, best_model


# ============================================================
# STEP 7: EVALUATE
# ============================================================
def evaluate_best_model(results, best_model,
                         X_test_scaled, y_test,
                         feature_columns):
    """Evaluate best model"""

    print("\n" + "="*70)
    print(" STEP 7: MODEL COMPARISON")
    print("="*70)

    print(f"\n{'Model':<40} {'Accuracy':>10} {'ROC-AUC':>10} {'CV':>10}")
    print("-"*73)

    for r in results:
        print(
            f"{r['name']:<40} "
            f"{r['accuracy']*100:>9.2f}% "
            f"{r['roc_auc']:>10.4f} "
            f"{r['cv_mean']*100:>9.2f}%"
        )

    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n Best: {best_result['name']}")
    print(f"   Accuracy: {best_result['accuracy']*100:.2f}%")

    # Detailed report
    y_pred      = best_model.predict(X_test_scaled)
    y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

    print("\n Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Team2 Wins', 'Team1 Wins']
    ))

    cm = confusion_matrix(y_test, y_pred)
    print(" Confusion Matrix:")
    print(f"   True Negative  : {cm[0][0]}")
    print(f"   False Positive : {cm[0][1]}")
    print(f"   False Negative : {cm[1][0]}")
    print(f"   True Positive  : {cm[1][1]}")

    # Feature importance
    print("\n Feature Importance:")
    coef_df = pd.DataFrame({
        'Feature'    : feature_columns,
        'Coefficient': best_model.coef_[0],
        'Abs_Value'  : abs(best_model.coef_[0])
    }).sort_values('Abs_Value', ascending=False)

    print(coef_df[['Feature', 'Coefficient']].to_string(index=False))

    return best_result, y_pred, y_pred_prob


# ============================================================
# STEP 8: VISUALIZATIONS
# ============================================================
def create_visualizations(results, y_test,
                           y_pred, y_pred_prob,
                           best_model, feature_columns):
    """Create charts"""

    print("\nSTEP 8: CREATING VISUALIZATIONS")
    print("-"*70)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        'CricMind: ML Model Analysis (1170 matches)',
        fontsize=16, fontweight='bold'
    )

    # Plot 1: Accuracy Comparison
    names      = [r['name'].split('(')[0].strip() for r in results]
    accuracies = [r['accuracy']*100 for r in results]
    colors     = ['#2196F3','#4CAF50','#FF9800',
                  '#E91E63','#9C27B0','#00BCD4']

    bars = axes[0, 0].bar(
        range(len(names)), accuracies,
        color=colors[:len(names)]
    )
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(
        [f"M{i+1}" for i in range(len(names))]
    )
    axes[0, 0].set_ylim([40, 100])

    for bar, acc in zip(bars, accuracies):
        axes[0, 0].text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.3,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=9
        )

    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d',
        ax=axes[0, 1], cmap='Blues',
        xticklabels=['Team2 Win', 'Team1 Win'],
        yticklabels=['Team2 Win', 'Team1 Win']
    )
    axes[0, 1].set_title('Confusion Matrix')
    axes[0, 1].set_ylabel('Actual')
    axes[0, 1].set_xlabel('Predicted')

    # Plot 3: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc     = roc_auc_score(y_test, y_pred_prob)

    axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC (AUC={roc_auc:.2f})')
    axes[1, 0].plot([0,1],[0,1], 'navy', lw=2,
                    linestyle='--', label='Random')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].legend(loc='lower right')

    # Plot 4: Feature Importance
    coef = abs(best_model.coef_[0])
    axes[1, 1].barh(feature_columns, coef, color='#4CAF50')
    axes[1, 1].set_title('Feature Importance')
    axes[1, 1].set_xlabel('|Coefficient|')

    plt.tight_layout()

    plot_file = 'data/processed/ml_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {plot_file}")


# ============================================================
# STEP 9: SAVE MODEL
# ============================================================
def save_model(best_model, scaler, le_team,
               le_venue, feature_columns, best_result):
    """Save model"""

    print("\n STEP 9: SAVING MODEL")
    print("-"*70)

    Path("models").mkdir(exist_ok=True)

    model_package = {
        'model'          : best_model,
        'scaler'         : scaler,
        'le_team'        : le_team,
        'le_venue'       : le_venue,
        'venue_map'      : VENUE_MAP,      # Save venue map too!
        'feature_columns': feature_columns,
        'accuracy'       : best_result['accuracy'],
        'model_name'     : best_result['name'],
    }

    model_path = 'models/win_predictor.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)

    print(f" Saved: {model_path}")
    print(f"   Model   : {best_result['name']}")
    print(f"   Accuracy: {best_result['accuracy']*100:.2f}%")

    return model_path


# ============================================================
# STEP 10: TEST PREDICTION
# ============================================================
def test_prediction(model_path):
    """Test saved model"""

    print("\nSTEP 10: TESTING PREDICTION")
    print("-"*70)

    with open(model_path, 'rb') as f:
        package = pickle.load(f)

    model           = package['model']
    scaler          = package['scaler']
    le_team         = package['le_team']
    le_venue        = package['le_venue']
    feature_columns = package['feature_columns']

    # Test matches
    test_cases = [
        {
            'team1'        : 'Mumbai Indians',
            'team2'        : 'Chennai Super Kings',
            'venue'        : 'Wankhede Stadium',
            'toss_won'     : 1,
            'toss_decision': 'bat'
        },
        {
            'team1'        : 'Royal Challengers Bangalore',
            'team2'        : 'Kolkata Knight Riders',
            'venue'        : 'Chinnaswamy Stadium',
            'toss_won'     : 0,
            'toss_decision': 'field'
        },
    ]

    for tc in test_cases:
        print(f"\n {tc['team1']} vs {tc['team2']}")
        print(f"   Venue: {tc['venue']}")

        try:
            t1_enc = le_team.transform([tc['team1']])[0]
            t2_enc = le_team.transform([tc['team2']])[0]

            try:
                v_enc = le_venue.transform([tc['venue']])[0]
            except:
                v_enc = 0

            bat_first = int(
                (tc['toss_won'] == 1 and tc['toss_decision'] == 'bat') or
                (tc['toss_won'] == 0 and tc['toss_decision'] == 'field')
            )

            home_adv = int(
                any(w.lower() in tc['venue'].lower()
                    for w in tc['team1'].split() if len(w) > 3)
            )

            features = pd.DataFrame({
                'toss_won'      : [tc['toss_won']],
                'bat_first'     : [bat_first],
                'team1_encoded' : [t1_enc],
                'team2_encoded' : [t2_enc],
                'venue_encoded' : [v_enc],
                'home_advantage': [home_adv]
            })

            scaled = scaler.transform(features)
            prob   = model.predict_proba(scaled)[0]

            t1_prob = prob[1] * 100
            t2_prob = prob[0] * 100

            print(f"   {tc['team1']:30s}: {t1_prob:.1f}%")
            print(f"   {tc['team2']:30s}: {t2_prob:.1f}%")

            winner = tc['team1'] if t1_prob > t2_prob else tc['team2']
            print(f"   🏆 Predicted Winner: {winner}")

        except Exception as e:
            print(f"     Error: {e}")


# ============================================================
# MAIN
# ============================================================
def main():
    """Run complete ML pipeline"""

    # Load
    df = load_data()
    if df is None:
        return

    # Feature engineering
    data, le_team, le_venue = engineer_features(df)

    # Prepare
    X, y, feature_columns = prepare_features(data)

    if len(X) < 20:
        print(f"  Only {len(X)} samples - need more data!")
        return

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Normalize
    X_train_sc, X_test_sc, scaler = normalize_features(
        X_train, X_test
    )

    # Train
    results, best_model = train_models(
        X_train_sc, X_test_sc,
        y_train, y_test
    )

    # Evaluate
    best_result, y_pred, y_pred_prob = evaluate_best_model(
        results, best_model,
        X_test_sc, y_test,
        feature_columns
    )

    # Visualize
    create_visualizations(
        results, y_test,
        y_pred, y_pred_prob,
        best_model, feature_columns
    )

    # Save
    model_path = save_model(
        best_model, scaler,
        le_team, le_venue,
        feature_columns, best_result
    )

    # Test
    test_prediction(model_path)

    print("\n" + "="*70)
    print("🎉 ML PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n Best Model : {best_result['name']}")
    print(f" ccuracy   : {best_result['accuracy']*100:.2f}%")
    


if __name__ == "__main__":
    main()

