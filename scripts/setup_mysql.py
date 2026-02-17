# scripts/setup_mysql.py
"""
MySQL Database Setup for CricMind
Uses local MySQL (Workbench)
"""

import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION - .env se load hoga
# ============================================================
MYSQL_HOST     = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT     = os.getenv('MYSQL_PORT', '3306')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'cricmind')
MYSQL_USERNAME = os.getenv('MYSQL_USERNAME', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
# ============================================================


def connect_mysql():
    """Connect to MySQL database"""

    print("="*60)
    print("🗄️  MYSQL DATABASE SETUP")
    print("="*60)

    print(f"\n🔑 Connection Details:")
    print(f"   Host    : {MYSQL_HOST}")
    print(f"   Port    : {MYSQL_PORT}")
    print(f"   Database: {MYSQL_DATABASE}")
    print(f"   Username: {MYSQL_USERNAME}")

    try:
        # URL encode password - dot aur special chars fix!
        safe_pass = quote_plus(MYSQL_PASSWORD)

        # Connection URL
        db_url = (
            f"mysql+pymysql://{MYSQL_USERNAME}:{safe_pass}"
            f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
        )

        # Create engine
        engine = create_engine(db_url)

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        print(f"✅ MySQL Connected Successfully!")
        return engine

    except Exception as e:
        print(f" Connection Failed: {e}")
        
        return None


def create_tables(engine):
    """Create all tables"""

    print("\n" + "="*60)
    print("📋 CREATING TABLES")
    print("="*60)

    tables_sql = [

        # Table 1: Live matches (API se)
        ("""
        CREATE TABLE IF NOT EXISTS live_matches (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            match_id    VARCHAR(100),
            name        VARCHAR(500),
            match_type  VARCHAR(50),
            status      VARCHAR(500),
            venue       VARCHAR(500),
            date        VARCHAR(50),
            team1       VARCHAR(200),
            team2       VARCHAR(200),
            team1_runs  VARCHAR(50),
            team2_runs  VARCHAR(50),
            match_ended VARCHAR(20),
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """, 'live_matches'),

        # Table 2: Historical matches (Cricsheet se)
        ("""
        CREATE TABLE IF NOT EXISTS historical_matches (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            match_id      VARCHAR(100),
            date          VARCHAR(50),
            venue         VARCHAR(500),
            city          VARCHAR(200),
            team1         VARCHAR(200),
            team2         VARCHAR(200),
            toss_winner   VARCHAR(200),
            toss_decision VARCHAR(50),
            winner        VARCHAR(200),
            season        VARCHAR(20),
            match_type    VARCHAR(50),
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """, 'historical_matches'),

        # Table 3: Predictions (ML model se)
        ("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            match_id         VARCHAR(100),
            team1            VARCHAR(200),
            team2            VARCHAR(200),
            team1_win_prob   FLOAT,
            team2_win_prob   FLOAT,
            predicted_winner VARCHAR(200),
            actual_winner    VARCHAR(200),
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """, 'predictions'),

        # Table 4: Teams (dimension table)
        ("""
        CREATE TABLE IF NOT EXISTS teams (
            id           INT AUTO_INCREMENT PRIMARY KEY,
            team_name    VARCHAR(200),
            short_name   VARCHAR(10),
            home_ground  VARCHAR(200),
            city         VARCHAR(100),
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """, 'teams')
    ]

    try:
        with engine.connect() as conn:
            for sql, name in tables_sql:
                conn.execute(text(sql))
                conn.commit()
                print(f" Table created: {name}")

        return True

    except Exception as e:
        print(f" Table creation failed: {e}")
        return False


def load_data(engine):
    """Load CSV data into MySQL"""

    print("\n" + "="*60)
    print("📥 LOADING DATA INTO MYSQL")
    print("="*60)

    # Files to load
    files = [
        {
            'csv'  : 'data/processed/matches_dataframe.csv',
            'table': 'live_matches',
            'name' : 'Live API Matches'
        },
        {
            'csv'  : 'data/processed/cricsheet_matches.csv',
            'table': 'historical_matches',
            'name' : 'Historical IPL Matches'
        }
    ]

    for file in files:
        csv_path = file['csv']
        table    = file['table']
        name     = file['name']

        print(f"\n📄 Loading: {name}")
        print(f"   From : {csv_path}")
        print(f"   To   : MySQL → {table}")

        # File exist karta hai?
        if not Path(csv_path).exists():
            print(f"     File not found - skipping")
            continue

        try:
            # Read CSV
            df = pd.read_csv(csv_path)

            # Clean - missing values fill karo
            df = df.fillna('Unknown')

            # Convert all to string (MySQL compatibility)
            df = df.astype(str)

            # Load to MySQL
            df.to_sql(
                table,
                engine,
                if_exists='replace',
                index=False
            )

            print(f"    Loaded {len(df)} rows!")

        except Exception as e:
            print(f"    Failed: {e}")

    # Teams data load karo
    print(f"\n📄 Loading: IPL Teams")

    teams_data = {
        'team_name'  : [
            'Mumbai Indians',
            'Chennai Super Kings',
            'Royal Challengers Bangalore',
            'Kolkata Knight Riders',
            'Delhi Capitals',
            'Rajasthan Royals',
            'Punjab Kings',
            'Sunrisers Hyderabad',
            'Gujarat Titans',
            'Lucknow Super Giants'
        ],
        'short_name' : [
            'MI', 'CSK', 'RCB', 'KKR',
            'DC', 'RR', 'PBKS', 'SRH',
            'GT', 'LSG'
        ],
        'home_ground': [
            'Wankhede Stadium',
            'MA Chidambaram Stadium',
            'M Chinnaswamy Stadium',
            'Eden Gardens',
            'Arun Jaitley Stadium',
            'SMS Stadium',
            'PCA Stadium',
            'Rajiv Gandhi Intl Stadium',
            'Narendra Modi Stadium',
            'Ekana Cricket Stadium'
        ],
        'city': [
            'Mumbai', 'Chennai', 'Bangalore',
            'Kolkata', 'Delhi', 'Jaipur',
            'Mohali', 'Hyderabad',
            'Ahmedabad', 'Lucknow'
        ]
    }

    try:
        teams_df = pd.DataFrame(teams_data)
        teams_df.to_sql(
            'teams',
            engine,
            if_exists='replace',
            index=False
        )
        print(f"    Loaded {len(teams_df)} teams!")

    except Exception as e:
        print(f"    Teams load failed: {e}")


def run_queries(engine):
    """Run SQL queries"""

    print("\n" + "="*60)
    print("🔍 SQL QUERIES - DATA VERIFY")
    print("="*60)

    queries = [
        (
            'Total Matches in Database',
            'SELECT COUNT(*) as total FROM historical_matches'
        ),
        (
            'Top 5 Winning Teams',
            """
            SELECT winner, COUNT(*) as wins
            FROM historical_matches
            WHERE winner != 'Unknown'
            GROUP BY winner
            ORDER BY wins DESC
            LIMIT 5
            """
        ),
        (
            'Matches Per Season',
            """
            SELECT season, COUNT(*) as matches
            FROM historical_matches
            GROUP BY season
            ORDER BY season
            """
        ),
        (
            'Toss Decision Analysis',
            """
            SELECT toss_decision, COUNT(*) as count
            FROM historical_matches
            WHERE toss_decision != 'Unknown'
            GROUP BY toss_decision
            """
        ),
        (
            'Top 5 Venues',
            """
            SELECT venue, COUNT(*) as matches
            FROM historical_matches
            GROUP BY venue
            ORDER BY matches DESC
            LIMIT 5
            """
        ),
        (
            'All IPL Teams',
            """
            SELECT team_name, short_name, city
            FROM teams
            ORDER BY team_name
            """
        )
    ]

    with engine.connect() as conn:
        for name, sql in queries:
            print(f"\n{name}:")

            try:
                result = pd.read_sql(text(sql), conn)
                print(result.to_string(index=False))

            except Exception as e:
                print(f"   Error: {e}")


def show_summary(engine):
    """Show database summary"""

    print("\n" + "="*60)
    print(" DATABASE SUMMARY")
    print("="*60)

    tables = [
        'live_matches',
        'historical_matches',
        'predictions',
        'teams'
    ]

    with engine.connect() as conn:
        for table in tables:
            try:
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM {table}")
                )
                count = result.scalar()
                print(f"   📋 {table:25s}: {count} rows")
            except:
                print(f"   📋 {table:25s}: empty")

    print(f"\n🔗 MySQL Workbench mein dekho:")
    print(f"   Database : {MYSQL_DATABASE}")
    print(f"   Tables   : live_matches, historical_matches,")
    print(f"              predictions, teams")


def main():
    """Main - sab steps run karo"""

    print("\n🚀 STARTING MYSQL SETUP")

    # Step 1: Connect
    engine = connect_mysql()
    if not engine:
        return

    # Step 2: Create tables
    if not create_tables(engine):
        return

    # Step 3: Load data
    load_data(engine)

    # Step 4: SQL queries
    run_queries(engine)

    # Step 5: Summary
    show_summary(engine)



if __name__ == "__main__":
    main()