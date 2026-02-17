# scripts/json_to_dataframe.py
"""
Learn how to convert nested JSON to clean DataFrame
"""

import json
import pandas as pd
from pathlib import Path

def json_to_dataframe():
    """Convert API JSON response to pandas DataFrame"""
    
    print("="*70)
    print("CONVERTING JSON TO DATAFRAME")
    print("="*70)
    
    # Step 1: Find the most recent JSON file
    json_files = list(Path('data/raw').glob('learning_first_api_call_*.json'))
    
    if not json_files:
        print(" No JSON file found. Run first_api_call.py first!")
        return
    
    latest_file = sorted(json_files)[-1]
    print(f"\n Reading file: {latest_file}")
    
    # Step 2: Load JSON
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    matches = data['data']
    print(f" Loaded {len(matches)} matches")
    
    # Step 3: Extract flat structure from nested JSON
    print("\n Extracting data from nested structure...")
    
    flattened_data = []
    
    for match in matches:
        # Create a flat dictionary for each match
        flat_match = {
            'match_id': match.get('id'),
            'name': match.get('name'),
            'match_type': match.get('matchType'),
            'status': match.get('status'),
            'venue': match.get('venue'),
            'date': match.get('date'),
            'date_time_gmt': match.get('dateTimeGMT'),
            
            # Extract teams
            'team1': match.get('teams', [None, None])[0] if len(match.get('teams', [])) > 0 else None,
            'team2': match.get('teams', [None, None])[1] if len(match.get('teams', [])) > 1 else None,
            
            # Extract scores (if available)
            'team1_runs': match.get('score', [{}])[0].get('r') if match.get('score') else None,
            'team1_wickets': match.get('score', [{}])[0].get('w') if match.get('score') else None,
            'team1_overs': match.get('score', [{}])[0].get('o') if match.get('score') else None,
            
            'team2_runs': match.get('score', [{}])[1].get('r') if match.get('score') and len(match.get('score', [])) > 1 else None,
            'team2_wickets': match.get('score', [{}])[1].get('w') if match.get('score') and len(match.get('score', [])) > 1 else None,
            'team2_overs': match.get('score', [{}])[1].get('o') if match.get('score') and len(match.get('score', [])) > 1 else None,
            
            # Match status flags
            'match_started': match.get('matchStarted'),
            'match_ended': match.get('matchEnded'),
            'series_id': match.get('series_id'),
        }
        
        flattened_data.append(flat_match)
    
    # Step 4: Create DataFrame
    df = pd.DataFrame(flattened_data)
    
    print(f" Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    # Step 5: Display DataFrame info
    print("\n" + "="*70)
    print(" DATAFRAME STRUCTURE:")
    print("="*70)
    print(df.info())
    
    # Step 6: Show first few rows
    print("\n" + "="*70)
    print(" FIRST 5 MATCHES:")
    print("="*70)
    print(df.head())
    
    # Step 7:== Show specific columns
    print("\n" + "="*70)
    print(" KEY MATCH INFORMATION:")
    print("="*70)
    print(df[['name', 'match_type', 'team1', 'team2', 'status']].head(10))
    
    # Step 8: Filter only T20 matches
    print("\n" + "="*70)
    print("FILTERING: Only T20 Matches")
    print("="*70)
    t20_matches = df[df['match_type'] == 't20']
    print(f"Found {len(t20_matches)} T20 matches:")
    print(t20_matches[['name', 'team1', 'team2', 'team1_runs', 'team2_runs']])
    
    # Step 9: Filter completed matches
    print("\n" + "="*70)
    print("FILTERING: Completed Matches with Scores")
    print("="*70)
    completed = df[df['match_ended'] == True]
    completed_with_scores = completed[completed['team1_runs'].notna()]
    
    print(f"Found {len(completed_with_scores)} completed matches:")
    print(completed_with_scores[['name', 'team1', 'team1_runs', 'team2', 'team2_runs', 'status']])
    
    # Step 10: Save to CSV
    csv_file = 'data/processed/matches_dataframe.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n Saved DataFrame to: {csv_file}")
    
    # Step 11: Save to Excel (for Power BI later)
    excel_file = 'data/powerbi/matches_data.xlsx'
    df.to_excel(excel_file, index=False, sheet_name='Matches')
    print(f"Saved to Excel: {excel_file}")
    
    return df

if __name__ == "__main__":
    df = json_to_dataframe()