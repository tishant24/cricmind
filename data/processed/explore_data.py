

import pandas as pd
from pathlib import Path

def explore_collected_data():
    """Explore all the data you've collected so far"""
    
    print("="*70)
    print("🏏 YOUR CRICKET DATA SUMMARY")
    print("="*70)
    
    # Check which files exist
    files = {
        'Live API Data': 'data/processed/matches_dataframe.csv',
        'Cricsheet Historical': 'data/processed/cricsheet_matches.csv',
        'API Excel': 'data/powerbi/matches_data.xlsx',
        'Cricsheet Excel': 'data/powerbi/cricsheet_data.xlsx'
    }
    
    print("\n📂 FILES CHECK:")
    existing_files = {}
    
    for name, path in files.items():
        if Path(path).exists():
            print(f"   ✅ {name}: {path}")
            existing_files[name] = path
        else:
            print(f"   ❌ {name}: Not found")
    
    # Load and analyze existing files
    if 'Live API Data' in existing_files:
        print("\n" + "="*70)
        print("📊 LIVE DATA (FROM API)")
        print("="*70)
        
        df_live = pd.read_csv(existing_files['Live API Data'])
        
        print(f"\n📈 Statistics:")
        print(f"   Total matches: {len(df_live)}")
        print(f"   Columns: {len(df_live.columns)}")
        print(f"   Column names: {', '.join(df_live.columns)}")
        
        print(f"\n🏏 Match Types:")
        if 'match_type' in df_live.columns:
            print(df_live['match_type'].value_counts())
        
        print(f"\n📋 Sample Data (First 3):")
        print(df_live.head(3))
    
    if 'Cricsheet Historical' in existing_files:
        print("\n" + "="*70)
        print("📊 HISTORICAL DATA (FROM CRICSHEET)")
        print("="*70)
        
        df_hist = pd.read_csv(existing_files['Cricsheet Historical'])
        
        print(f"\n📈 Statistics:")
        print(f"   Total matches: {len(df_hist)}")
        print(f"   Columns: {len(df_hist.columns)}")
        print(f"   Date range: {df_hist['date'].min()} to {df_hist['date'].max()}")
        
        print(f"\n🏆 Teams in Data:")
        # Get unique teams
        all_teams = set()
        if 'team1' in df_hist.columns:
            all_teams.update(df_hist['team1'].unique())
        if 'team2' in df_hist.columns:
            all_teams.update(df_hist['team2'].unique())
        
        # Remove 'Unknown'
        all_teams = sorted([t for t in all_teams if t != 'Unknown'])
        
        print(f"   Total teams: {len(all_teams)}")
        for team in all_teams[:10]:  # Show first 10
            print(f"   • {team}")
        
        if len(all_teams) > 10:
            print(f"   ... and {len(all_teams)-10} more")
        
        print(f"\n📊 Seasons Covered:")
        if 'season' in df_hist.columns:
            seasons = df_hist['season'].value_counts().sort_index()
            for season, count in seasons.items():
                print(f"   {season}: {count} matches")
    
    # Overall summary
    print("\n" + "="*70)
    print("🎯 YOUR ACHIEVEMENT SUMMARY")
    print("="*70)
    
    total_matches = 0
    if 'Live API Data' in existing_files:
        total_matches += len(pd.read_csv(existing_files['Live API Data']))
    if 'Cricsheet Historical' in existing_files:
        total_matches += len(pd.read_csv(existing_files['Cricsheet Historical']))
    
    print(f"\n✅ Total cricket matches collected: {total_matches}")
    print(f"✅ Data sources integrated: {len(existing_files)}")
    print(f"✅ Files ready for Power BI: {sum(1 for k in existing_files if 'Excel' in k)}")
    
    print("\n📚 What You Can Do Now:")
    print("   1. Train ML model with historical data")
    print("   2. Create Power BI dashboard")
    print("   3. Upload to AWS S3")
    print("   4. Build prediction app")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    explore_collected_data()