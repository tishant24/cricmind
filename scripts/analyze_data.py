import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_cricket_data():
    """Perform exploratory data analysis on cricket matches"""
    
    print("="*70)
    print("CRICKET DATA ANALYSIS")
    print("="*70)
    
    # Load the processed data
    csv_file = 'data/processed/matches_dataframe.csv'
    
    if not Path(csv_file).exists():
        print(f" File not found: {csv_file}")
        print("   Run json_to_dataframe.py first!")
        return
    
    df = pd.read_csv(csv_file)
    print(f"\n Loaded {len(df)} matches from CSV")
    
    # ==========================================
    # ANALYSIS 1: Match Types Distribution
    # ==========================================
    print("\n" + "="*70)
    print(" ANALYSIS 1: Match Types Distribution")
    print("="*70)
    
    match_types = df['match_type'].value_counts()
    print(match_types)
    
    print("\n Percentage:")
    print(df['match_type'].value_counts(normalize=True) * 100)
    
    # ==========================================
    # ANALYSIS 2: Completed vs Ongoing Matches
    # ==========================================
    print("\n" + "="*70)
    print(" ANALYSIS 2: Match Status")
    print("="*70)
    
    completed = df['match_ended'].sum()
    ongoing = len(df) - completed
    
    print(f" Completed: {completed} matches")
    print(f" Ongoing: {ongoing} matches")
    print(f" Completion Rate: {(completed/len(df)*100):.1f}%")
    
    # ==========================================
    # ANALYSIS 3: Highest Scoring T20 Matches
    # ==========================================
    print("\n" + "="*70)
    print(" ANALYSIS 3: Highest Scoring T20 Matches")
    print("="*70)
    
    # Filter T20 matches with scores
    t20_df = df[df['match_type'] == 't20'].copy()
    
    # Calculate total runs
    t20_df['total_runs'] = t20_df['team1_runs'].fillna(0) + t20_df['team2_runs'].fillna(0)
    
    # Get top 5 highest scoring
    top_scoring = t20_df.nlargest(5, 'total_runs')
    
    print("\n Top 5 Highest Scoring T20 Matches:")
    for idx, row in top_scoring.iterrows():
        print(f"\n{idx+1}. {row['name']}")
        print(f"   {row['team1']}: {int(row['team1_runs'])} runs")
        print(f"   {row['team2']}: {int(row['team2_runs'])} runs")
        print(f"   Total: {int(row['total_runs'])} runs")
        print(f"   Winner: {row['status']}")
    
    # ==========================================
    # ANALYSIS 4: Average Scores by Match Type
    # ==========================================
    print("\n" + "="*70)
    print(" ANALYSIS 4: Average Scores by Match Type")
    print("="*70)
    
    avg_by_type = df.groupby('match_type').agg({
        'team1_runs': 'mean',
        'team2_runs': 'mean'
    }).round(2)
    
    print(avg_by_type)
    
    # ==========================================
    # ANALYSIS 5: Most Common Venues
    # ==========================================
    print("\n" + "="*70)
    print(" ANALYSIS 5: Top 5 Most Common Venues")
    print("="*70)
    
    top_venues = df['venue'].value_counts().head(5)
    print(top_venues)
    
    # ==========================================
    # ANALYSIS 6: Win Margin Analysis (T20)
    # ==========================================
    print("\n" + "="*70)
    print(" ANALYSIS 6: Win Margins in T20 Matches")
    print("="*70)
    
    t20_completed = t20_df[t20_df['match_ended'] == True].copy()
    
    # Calculate run difference
    t20_completed['run_margin'] = abs(
        t20_completed['team1_runs'] - t20_completed['team2_runs']
    )
    
    print(f"\n Win Margin Statistics (T20):")
    print(f"   Average margin: {t20_completed['run_margin'].mean():.1f} runs")
    print(f"   Largest margin: {t20_completed['run_margin'].max():.0f} runs")
    print(f"   Smallest margin: {t20_completed['run_margin'].min():.0f} runs")
    
    # Find closest match
    closest_match = t20_completed.loc[t20_completed['run_margin'].idxmin()]
    print(f"\n  Closest Match:")
    print(f"   {closest_match['name']}")
    print(f"   Margin: {int(closest_match['run_margin'])} runs")
    
    # ==========================================
    # ANALYSIS 7: Team Performance Summary
    # ==========================================
    print("\n" + "="*70)
    print(" ANALYSIS 7: Team Appearances")
    print("="*70)
    
    # Combine team1 and team2 columns
    all_teams = pd.concat([df['team1'], df['team2']])
    team_counts = all_teams.value_counts().head(10)
    
    print("\n Top 10 Most Active Teams:")
    print(team_counts)
    
    # ==========================================
    # SAVE ANALYSIS RESULTS
    # ==========================================
    print("\n" + "="*70)
    print(" SAVING ANALYSIS RESULTS")
    print("="*70)
    
    # Create summary statistics
    summary = {
        'Total Matches': len(df),
        'T20 Matches': len(df[df['match_type'] == 't20']),
        'Test Matches': len(df[df['match_type'] == 'test']),
        'ODI Matches': len(df[df['match_type'] == 'odi']),
        'Completed': int(df['match_ended'].sum()),
        'Ongoing': int((~df['match_ended']).sum()),
        'Avg T20 Score': f"{df[df['match_type']=='t20']['team1_runs'].mean():.1f}",
        'Unique Teams': len(all_teams.unique()),
        'Unique Venues': df['venue'].nunique()
    }
    
    # Save summary to file
    summary_file = 'data/processed/analysis_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("CRICKET DATA ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key:20s}: {value}\n")
    
    print(f" Analysis summary saved to: {summary_file}")
    
    # ==========================================
    # CREATE VISUALIZATION
    # ==========================================
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cricket Matches Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Match Type Distribution
    match_types.plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Match Types Distribution')
    axes[0, 0].set_xlabel('Match Type')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=0)
    
    # Plot 2: Completed vs Ongoing
    status_data = pd.Series({
        'Completed': completed,
        'Ongoing': ongoing
    })
    status_data.plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    axes[0, 1].set_title('Match Status')
    axes[0, 1].set_ylabel('')
    
    # Plot 3: Top 5 Venues
    top_venues.plot(kind='barh', ax=axes[1, 0], color='coral')
    axes[1, 0].set_title('Top 5 Venues')
    axes[1, 0].set_xlabel('Number of Matches')
    
    # Plot 4: Average Scores by Match Type
    avg_by_type.plot(kind='bar', ax=axes[1, 1], color=['steelblue', 'orange'])
    axes[1, 1].set_title('Average Scores by Match Type')
    axes[1, 1].set_ylabel('Average Runs')
    axes[1, 1].set_xlabel('Match Type')
    axes[1, 1].legend(['Team 1', 'Team 2'])
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'data/processed/analysis_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualizations saved to: {plot_file}")
    
    plt.close()
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "="*70)
    print("✅ DATA ANALYSIS COMPLETE!")
    print("="*70)
    
    print("\nKEY INSIGHTS:")
    print(f"   • Analyzed {len(df)} cricket matches")
    print(f"   • {len(df[df['match_type']=='t20'])} T20 matches")
    print(f"   • {completed} matches completed")
    print(f"   • {len(all_teams.unique())} unique teams")
    print(f"   • {df['venue'].nunique()} different venues")
    
  
    
    print("\n OUTPUT FILES:")
    print(f"   • {summary_file}")
    print(f"   • {plot_file}")
    
    return df

if __name__ == "__main__":
    df = analyze_cricket_data()