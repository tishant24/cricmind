import requests
import zipfile
import json
import pandas as pd
from pathlib import Path

class CricsheetDownloader:
    """IPL data from Cricsheet"""
    
    def __init__(self):
        """Initialize"""
        self.base_url = "https://cricsheet.org/downloads"
        self.data_dir = Path("data/raw/cricsheet")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("CRICSHEET DATA DOWNLOADER")
        print("="*70)
    
    def download_ipl_data(self):
        """Download IPL dataset"""
        print("\n STEP 1: Downloading IPL Dataset")
        print("-"*70)
        
        url = f"{self.base_url}/ipl_json.zip"
        zip_path = self.data_dir / "ipl_json.zip"
        
        print(f"URL: {url}")
        print(f"Saving to: {zip_path}")
        print("\n⏳ Downloading (2-3 minutes)...")
        
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            mb_down = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r   Progress: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", 
                                  end='', flush=True)
            
            print(f"\n✅ Downloaded: {zip_path.stat().st_size / (1024*1024):.1f} MB")
            return zip_path
            
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            return None
    
    def extract_zip(self, zip_path):
        """Extract ZIP file"""
        print("\n📂 STEP 2: Extracting Files")
        print("-"*70)
        
        extract_dir = self.data_dir / "ipl_json"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                files = zip_ref.namelist()
                print(f"Total files: {len(files)}")
                print(f"Extracting to: {extract_dir}")
                zip_ref.extractall(extract_dir)
            
            print(f"✅ Extracted {len(files)} files")
            return extract_dir
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return None
    
    def process_match_files(self, extract_dir, limit=None):
        """Process JSON match files"""
        print(f"\n🔄 STEP 3: Processing Match Files")
        print("-"*70)
        
        if limit:
            json_files = list(extract_dir.glob("*.json"))[:limit]
        else:
            json_files = list(extract_dir.glob("*.json"))  # Sab files!
        print(f"Processing {len(json_files)} files...")
        
        matches = []
        
        for i, json_file in enumerate(json_files, 1):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                info = data.get('info', {})
                teams = info.get('teams', [])
                
                # Create match dictionary - ALL VALUES AS STRINGS
                match = {
                    'match_id': json_file.stem,
                    'date': str(info.get('dates', [None])[0]) if info.get('dates') else 'Unknown',
                    'venue': str(info.get('venue', 'Unknown')),
                    'city': str(info.get('city', 'Unknown')),
                    'team1': str(teams[0]) if len(teams) > 0 else 'Unknown',
                    'team2': str(teams[1]) if len(teams) > 1 else 'Unknown',
                    'toss_winner': str(info.get('toss', {}).get('winner', 'Unknown')),
                    'toss_decision': str(info.get('toss', {}).get('decision', 'Unknown')),
                    'winner': str(info.get('outcome', {}).get('winner', 'Unknown')),
                    'season': str(info.get('season', 'Unknown')),  # STRING!
                    'match_type': str(info.get('match_type', 'Unknown')),
                }
                
                matches.append(match)
                
                if i % 50 == 0:
                    print(f"   Processed {i}/{len(json_files)}...")
                
            except Exception as e:
                continue
        
        print(f"✅ Processed {len(matches)} matches")
        return matches
    
    def create_dataframe(self, matches):
        """Create DataFrame"""
        print("\n📊 STEP 4: Creating DataFrame")
        print("-"*70)
        
        df = pd.DataFrame(matches)
        print(f"✅ DataFrame: {len(df)} rows × {len(df.columns)} columns")
        
        print("\n📋 Sample:")
        print(df[['date', 'team1', 'team2', 'winner', 'season']].head())
        
        return df
    
    def save_data(self, df):
        """Save data"""
        print("\n💾 STEP 5: Saving Data")
        print("-"*70)
        
        # CSV
        csv_file = "data/processed/cricsheet_matches.csv"
        df.to_csv(csv_file, index=False)
        print(f"✅ CSV: {csv_file}")
        
        # Excel
        excel_file = "data/powerbi/cricsheet_data.xlsx"
        df.to_excel(excel_file, index=False)
        print(f"✅ Excel: {excel_file}")
        
        return csv_file, excel_file
    
    def analyze_data(self, df):
        """Analyze data"""
        print("\n📊 STEP 6: Analysis")
        print("-"*70)
        
        print(f"\n📈 Statistics:")
        print(f"   Total: {len(df)} matches")
        print(f"   Seasons: {df['season'].nunique()}")
        print(f"   Venues: {df['venue'].nunique()}")
        
        print(f"\n🏆 Top Venues:")
        print(df['venue'].value_counts().head())
        
        print(f"\n📊 Seasons:")
        # Now safe because all seasons are strings!
        season_counts = df['season'].value_counts().sort_index()
        for season, count in season_counts.items():
            print(f"   {season}: {count} matches")
    
    def run_pipeline(self, download_new=False):
        """Run full pipeline"""
        print("\n🚀 STARTING PIPELINE")
        print("="*70)
        
        extract_dir = self.data_dir / "ipl_json"
        
        # Check cache
        if extract_dir.exists() and not download_new:
            json_files = list(extract_dir.glob("*.json"))
            if json_files:
                print(f"\n✅ Using cached data ({len(json_files)} files)")
        else:
            # Download
            zip_path = self.download_ipl_data()
            if not zip_path:
                return None
            
            # Extract
            extract_dir = self.extract_zip(zip_path)
            if not extract_dir:
                return None
        
        # Process
        matches = self.process_match_files(extract_dir)
        if not matches:
            return None
        
        # DataFrame
        df = self.create_dataframe(matches)
        
        # Save
        self.save_data(df)
        
        # Analyze
        self.analyze_data(df)
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        
        return df

if __name__ == "__main__":
    downloader = CricsheetDownloader()
    df = downloader.run_pipeline(download_new=False)  # Set True to re-download
    