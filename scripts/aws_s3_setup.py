# scripts/aws_s3_setup.py
"""
AWS S3 Setup - Mumbai Region (ap-south-1)
Automatic bucket creation + file upload
"""

import boto3
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ============================================================
# CONFIGURATION - Sirf yahan change karo agar zaroorat ho
# ============================================================
BUCKET_NAME = "cricmind-data-2026"   # Unique name chahiye
REGION      = "ap-south-1"           # Mumbai ✅
# ============================================================


def get_s3_client():
    """
    AWS S3 client banao
    .env se keys load karta hai
    """
    
    print("="*60)
    print("☁️  AWS S3 SETUP - MUMBAI REGION")
    print("="*60)

    # Keys load karo
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    # Validate
    print("\n🔑 Checking credentials...")

    if not access_key:
        print(" AWS_ACCESS_KEY_ID not found in .env!")
        return None

    if not secret_key:
        print(" AWS_SECRET_ACCESS_KEY not found in .env!")
        return None

    print(f"✅ Access Key: {access_key[:8]}...{access_key[-4:]}")
    print(f"✅ Secret Key: SET")
    print(f"✅ Region: {REGION} (Mumbai)")

    # S3 client banao
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id     = access_key,
            aws_secret_access_key = secret_key,
            region_name           = REGION
        )

        # Test connection
        s3.list_buckets()
        print(f"✅ Connected to AWS!")

        return s3

    except Exception as e:
        print(f" Connection failed: {e}")
        return None


def create_bucket(s3):
    """
    S3 bucket create karo
    Agar already exist karta hai to skip karo
    """

    print("\n" + "="*60)
    print(" STEP 1: CREATE S3 BUCKET")
    print("="*60)

    try:
        # Check existing buckets
        response  = s3.list_buckets()
        existing  = [b['Name'] for b in response['Buckets']]

        # Already exists?
        if BUCKET_NAME in existing:
            print(f"\n✅ Bucket already exists: {BUCKET_NAME}")
            print(f"   Skipping creation...")
            return True

        # Create bucket
        print(f"\n Creating bucket: {BUCKET_NAME}")
        print(f"   Region: {REGION} (Mumbai)")

        # Mumbai ke liye LocationConstraint zaroori hai
        s3.create_bucket(
            Bucket=BUCKET_NAME,
            CreateBucketConfiguration={
                'LocationConstraint': REGION   # ap-south-1
            }
        )

        print(f"\n Bucket Created Successfully!")
        print(f"   Name   : {BUCKET_NAME}")
        print(f"   Region : {REGION} (Mumbai)")
        print(f"   URL    : https://s3.console.aws.amazon.com/s3/buckets/{BUCKET_NAME}")

        return True

    except Exception as e:
        print(f"\n❌ Bucket creation failed!")
        print(f"   Error: {e}")

        # Common errors
        if 'BucketAlreadyExists' in str(e):
            print("\n Fix: Bucket name already taken globally!")
            print("   Change BUCKET_NAME at top of script")
            print("   Example: cricmind-data-2026-yourname")

        elif 'InvalidAccessKeyId' in str(e):
            print("\n Fix: Wrong Access Key!")
            print("   Check AWS_ACCESS_KEY_ID in .env file")

        elif 'SignatureDoesNotMatch' in str(e):
            print("\n Fix: Wrong Secret Key!")
            print("   Check AWS_SECRET_ACCESS_KEY in .env file")

        return False


def upload_files(s3):
    """
    Local files ko S3 mein upload karo
    """

    print("\n" + "="*60)
    print(" STEP 2: UPLOAD FILES")
    print("="*60)

    # Files list
    files = [
        {
            'local'  : 'data/processed/matches_dataframe.csv',
            's3_key' : 'processed/live_matches.csv',
            'name'   : 'Live API Matches'
        },
        {
            'local'  : 'data/processed/cricsheet_matches.csv',
            's3_key' : 'processed/historical_matches.csv',
            'name'   : 'Historical IPL Matches'
        },
        {
            'local'  : 'data/powerbi/matches_data.xlsx',
            's3_key' : 'powerbi/live_data.xlsx',
            'name'   : 'Power BI Live Data'
        },
        {
            'local'  : 'data/powerbi/cricsheet_data.xlsx',
            's3_key' : 'powerbi/historical_data.xlsx',
            'name'   : 'Power BI Historical Data'
        },
    ]

    uploaded = 0
    skipped  = 0
    failed   = 0

    for file in files:
        local   = file['local']
        s3_key  = file['s3_key']
        name    = file['name']

        print(f"\n📄 {name}")
        print(f"   Local : {local}")
        print(f"   S3    : s3://{BUCKET_NAME}/{s3_key}")

        # File exist karta hai?
        if not Path(local).exists():
            print(f"   ⚠️  File not found - skipping")
            skipped += 1
            continue

        # File size
        size_kb = Path(local).stat().st_size / 1024
        print(f"   Size  : {size_kb:.1f} KB")

        # Upload
        try:
            s3.upload_file(local, BUCKET_NAME, s3_key)
            print(f"   ✅ Uploaded!")
            uploaded += 1

        except Exception as e:
            print(f"Failed: {e}")
            failed += 1

    # Summary
    print(f"\n📊 Upload Summary:")
    print(f"   ✅ Uploaded : {uploaded} files")
    print(f"   ⚠️  Skipped  : {skipped} files")
    print(f"   ❌ Failed   : {failed} files")

    return uploaded


def verify_uploads(s3):
    """
    S3 mein files verify karo
    """

    print("\n" + "="*60)
    print("🔍 STEP 3: VERIFY UPLOADS")
    print("="*60)

    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)

        # Empty bucket?
        if 'Contents' not in response:
            print("\n⚠️  Bucket is empty!")
            return

        files      = response['Contents']
        total_size = sum(f['Size'] for f in files)

        print(f"\n✅ Files in S3:\n")

        for f in files:
            size_kb = f['Size'] / 1024
            print(f"   📄 {f['Key']}")
            print(f"      Size: {size_kb:.1f} KB")
            print()

        print(f"   Total: {len(files)} files | {total_size/1024:.1f} KB")

    except Exception as e:
        print(f"❌ Verify failed: {e}")


def main():
    """
    Main function - sab steps run karo
    """

    print("\n🚀 STARTING AWS S3 PIPELINE - MUMBAI")

    # Step 0: Connect
    s3 = get_s3_client()
    if not s3:
        print("\n❌ Fix credentials first!")
        print("   1. Open .env file")
        print("   2. Add AWS_ACCESS_KEY_ID=your_key")
        print("   3. Add AWS_SECRET_ACCESS_KEY=your_secret")
        return

    # Step 1: Create Bucket
    bucket_ok = create_bucket(s3)
    if not bucket_ok:
        print("\n❌ Fix bucket issue first!")
        return

    # Step 2: Upload Files
    uploaded = upload_files(s3)

    # Step 3: Verify
    verify_uploads(s3)

    # Done!
    print("\n" + "="*60)
    print("🎉 AWS S3 COMPLETE!")
    print("="*60)
    print(f"\n✅ Data is now in AWS Cloud (Mumbai)!")
    print(f"\n🔗 View in AWS Console:")
    print(f"   https://s3.console.aws.amazon.com/s3/buckets/{BUCKET_NAME}")
    print(f"\n💼 Resume mein likhna:")
    print(f"   'Deployed data pipeline to AWS S3 (Mumbai region)'")
    print(f"   'Used boto3 SDK with IAM security'")


if __name__ == "__main__":
    main()