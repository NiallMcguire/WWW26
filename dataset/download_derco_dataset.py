#!/usr/bin/env python3
"""
Direct download of DERCo preprocessed RSVP data (articles 0, 1, 2)
Uses pre-discovered folder IDs for faster, more reliable downloads
"""

import requests
from pathlib import Path
from tqdm import tqdm
import time


def download_file(url, output_path):
    """Download file with progress bar"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)


def get_folder_contents(folder_id):
    """Get contents of OSF folder with pagination support"""
    url = f"https://api.osf.io/v2/nodes/rkqbu/files/osfstorage/{folder_id}/"

    all_items = []
    while url:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        all_items.extend(data.get('data', []))

        # Check for next page
        url = data.get('links', {}).get('next')
        if url:
            print(f"  📄 Fetching next page...")
            time.sleep(0.2)  # Be nice to the API

    # Return in the same format as before
    return {'data': all_items}


# Hardcoded folder ID for preprocessed data (already discovered)
PREPROCESSED_FOLDER_ID = "658ee0457094e950c8a17449"
OUTPUT_DIR = Path("DERCo_preprocessed_rsvp")

print("=" * 70)
print("DERCo Direct Download - RSVP Articles 0, 1, 2")
print("=" * 70)
print(f"Output: {OUTPUT_DIR.absolute()}")
print("=" * 70)

# Get all participant folders
print("\n📂 Fetching participant list...")
participants_data = get_folder_contents(PREPROCESSED_FOLDER_ID)
participants = [
    (item['id'], item['attributes']['name'])
    for item in participants_data['data']
    if item['attributes']['kind'] == 'folder'
]

print(f"✅ Found {len(participants)} participants")

# Track statistics
total_files = 0
downloaded = 0
skipped = 0
failed = 0

# Process each participant
for participant_id, participant_name in participants:
    print(f"\n📁 {participant_name}")

    try:
        # Get articles for this participant
        articles_data = get_folder_contents(participant_id)

        # Filter for articles 0, 1, 2
        target_articles = [
            (item['id'], item['attributes']['name'])
            for item in articles_data['data']
            if item['attributes']['kind'] == 'folder'
               and item['attributes']['name'] in ['article_0', 'article_1', 'article_2']
        ]

        # Process each article
        for article_id, article_name in sorted(target_articles):
            try:
                # Get files in article folder
                files_data = get_folder_contents(article_id)

                # Find preprocessed_epoch.fif
                for file_item in files_data['data']:
                    if file_item['attributes']['name'] == 'preprocessed_epoch.fif':
                        total_files += 1

                        # Output path
                        output_path = OUTPUT_DIR / participant_name / article_name / 'preprocessed_epoch.fif'

                        # Skip if exists
                        if output_path.exists():
                            size_mb = output_path.stat().st_size / (1024 * 1024)
                            print(f"  ⏭️  {article_name} ({size_mb:.1f} MB) - already exists")
                            skipped += 1
                            continue

                        # Download
                        download_url = file_item['links']['download']
                        size_mb = file_item['attributes'].get('size', 0) / (1024 * 1024)
                        print(f"  📥 {article_name} ({size_mb:.1f} MB)")

                        download_file(download_url, output_path)
                        downloaded += 1

                        # Be nice to the server
                        time.sleep(0.3)

            except Exception as e:
                print(f"  ❌ Error downloading {article_name}: {e}")
                failed += 1

    except Exception as e:
        print(f"  ❌ Error processing participant: {e}")
        continue

# Summary
print("\n" + "=" * 70)
print("📊 Download Summary")
print("=" * 70)
print(f"Total files found: {total_files}")
print(f"✅ Downloaded: {downloaded}")
print(f"⏭️  Skipped (exist): {skipped}")
print(f"❌ Failed: {failed}")
print(f"📁 Output: {OUTPUT_DIR.absolute()}")
print("=" * 70)

# Next steps
if downloaded > 0:
    print("\n📝 Next Steps:")
    print("1. Verify downloads:")
    print(f"   find {OUTPUT_DIR} -name '*.fif' | wc -l")
    print(f"   # Should be {len(participants) * 3} files ({len(participants)} participants × 3 articles)")
    print("\n2. Load data with MNE:")
    print("   import mne")
    print(f"   epochs = mne.read_epochs('{OUTPUT_DIR}/ACB71/article_0/preprocessed_epoch.fif')")
    print("\n3. Create DERCo ICT pairs dataset for your pipeline")