#!/usr/bin/env python3
"""
Download DERCo Dataset from OSF for Brain Passage Retrieval
Downloads preprocessed EEG data (classic RSVP only - articles 0, 1, 2)

Usage:
    python download_derco_dataset.py --output_dir ./DERCo_Dataset
"""

import requests
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import time


def download_file(url, output_path, desc=None):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f, tqdm(
            desc=desc or output_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def get_osf_files(project_id='rkqbu', path=''):
    """Get file listing from OSF project"""
    if path:
        api_url = f"https://api.osf.io/v2/nodes/{project_id}/files/osfstorage/{path}"
    else:
        api_url = f"https://api.osf.io/v2/nodes/{project_id}/files/osfstorage/"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching file list: {e}")
        return None


def explore_osf_structure(project_id='rkqbu', path='', level=0):
    """Recursively explore OSF file structure"""
    files_data = get_osf_files(project_id, path)
    
    if not files_data or 'data' not in files_data:
        return []
    
    file_list = []
    
    for item in files_data['data']:
        name = item['attributes']['name']
        kind = item['attributes']['kind']
        item_path = item['attributes'].get('path', '')
        
        indent = "  " * level
        
        if kind == 'folder':
            print(f"{indent}📁 {name}/")
            # Recursively explore subfolders
            subfolder_id = item['id']
            subfiles = explore_osf_structure(project_id, subfolder_id, level + 1)
            file_list.extend(subfiles)
        
        elif kind == 'file':
            size_mb = item['attributes'].get('size', 0) / (1024 * 1024)
            print(f"{indent}📄 {name} ({size_mb:.2f} MB)")
            
            file_info = {
                'name': name,
                'path': item_path,
                'download_url': item['links']['download'],
                'size': item['attributes'].get('size', 0)
            }
            file_list.append(file_info)
    
    return file_list


def download_derco_preprocessed(output_dir='DERCo_Dataset', explore_only=False):
    """
    Download DERCo preprocessed EEG data for classic RSVP (articles 0, 1, 2)
    
    Args:
        output_dir: Directory to save downloaded files
        explore_only: If True, only explore structure without downloading
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DERCo Dataset Downloader")
    print("=" * 70)
    print(f"OSF Project: https://osf.io/rkqbu/")
    print(f"Output directory: {output_path.absolute()}")
    print("=" * 70)
    
    # Step 1: Explore the file structure
    print("\n📂 Exploring OSF project structure...")
    print("-" * 70)
    
    all_files = explore_osf_structure('rkqbu')
    
    print(f"\n✅ Found {len(all_files)} files total")
    
    if explore_only:
        print("\n🔍 Exploration complete (--explore-only flag set)")
        return
    
    # Step 2: Filter for preprocessed EEG data we need
    print("\n📥 Filtering for preprocessed RSVP data (articles 0, 1, 2)...")
    
    target_files = []
    for file_info in all_files:
        path = file_info['path']
        name = file_info['name']
        
        # We want: EEG_data/preprocessed/{participant}/article_{0,1,2}/preprocessed_epoch.fif
        if 'preprocessed' in path and 'preprocessed_epoch.fif' in name:
            # Check if it's article 0, 1, or 2 (classic RSVP, not RSVP-with-flanker)
            if any(f'article_{i}' in path for i in [0, 1, 2]):
                target_files.append(file_info)
    
    print(f"✅ Found {len(target_files)} preprocessed epoch files for articles 0-2")
    
    if not target_files:
        print("\n⚠️  No preprocessed files found!")
        print("The OSF repository structure may have changed.")
        print("Please check: https://osf.io/rkqbu/files/osfstorage")
        return
    
    # Step 3: Download files
    print(f"\n📥 Downloading {len(target_files)} files...")
    print("-" * 70)
    
    successful = 0
    failed = 0
    
    for file_info in target_files:
        # Reconstruct local path
        relative_path = file_info['path'].lstrip('/')
        local_path = output_path / relative_path
        
        # Skip if already exists
        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"⏭️  Skipping (exists): {relative_path} ({size_mb:.2f} MB)")
            successful += 1
            continue
        
        # Download
        print(f"\n📥 Downloading: {relative_path}")
        if download_file(file_info['download_url'], local_path, desc=file_info['name']):
            successful += 1
        else:
            failed += 1
        
        # Be nice to the server
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Download Summary")
    print("=" * 70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_path.absolute()}")
    print("=" * 70)
    
    # Print next steps
    print("\n📝 Next Steps:")
    print("1. Verify downloaded files:")
    print(f"   ls -lh {output_path}/EEG_data/preprocessed/*/article_{{0,1,2}}/")
    print("\n2. Download story texts from supplementary materials")
    print("   (See OSF page for Supplementary File 1)")
    print("\n3. Create ICT pairs dataset for brain passage retrieval")
    print("   (You'll need to implement a DERCo-specific dataloader)")


def main():
    parser = argparse.ArgumentParser(
        description='Download DERCo Dataset from OSF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore structure only
  python download_derco_dataset.py --explore-only
  
  # Download to default directory
  python download_derco_dataset.py
  
  # Download to custom directory
  python download_derco_dataset.py --output_dir /path/to/DERCo_Dataset
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='DERCo_Dataset',
        help='Directory to save downloaded files (default: DERCo_Dataset)'
    )
    
    parser.add_argument(
        '--explore-only',
        action='store_true',
        help='Only explore file structure without downloading'
    )
    
    args = parser.parse_args()
    
    try:
        download_derco_preprocessed(
            output_dir=args.output_dir,
            explore_only=args.explore_only
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
