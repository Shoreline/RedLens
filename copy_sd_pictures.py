#!/usr/bin/env python3
"""
Script to copy and rename image files to VisualSketchpad prebaked_images directory.

Usage:
    python copy_sd_pictures.py <file1_path> <file2_path> [comt_sample_id]
    
The script will copy file1 as sd_good.png and file2 as sd_bad.png to all category directories.
"""

import os
import sys
import shutil
from pathlib import Path


def get_categories(base_path):
    """
    Automatically detect all category directories in the prebaked_images folder.
    
    Args:
        base_path: Path to the prebaked_images directory
        
    Returns:
        List of category directory names
    """
    categories = []
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return categories
    
    for item in sorted(base_path.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            categories.append(item.name)
    
    return categories


def copy_and_rename_files(file1_path, file2_path, comt_sample_id='deletion-0107'):
    """
    Copy two files to all category directories and rename them.
    
    Args:
        file1_path: Path to first file (will be renamed to sd_good.png)
        file2_path: Path to second file (will be renamed to sd_bad.png)
        comt_sample_id: Sample ID for the subdirectory (default: 'deletion-0107')
    """
    # Convert to Path objects
    file1 = Path(file1_path).expanduser()
    file2 = Path(file2_path).expanduser()
    
    # Validate input files exist
    if not file1.exists():
        print(f"Error: File not found: {file1}")
        return False
    if not file2.exists():
        print(f"Error: File not found: {file2}")
        return False
    
    # Base path for prebaked_images
    base_path = Path.home() / "code" / "VisualSketchpad" / "agent" / "prebaked_images"
    
    # Get all categories
    categories = get_categories(base_path)
    
    if not categories:
        print(f"Error: No categories found in {base_path}")
        return False
    
    print(f"Found {len(categories)} categories: {', '.join(categories)}")
    print(f"Sample ID: {comt_sample_id}")
    print(f"Source files:")
    print(f"  - {file1} -> sd_good.png")
    print(f"  - {file2} -> sd_bad.png")
    print()
    
    success_count = 0
    
    # Process each category
    for category in categories:
        # Create target directory
        target_dir = base_path / category / comt_sample_id
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Define target file paths
        target_good = target_dir / "sd_good.png"
        target_bad = target_dir / "sd_bad.png"
        
        try:
            # Copy and rename files (will replace if exists)
            shutil.copy2(file1, target_good)
            shutil.copy2(file2, target_bad)
            
            print(f"✓ {category}/{comt_sample_id}/ - Files copied successfully")
            success_count += 1
            
        except Exception as e:
            print(f"✗ {category}/{comt_sample_id}/ - Error: {e}")
    
    print()
    print(f"Completed: {success_count}/{len(categories)} categories processed successfully")
    
    return success_count == len(categories)


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 3:
        print("Usage: python copy_sd_pictures.py <file1_path> <file2_path> [comt_sample_id]")
        print()
        print("Arguments:")
        print("  file1_path      : Path to first file (will be renamed to sd_good.png)")
        print("  file2_path      : Path to second file (will be renamed to sd_bad.png)")
        print("  comt_sample_id  : Optional sample ID (default: 'deletion-0107')")
        print()
        print("Example:")
        print("  python copy_sd_pictures.py image1.jpg image2.jpg")
        print("  python copy_sd_pictures.py image1.jpg image2.jpg my-sample-123")
        sys.exit(1)
    
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    comt_sample_id = sys.argv[3] if len(sys.argv) > 3 else 'deletion-0107'
    
    success = copy_and_rename_files(file1_path, file2_path, comt_sample_id)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
