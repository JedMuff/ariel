#!/usr/bin/env python3
"""
Script to copy all database.csv files from __data__/ subdirectories
to a specified destination folder, renaming them to their experiment names.
"""

import os
import shutil
import argparse
from pathlib import Path


def copy_databases(data_dir: str = "__data__", dest_dir: str = None):
    """
    Copy all database.csv files from experiment directories to destination folder.

    Args:
        data_dir: Source directory containing experiment subdirectories (default: __data__)
        dest_dir: Destination directory to copy files to
    """
    if dest_dir is None:
        raise ValueError("Destination directory must be specified")

    # Convert to Path objects
    data_path = Path(data_dir)
    dest_path = Path(dest_dir)

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # Check if data directory exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found")

    # Track copied files
    copied_count = 0
    skipped_count = 0

    print(f"Searching for database.csv files in {data_path}...")
    print(f"Destination: {dest_path.absolute()}\n")

    # Iterate through all subdirectories in data_dir
    for experiment_dir in sorted(data_path.iterdir()):
        if not experiment_dir.is_dir():
            continue

        # Look for database.csv in this experiment directory
        database_file = experiment_dir / "database.csv"

        if database_file.exists():
            # Use experiment directory name as the new filename
            experiment_name = experiment_dir.name
            dest_file = dest_path / f"{experiment_name}.csv"

            # Copy the file
            try:
                shutil.copy2(database_file, dest_file)
                print(f"✓ Copied: {experiment_name}")
                copied_count += 1
            except Exception as e:
                print(f"✗ Error copying {experiment_name}: {e}")
                skipped_count += 1
        else:
            print(f"⊘ Skipped: {experiment_dir.name} (no database.csv found)")
            skipped_count += 1

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files copied: {copied_count}")
    print(f"  Files skipped: {skipped_count}")
    print(f"  Destination: {dest_path.absolute()}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy database.csv files from __data__/ experiments to a destination folder"
    )
    parser.add_argument(
        "dest_dir",
        type=str,
        help="Destination directory to copy database files to"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="__data__",
        help="Source data directory (default: __data__)"
    )

    args = parser.parse_args()

    try:
        copy_databases(data_dir=args.data_dir, dest_dir=args.dest_dir)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
