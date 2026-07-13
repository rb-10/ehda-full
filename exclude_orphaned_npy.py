import os
import shutil
import sqlite3
from pathlib import Path

def main():
    base_dir = Path("data")
    db_path = base_dir / "data.db"
    raw_dir = base_dir / "raw_waveforms"
    excluded_dir = base_dir / "excluded_raw_waveforms"

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    if not raw_dir.exists():
        print(f"Raw waveforms directory not found at {raw_dir}")
        return

    excluded_dir.mkdir(parents=True, exist_ok=True)

    # Connect to db and get all valid raw_data_file names
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='measurements'")
    if not cursor.fetchone():
        print("Table 'measurements' not found in database.")
        conn.close()
        return

    cursor.execute("SELECT raw_data_file FROM measurements WHERE raw_data_file IS NOT NULL")
    valid_files = set(row[0] for row in cursor.fetchall())
    conn.close()

    print(f"Found {len(valid_files)} file references in the database.")

    # Loop through the raw directory
    moved_count = 0
    total_count = 0
    
    for filename in os.listdir(raw_dir):
        if filename.endswith(".npy"):
            total_count += 1
            if filename not in valid_files:
                src_path = raw_dir / filename
                dest_path = excluded_dir / filename
                shutil.move(str(src_path), str(dest_path))
                moved_count += 1

    print(f"\nCleanup complete.")
    print(f"Total .npy files checked: {total_count}")
    print(f"Files excluded (moved to {excluded_dir}): {moved_count}")

if __name__ == "__main__":
    main()
