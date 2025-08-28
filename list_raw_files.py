#!/usr/bin/env python3
"""
Script to list available files in the raw dataset directory
Helps users choose which files to use for training
"""

import os
import pandas as pd

def list_raw_files(raw_dataset_path='raw_dt/rawdataset/'):
    """List all available files in the raw dataset with basic info"""
    
    if not os.path.exists(raw_dataset_path):
        print(f"Raw dataset path {raw_dataset_path} does not exist!")
        return
    
    print(f"Available files in {raw_dataset_path}:")
    print("=" * 80)
    
    files_info = []
    
    for filename in os.listdir(raw_dataset_path):
        if filename.endswith(('.csv', '.xls', '.xlsx')):
            file_path = os.path.join(raw_dataset_path, filename)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            
            # Try to get basic info about the file
            try:
                if filename.endswith('.csv'):
                    # Read first few lines to get column info
                    df_sample = pd.read_csv(file_path, nrows=5)
                    num_rows = sum(1 for line in open(file_path)) - 1  # Approximate
                else:
                    df_sample = pd.read_excel(file_path, nrows=5)
                    num_rows = len(pd.read_excel(file_path))
                
                columns = list(df_sample.columns)
                
                # Check if it has SourceIndex (useful for labeling)
                has_source_index = 'SourceIndex' in columns
                
                files_info.append({
                    'filename': filename,
                    'size_mb': file_size,
                    'rows': num_rows,
                    'columns': len(columns),
                    'has_source_index': has_source_index,
                    'sample_columns': columns[:5]  # First 5 columns
                })
                
            except Exception as e:
                files_info.append({
                    'filename': filename,
                    'size_mb': file_size,
                    'rows': 'Error reading',
                    'columns': 'Error reading',
                    'has_source_index': False,
                    'sample_columns': []
                })
    
    # Sort by file size
    files_info.sort(key=lambda x: x['size_mb'], reverse=True)
    
    for info in files_info:
        print(f"File: {info['filename']}")
        print(f"  Size: {info['size_mb']:.1f} MB")
        print(f"  Rows: {info['rows']}")
        print(f"  Columns: {info['columns']}")
        print(f"  Has SourceIndex: {info['has_source_index']}")
        if info['sample_columns']:
            print(f"  Sample columns: {', '.join(info['sample_columns'])}")
        print()
    
    print("=" * 80)
    print("Example usage:")
    print("python train.py --model emitter_encoder --loss triplet --epochs 10 \\")
    print("  --train_files raw_dt/rawdataset/Raw_1.csv raw_dt/rawdataset/s6_1.csv \\")
    print("  --test_files raw_dt/rawdataset/s6_2.csv raw_dt/rawdataset/s6_3.csv \\")
    print("  --output_dir results_custom/")

if __name__ == "__main__":
    list_raw_files() 