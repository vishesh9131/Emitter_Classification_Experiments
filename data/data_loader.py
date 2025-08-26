#!/usr/bin/env python3
"""
Data loading and preprocessing utilities for Emitter Classification
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def load_excel_data(data_path: str, features: List[str], label_col: str = 'Name') -> pd.DataFrame:
    """
    Load data from Excel files in a directory
    
    Args:
        data_path: Path to directory containing Excel files
        features: List of feature column names
        label_col: Name of the label column
        
    Returns:
        Combined DataFrame
    """
    all_data = []
    
    # get all Excel files in the directory
    excel_files = []
    for file in os.listdir(data_path):
        if file.endswith(('.xls', '.xlsx')):
            excel_files.append(os.path.join(data_path, file))
    
    if not excel_files:
        raise ValueError(f"No Excel files found in {data_path}")
    
    # load each file
    for file_path in excel_files:
        try:
            # try different engines for Excel files
            df = None
            if file_path.endswith('.xls'):
                try:
                    df = pd.read_excel(file_path, engine='xlrd')
                except Exception as e1:
                    print(f"Failed to read {file_path} with xlrd: {e1}")
                    try:
                        df = pd.read_excel(file_path, engine='openpyxl')
                    except Exception as e2:
                        print(f"Failed to read {file_path} with openpyxl: {e2}")
                        continue
            else:  # .xlsx files
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                except Exception as e1:
                    print(f"Failed to read {file_path} with openpyxl: {e1}")
                    try:
                        df = pd.read_excel(file_path, engine='xlrd')
                    except Exception as e2:
                        print(f"Failed to read {file_path} with xlrd: {e2}")
                        continue
            
            if df is None:
                print(f"Could not read {file_path} with any engine")
                continue
            
            # filter out deleted emitters if Status column exists
            if 'Status' in df.columns:
                df = df[df['Status'] != 'DELETE_EMITTER']
            
            # select only required columns
            required_cols = [label_col] + features
            if all(col in df.columns for col in required_cols):
                df = df[required_cols]
                all_data.append(df)
                print(f"Successfully loaded {file_path}")
            else:
                print(f"Warning: Missing columns in {file_path}. Available: {list(df.columns)}")
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        raise ValueError("No valid data files found")
    
    # combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def load_csv_data(data_path: str, features: List[str], label_col: str = 'Name') -> pd.DataFrame:
    """
    Load data from CSV files in a directory
    
    Args:
        data_path: Path to directory containing CSV files
        features: List of feature column names
        label_col: Name of the label column
        
    Returns:
        Combined DataFrame
    """
    all_data = []
    
    # column name mappings for test data
    column_mappings = {
        'Azimuth(º)': 'Azimuth(deg)',
        'Elevation(º)': 'Elevation/ANT.Power.2',
        'Power(dBm)': 'Power ',
        'PW(µs)': 'PW(usec)',
        'Freq(MHz)': 'Frequency(MHz)',
        'Name': 'Name'  # assuming Name column exists or we'll handle it differently
    }
    
    # get all CSV files in the directory
    csv_files = []
    for file in os.listdir(data_path):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(data_path, file))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}")
    
    # load each file
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # create a copy of the dataframe for mapping
            df_mapped = df.copy()
            
            # map column names if they exist
            for original_col, mapped_col in column_mappings.items():
                if mapped_col in df.columns and original_col not in df.columns:
                    df_mapped[original_col] = df[mapped_col]
            
            # for test data, we might not have a Name column, so create a dummy one
            if label_col not in df_mapped.columns:
                df_mapped[label_col] = f"test_emitter_{len(all_data)}"
            
            # select only required columns (after mapping)
            required_cols = [label_col] + features
            available_cols = [col for col in required_cols if col in df_mapped.columns]
            
            if len(available_cols) >= len(features) + 1:  # at least label + features
                df_mapped = df_mapped[available_cols]
                all_data.append(df_mapped)
            else:
                print(f"Warning: Missing columns in {file_path}. Available: {list(df_mapped.columns)}")
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        raise ValueError("No valid data files found")
    
    # combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def preprocess_data(df: pd.DataFrame, 
                   features: List[str], 
                   label_col: str = 'Name',
                   scaler: Optional[RobustScaler] = None) -> Tuple[np.ndarray, np.ndarray, Dict, RobustScaler]:
    """
    Preprocess data for training
    
    Args:
        df: Input DataFrame
        features: List of feature column names
        label_col: Name of the label column
        scaler: Optional pre-fitted scaler
        
    Returns:
        Tuple of (features, labels, label_mapping, scaler)
    """
    # extract features and labels
    X = df[features].values.astype(np.float32)
    y, unique_labels = pd.factorize(df[label_col])
    
    # create label mapping
    label_mapping = {i: label for i, label in enumerate(unique_labels)}
    
    # scale features
    if scaler is None:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, label_mapping, scaler

def load_data(train_path: str, 
              test_path: str, 
              features: List[str], 
              label_col: str = 'Name') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, RobustScaler]:
    """
    Load and preprocess training and test data
    
    Args:
        train_path: Path to training data directory
        test_path: Path to test data directory
        features: List of feature column names
        label_col: Name of the label column
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, label_mapping, scaler)
    """
    # load training data
    if os.path.isdir(train_path):
        train_df = load_excel_data(train_path, features, label_col)
    else:
        raise ValueError(f"Training data path {train_path} is not a directory")
    
    # load test data
    if os.path.isdir(test_path):
        test_df = load_csv_data(test_path, features, label_col)
    else:
        raise ValueError(f"Test data path {test_path} is not a directory")
    
    # preprocess training data
    X_train, y_train, label_mapping, scaler = preprocess_data(
        train_df, features, label_col
    )
    
    # preprocess test data with same scaler
    X_test, y_test, _, _ = preprocess_data(
        test_df, features, label_col, scaler
    )
    
    return X_train, y_train, X_test, y_test, label_mapping, scaler 