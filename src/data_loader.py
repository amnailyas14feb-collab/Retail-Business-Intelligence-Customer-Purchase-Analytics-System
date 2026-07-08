import os
import pandas as pd
import logging
from typing import Tuple, Optional

# Set up logging for professional execution tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_merge_datasets(
    file1_path: str = "data/Retail 2009-10.csv",
    file2_path: str = "data/Retail 2010-11.csv",
    merged_output_path: str = "data/online_retail_II_merged.csv"
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Checks for a pre-merged dataset. If missing, loads the two Online Retail II 
    original files, merges them, and saves the result to a single consolidated file
    while preserving the original files.
    
    Args:
        file1_path (str): Path to the first year dataset (2009-10).
        file2_path (str): Path to the second year dataset (2010-11).
        merged_output_path (str): Destination path for the consolidated merged CSV.
        
    Returns:
        Tuple[Optional[pd.DataFrame], str]:
            - pd.DataFrame: The consolidated DataFrame, or None if failed.
            - str: Summary execution message or error details.
    """
    # Create target directories if they do not exist
    os.makedirs(os.path.dirname(merged_output_path), exist_ok=True)
    
    # 1. If consolidated file already exists, load and return it
    if os.path.exists(merged_output_path):
        logger.info(f"Consolidated dataset found at {merged_output_path}. Loading...")
        try:
            df = pd.read_csv(merged_output_path, low_memory=False)
            return df, f"Successfully loaded cached merged dataset with {df.shape[0]:,} rows and {df.shape[1]} columns."
        except Exception as e:
            logger.error(f"Error loading consolidated dataset: {str(e)}. Re-merging raw files...")
            
    # Fallback to root directory if not found in data/
    if not os.path.exists(file1_path):
        root_fallback = os.path.basename(file1_path)
        if os.path.exists(root_fallback):
            file1_path = root_fallback
            
    if not os.path.exists(file2_path):
        root_fallback = os.path.basename(file2_path)
        if os.path.exists(root_fallback):
            file2_path = root_fallback

    # 2. Verify original files exist
    if not os.path.exists(file1_path):
        return None, f"Error: Original dataset '{file1_path}' not found."
    if not os.path.exists(file2_path):
        return None, f"Error: Original dataset '{file2_path}' not found."
        
    # 3. Read and merge raw datasets
    logger.info("Merging original Retail datasets. This may take a few seconds...")
    try:
        logger.info(f"Loading raw file: {file1_path}")
        df1 = pd.read_csv(file1_path, encoding='unicode_escape', low_memory=False)
        logger.info(f"Loaded {len(df1):,} rows from {file1_path}")
        
        logger.info(f"Loading raw file: {file2_path}")
        df2 = pd.read_csv(file2_path, encoding='unicode_escape', low_memory=False)
        logger.info(f"Loaded {len(df2):,} rows from {file2_path}")
        
        # Concatenate rows
        logger.info("Concatenating datasets...")
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Save consolidated dataset
        logger.info(f"Saving merged dataset to {merged_output_path}...")
        merged_df.to_csv(merged_output_path, index=False)
        logger.info("Consolidated dataset saved.")
        
        return merged_df, f"Successfully merged raw files. Consolidated dataset contains {merged_df.shape[0]:,} rows and {merged_df.shape[1]} columns."
        
    except Exception as e:
        logger.exception("Error merging datasets:")
        return None, f"Error merging original files: {str(e)}"
