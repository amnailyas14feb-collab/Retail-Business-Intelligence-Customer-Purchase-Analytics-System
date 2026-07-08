import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

def clean_retail_data(
    df: pd.DataFrame,
    drop_missing_customer: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans the Online Retail II dataset:
    - Removes duplicate rows.
    - Standardizes column names and text contents.
    - Standardizes the InvoiceDate to pandas datetime.
    - Manages returned orders (negative quantities or invoices starting with 'C') 
      by logging and separating them, returning a clean purchase-only dataset.
    - Filters out invalid prices (Price <= 0).
    - Handles missing values (drops rows with null Customer ID or Description if requested).
    
    Args:
        df (pd.DataFrame): Raw concatenated Online Retail II DataFrame.
        drop_missing_customer (bool): If True, drops rows with missing Customer IDs.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - pd.DataFrame: Cleaned purchase-only DataFrame.
            - Dict[str, Any]: Audit logs detailing records removed or altered.
    """
    initial_rows = len(df)
    logs = []
    
    # 1. Standardize column names
    cleaned_df = df.copy()
    cleaned_df.columns = [col.lstrip('\ufeff\xef\xbb\xbf').strip() for col in cleaned_df.columns]
    
    # Rename 'Customer ID' to 'Customer_ID' for code compatibility if needed, 
    # but let's keep the standard 'Customer ID' and reference it as is or clean it.
    # To keep code clean, let's standardize columns: 'Invoice', 'StockCode', 'Description', 
    # 'Quantity', 'InvoiceDate', 'Price', 'Customer_ID', 'Country'
    if 'Customer ID' in cleaned_df.columns:
        cleaned_df = cleaned_df.rename(columns={'Customer ID': 'Customer_ID'})
        
    # 2. Parse InvoiceDate
    try:
        cleaned_df['InvoiceDate'] = pd.to_datetime(cleaned_df['InvoiceDate'], errors='coerce')
        # Drop rows where date could not be parsed
        invalid_dates = cleaned_df['InvoiceDate'].isnull().sum()
        if invalid_dates > 0:
            cleaned_df = cleaned_df.dropna(subset=['InvoiceDate'])
            logs.append(f"Dropped {invalid_dates} rows with unparsable InvoiceDate.")
    except Exception as e:
        logger.error(f"Error parsing InvoiceDate: {str(e)}")
        logs.append(f"Error parsing date columns: {str(e)}")

    # 3. Handle Duplicates
    df_no_dups = cleaned_df.drop_duplicates()
    duplicates_removed = len(cleaned_df) - len(df_no_dups)
    cleaned_df = df_no_dups
    if duplicates_removed > 0:
        logs.append(f"Removed {duplicates_removed:,} duplicate transaction rows.")

    # 4. Manage Missing Values
    # Check nulls before dropping
    null_customer_count = int(cleaned_df['Customer_ID'].isnull().sum())
    null_desc_count = int(cleaned_df['Description'].isnull().sum())
    
    if null_desc_count > 0:
        cleaned_df = cleaned_df.dropna(subset=['Description'])
        logs.append(f"Dropped {null_desc_count:,} rows missing Description.")
        
    if drop_missing_customer and null_customer_count > 0:
        cleaned_df = cleaned_df.dropna(subset=['Customer_ID'])
        logs.append(f"Dropped {null_customer_count:,} rows missing Customer ID.")
        # Ensure Customer_ID is represented as integer category
        cleaned_df['Customer_ID'] = cleaned_df['Customer_ID'].astype(int)
    elif null_customer_count > 0:
        # Impute missing customers as 'Unknown Customer' or Guest
        cleaned_df['Customer_ID'] = cleaned_df['Customer_ID'].fillna(-1).astype(int)
        logs.append(f"Imputed {null_customer_count:,} missing Customer IDs with -1 (Guest Transactions).")

    # 5. Manage Returned Orders & Negative Quantities
    # Returns have quantity < 0 or Invoice starting with 'C'
    invoice_str = cleaned_df['Invoice'].astype(str).str.strip()
    is_return = invoice_str.str.startswith('C') | (cleaned_df['Quantity'] < 0)
    
    returns_df = cleaned_df[is_return]
    purchases_df = cleaned_df[~is_return]
    
    total_returned_items = abs(returns_df['Quantity'].sum())
    total_returned_value = abs((returns_df['Quantity'] * returns_df['Price']).sum())
    
    logs.append(
        f"Identified and separated {len(returns_df):,} returned/cancelled invoice lines "
        f"(Total Items Returned: {total_returned_items:,}, Est Value: ${total_returned_value:,.2f})."
    )
    
    # 6. Filter Invalid Prices on purchase-only dataset
    # We want Price > 0
    invalid_prices_df = purchases_df[purchases_df['Price'] <= 0]
    cleaned_purchases_df = purchases_df[purchases_df['Price'] > 0]
    
    if len(invalid_prices_df) > 0:
        logs.append(f"Filtered out {len(invalid_prices_df):,} purchase lines with zero or negative price (e.g. adjustments, gifts).")

    # Clean text values of string fields
    for col in ['StockCode', 'Description', 'Invoice', 'Country']:
        if col in cleaned_purchases_df.columns:
            cleaned_purchases_df[col] = cleaned_purchases_df[col].astype(str).str.strip()

    final_shape = cleaned_purchases_df.shape
    logs.append(f"Preprocessing completed. Active completed purchase rows: {final_shape[0]:,}.")
    
    summary = {
        'initial_rows': initial_rows,
        'final_rows': final_shape[0],
        'duplicates_removed': duplicates_removed,
        'null_customers_removed': null_customer_count if drop_missing_customer else 0,
        'null_descriptions_removed': null_desc_count,
        'returns_count': len(returns_df),
        'returns_items': int(total_returned_items),
        'returns_value': float(total_returned_value),
        'invalid_prices_removed': len(invalid_prices_df),
        'logs': logs
    }
    
    return cleaned_purchases_df, summary
