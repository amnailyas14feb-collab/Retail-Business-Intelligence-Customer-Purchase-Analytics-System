import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs high-performance vectorised feature engineering on the retail dataset.
    
    Generates:
    - Transactional: Sales (Quantity x Price)
    - Time-based: Year, Month, Month Name, Quarter, Day, Day Name, Hour, Week Number, Weekend
    - Invoice aggregations: Invoice Total, Items Per Invoice
    - Customer profile: Customer Total Spend, Customer Order Count, Customer Average Order Value
    - Product performance: Product Total Sales, Product Total Quantity
    - Country stats: Country Total Sales, Country Order Count
    - Segments: Order Size (Small, Medium, Large, Bulk)
    
    Args:
        df (pd.DataFrame): Cleaned purchase-only DataFrame.
        
    Returns:
        pd.DataFrame: Feature-engineered DataFrame.
    """
    logger.info("Starting feature engineering on cleaned dataset...")
    fe_df = df.copy()
    
    # 1. Transactional: Sales
    fe_df['Sales'] = fe_df['Quantity'] * fe_df['Price']
    
    # 2. Time-based features
    logger.info("Extracting time-based attributes...")
    dates = fe_df['InvoiceDate'].dt
    fe_df['Year'] = dates.year
    fe_df['Month'] = dates.month
    fe_df['Month_Name'] = dates.strftime('%B')
    fe_df['Quarter'] = dates.quarter
    fe_df['Day'] = dates.day
    fe_df['Day_Name'] = dates.strftime('%A')
    fe_df['Hour'] = dates.hour
    fe_df['Week_Number'] = dates.isocalendar().week.astype(int)
    fe_df['Weekend'] = (dates.dayofweek >= 5).astype(int)
    
    # 3. Invoice-level Aggregations
    logger.info("Calculating invoice-level aggregates...")
    inv_groups = fe_df.groupby('Invoice')
    invoice_totals = inv_groups['Sales'].sum().rename('Invoice_Total')
    invoice_items = inv_groups['Quantity'].sum().rename('Items_Per_Invoice')
    
    # Join back to transaction level
    fe_df = fe_df.join(invoice_totals, on='Invoice')
    fe_df = fe_df.join(invoice_items, on='Invoice')
    
    # 4. Customer-level Profile Attributes
    logger.info("Calculating customer profile metrics...")
    cust_groups = fe_df.groupby('Customer_ID')
    cust_spends = cust_groups['Sales'].sum().rename('Customer_Total_Spend')
    cust_orders = cust_groups['Invoice'].nunique().rename('Customer_Order_Count')
    
    # Join back
    fe_df = fe_df.join(cust_spends, on='Customer_ID')
    fe_df = fe_df.join(cust_orders, on='Customer_ID')
    
    # Customer Average Order Value (AOV)
    fe_df['Customer_Average_Order_Value'] = fe_df['Customer_Total_Spend'] / fe_df['Customer_Order_Count']
    
    # 5. Product-level Performance
    logger.info("Calculating product total metrics...")
    prod_groups = fe_df.groupby('StockCode')
    prod_sales = prod_groups['Sales'].sum().rename('Product_Total_Sales')
    prod_qtys = prod_groups['Quantity'].sum().rename('Product_Total_Quantity')
    
    # Join back
    fe_df = fe_df.join(prod_sales, on='StockCode')
    fe_df = fe_df.join(prod_qtys, on='StockCode')
    
    # 6. Country-level Statistics
    logger.info("Calculating country-level statistics...")
    country_groups = fe_df.groupby('Country')
    country_sales = country_groups['Sales'].sum().rename('Country_Total_Sales')
    country_orders = country_groups['Invoice'].nunique().rename('Country_Order_Count')
    
    # Join back
    fe_df = fe_df.join(country_sales, on='Country')
    fe_df = fe_df.join(country_orders, on='Country')
    
    # 7. Order Size Categories (np.select vectorised boundary conditions)
    logger.info("Segmenting order size categories...")
    conditions = [
        fe_df['Items_Per_Invoice'] <= 10,
        (fe_df['Items_Per_Invoice'] > 10) & (fe_df['Items_Per_Invoice'] <= 50),
        (fe_df['Items_Per_Invoice'] > 50) & (fe_df['Items_Per_Invoice'] <= 200)
    ]
    choices = ['Small', 'Medium', 'Large']
    fe_df['Order_Size'] = np.select(conditions, choices, default='Bulk')
    
    logger.info(f"Feature engineering complete. Total features: {fe_df.shape[1]}")
    return fe_df
