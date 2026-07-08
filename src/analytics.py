import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

def calculate_correlation(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Computes the Pearson correlation matrix for selected numeric columns.
    Excludes columns with zero variance to avoid mathematical undefined errors.
    
    Args:
        df (pd.DataFrame): The cleaned and engineered retail DataFrame.
        columns (List[str]): List of numeric columns to correlate.
        
    Returns:
        pd.DataFrame: Correlation matrix.
    """
    if not columns:
        return pd.DataFrame()
        
    # Filter columns to only include those in df and numeric
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    # Exclude columns with zero variance
    non_constant_cols = []
    for col in valid_cols:
        if df[col].nunique() > 1:
            non_constant_cols.append(col)
            
    if not non_constant_cols:
        logger.warning("No numeric columns with variance found to calculate correlation.")
        return pd.DataFrame()
        
    return df[non_constant_cols].corr(method='pearson')


def get_top_correlations(corr_matrix: pd.DataFrame, top_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts, de-duplicates, and ranks the strongest positive and negative correlation pairs.
    
    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.
        top_n (int): Number of top relationships to retrieve.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - pd.DataFrame: Ranked positive correlation pairs (columns: Variable 1, Variable 2, Correlation).
            - pd.DataFrame: Ranked negative correlation pairs (columns: Variable 1, Variable 2, Correlation).
    """
    if corr_matrix.empty:
        empty_df = pd.DataFrame(columns=['Variable 1', 'Variable 2', 'Correlation'])
        return empty_df, empty_df
        
    # Unstack the correlation matrix and drop self-correlations
    pairs = corr_matrix.unstack()
    pairs = pairs.drop(labels=[(col, col) for col in corr_matrix.columns])
    
    # Remove duplicate pairs (symmetric relationships, e.g. (A, B) and (B, A))
    unique_pairs = []
    seen = set()
    for (v1, v2), val in pairs.items():
        if (v2, v1) not in seen and not np.isnan(val):
            seen.add((v1, v2))
            unique_pairs.append({'Variable 1': v1, 'Variable 2': v2, 'Correlation': val})
            
    pairs_df = pd.DataFrame(unique_pairs)
    if pairs_df.empty:
        empty_df = pd.DataFrame(columns=['Variable 1', 'Variable 2', 'Correlation'])
        return empty_df, empty_df
        
    # Sort and split into positive and negative
    pos_corrs = pairs_df[pairs_df['Correlation'] > 0].sort_values(by='Correlation', ascending=False)
    neg_corrs = pairs_df[pairs_df['Correlation'] < 0].sort_values(by='Correlation', ascending=True) # closest to -1 first
    
    return pos_corrs.head(top_n).reset_index(drop=True), neg_corrs.head(top_n).reset_index(drop=True)


def get_correlation_strength(value: float) -> str:
    """Classifies correlation coefficient strength."""
    abs_val = abs(value)
    if abs_val >= 0.8:
        return "Very Strong"
    elif abs_val >= 0.6:
        return "Strong"
    elif abs_val >= 0.4:
        return "Moderate"
    elif abs_val >= 0.2:
        return "Weak"
    else:
        return "Negligible"


def get_retail_business_insight(v1: str, v2: str, corr_val: float) -> str:
    """
    Generates domain-specific customer purchase insights for Online Retail II engineered metrics.
    """
    v1_l, v2_l = v1.lower(), v2.lower()
    strength = get_correlation_strength(corr_val)
    direction = "positive" if corr_val > 0 else "negative"
    
    # Check relationships
    # 1. Quantity vs Price (Price Elasticity)
    if (v1_l == 'quantity' and v2_l == 'price') or (v2_l == 'quantity' and v1_l == 'price'):
        if corr_val < 0:
            return (
                f"We observe a {direction} correlation ({corr_val:.2f}, {strength}) between quantity and unit price. "
                "This reflects standard price elasticity of demand: as a product's price increases, customers purchase fewer units. "
                "To leverage this, look into optimizing pricing points for high-volume items to increase sales without crushing unit demand."
            )
        else:
            return (
                f"We observe a positive correlation ({corr_val:.2f}) between unit price and quantity. "
                "This represents a premium product effect where higher-priced items are bought in bulk or bundles, "
                "or indicates that specific wholesale items with higher prices are seeing bulk business orders."
            )
            
    # 2. Customer_Total_Spend vs Customer_Order_Count (Loyalty / Frequency vs Lifetime Value)
    if ('total_spend' in v1_l and 'order_count' in v2_l) or ('total_spend' in v2_l and 'order_count' in v1_l):
        return (
            f"There is a {strength} {direction} correlation ({corr_val:.2f}) between Customer Total Spend and Order Count. "
            "This suggests that transaction frequency is a massive driver of Customer Lifetime Value (CLV). "
            "Retaining customers and driving repeated purchases (via loyalty programs, targeted email campaigns, and personalized recommendation systems) "
            "is strongly linked to expanding top-line revenue."
        )
        
    # 3. Customer_Average_Order_Value vs Customer_Total_Spend (Basket Size vs Spend)
    if ('average_order_value' in v1_l and 'total_spend' in v2_l) or ('average_order_value' in v2_l and 'total_spend' in v1_l):
        return (
            f"We find a {strength} {direction} correlation ({corr_val:.2f}) between Customer Average Order Value (AOV) and Total Spend. "
            "This indicates that customer value is highly dependent on basket size. "
            "To boost average ticket size, the system suggests cross-selling, checkout product bundling, and "
            "offering free shipping thresholds (e.g., 'Free shipping on orders above $50')."
        )
        
    # 4. Items_Per_Invoice vs Invoice_Total (Cart Volume vs Cart Value)
    if ('items_per_invoice' in v1_l and 'invoice_total' in v2_l) or ('items_per_invoice' in v2_l and 'invoice_total' in v1_l):
        return (
            f"A {strength} {direction} relationship ({corr_val:.2f}) is visible between Items per Invoice and Invoice Total. "
            "Transactions with more items translate directly to larger revenues. "
            "This supports marketing strategies that encourage multi-item purchases (e.g., 'Buy 3, get 10% off' or wholesale bundle promotions)."
        )
        
    # 5. Customer_Order_Count vs Customer_Average_Order_Value (Purchase Frequency vs Basket Size)
    if ('order_count' in v1_l and 'average_order_value' in v2_l) or ('order_count' in v2_l and 'average_order_value' in v1_l):
        if corr_val < 0:
            return (
                f"We note a negative relationship ({corr_val:.2f}, {strength}) between Order Count and Average Order Value. "
                "This indicates that customers who purchase frequently tend to buy smaller baskets per visit, "
                "whereas customers who purchase rarely make large bulk orders. "
                "Marketing should split customers into 'frequent shoppers' (nurture with loyalty rewards) and 'bulk buyers' (encourage with quantity discounts)."
            )
        else:
            return (
                f"We note a positive correlation ({corr_val:.2f}) between order frequency and average order value. "
                "This indicates your VIP customers are compounding: they shop frequently AND place large orders. "
                "These high-value buyers are the core pillars of business growth and should receive white-glove loyalty services."
            )
            
    # 6. Sales vs Quantity or Price
    if (v1_l == 'sales' and v2_l in ['quantity', 'price']) or (v2_l == 'sales' and v1_l in ['quantity', 'price']):
        related = 'quantity' if 'quantity' in [v1_l, v2_l] else 'price'
        return (
            f"There is a {strength} {direction} correlation ({corr_val:.2f}) between Sales (Quantity x Price) and product {related}. "
            f"This indicates that fluctuations in product {related} heavily influence overall transactional revenue. "
            f"If it correlates stronger with quantity, the business is volume-driven; if with price, it is margin-driven."
        )

    # 7. Product_Total_Sales vs Product_Total_Quantity
    if ('product_total_sales' in v1_l and 'product_total_quantity' in v2_l) or ('product_total_sales' in v2_l and 'product_total_quantity' in v1_l):
        return (
            f"Product Total Sales and Quantity exhibit a {strength} {direction} correlation ({corr_val:.2f}). "
            "This highlights that catalog demand volume is the primary driver of product revenue. "
            "Focusing inventory management on these fast-moving goods ensures high stock rotation and prevents capital lockup in slow items."
        )
        
    # Generic fallback
    trend = "tends to increase" if corr_val > 0 else "tends to decrease"
    return (
        f"There is a {strength} {direction} correlation ({corr_val:.2f}) between '{v1}' and '{v2}'. "
        f"This indicates that as '{v1}' increases, '{v2}' {trend}. "
        "In customer purchase analytics, this relationship can help segment buyer groups, optimize inventory flow, "
        "and draft marketing strategies tailored to customer behaviors."
    )


def generate_business_insights(top_pos: pd.DataFrame, top_neg: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Compiles list of business insights for top positive and negative correlations.
    
    Args:
        top_pos (pd.DataFrame): Top positive correlations.
        top_neg (pd.DataFrame): Top negative correlations.
        
    Returns:
        List[Dict[str, Any]]: List of insights dictionaries.
    """
    insights = []
    
    # Process positive correlations
    for _, row in top_pos.iterrows():
        v1, v2, val = row['Variable 1'], row['Variable 2'], row['Correlation']
        insights.append({
            'var1': v1,
            'var2': v2,
            'correlation': val,
            'type': 'Positive',
            'strength': get_correlation_strength(val),
            'text': get_retail_business_insight(v1, v2, val)
        })
        
    # Process negative correlations
    for _, row in top_neg.iterrows():
        v1, v2, val = row['Variable 1'], row['Variable 2'], row['Correlation']
        insights.append({
            'var1': v1,
            'var2': v2,
            'correlation': val,
            'type': 'Negative',
            'strength': get_correlation_strength(val),
            'text': get_retail_business_insight(v1, v2, val)
        })
        
    return insights


def get_causation_explanation() -> Dict[str, str]:
    """
    Returns a comprehensive explanation of correlation vs. causation in retail analytics.
    """
    return {
        'headline': "Correlation Does Not Imply Causation",
        'core_concept': (
            "A correlation coefficient measures the strength and direction of a linear relationship between two variables. "
            "However, observing a correlation between two variables does not mean that one causes the change in the other."
        ),
        'retail_reasons': (
            "1. **Confounding Variables (Third-Cause Fallacy):** Two variables might correlate because they are both influenced by an unobserved "
            "third factor. For instance, online sales and physical store sales might both surge during December. Online sales did not cause "
            "in-store sales; both were driven by holiday shopping season demand.\n\n"
            "2. **Reverse Causality:** We might see a correlation between high customer satisfaction and high total spend. "
            "Do happy customers spend more (Satisfaction -> Spend), or does spending more grant them VIP status with discounts "
            "and loyalty perks that make them happier (Spend -> Satisfaction)?\n\n"
            "3. **Spurious Correlations (Coincidence):** With large datasets and many variables, some correlations will appear purely by chance "
            "without any logical business connection."
        ),
        'actionable_advice': (
            "To establish causality, businesses should run randomized controlled trials (A/B testing, like testing different discount "
            "rates on randomly selected groups) or utilize advanced econometric models (like regression discontinuity or instrumental variables) "
            "rather than acting blindly on a simple correlation matrix."
        )
    }


def compile_dataset_insights(filtered_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compiles high-level business analytics stats (best country, product, month, AOV, and trends)
    from the filtered dataset.
    """
    if filtered_df.empty:
        return {
            'top_country': "N/A",
            'top_country_revenue': 0.0,
            'best_selling_product': "N/A",
            'best_selling_units': 0,
            'highest_rev_month': "N/A",
            'highest_rev_month_revenue': 0.0,
            'aov': 0.0,
            'trends': ["No data available to compile trends."]
        }

    # 1. Top performing country
    country_sales = filtered_df.groupby('Country')['Sales'].sum()
    top_country = country_sales.idxmax() if not country_sales.empty else "N/A"
    top_country_revenue = country_sales.max() if not country_sales.empty else 0.0
    
    # 2. Best selling product (by quantity)
    # Filter out any non-descriptive descriptions if they are null
    df_desc = filtered_df.dropna(subset=['Description'])
    prod_qty = df_desc.groupby('Description')['Quantity'].sum()
    best_selling_product = prod_qty.idxmax() if not prod_qty.empty else "N/A"
    best_selling_units = prod_qty.max() if not prod_qty.empty else 0
    
    # 3. Highest revenue month
    month_sales = filtered_df.groupby('Month_Name')['Sales'].sum()
    highest_rev_month = month_sales.idxmax() if not month_sales.empty else "N/A"
    highest_rev_month_revenue = month_sales.max() if not month_sales.empty else 0.0
    
    # 4. Average order value (AOV)
    unique_invoices = filtered_df['Invoice'].nunique()
    total_sales = filtered_df['Sales'].sum()
    aov = total_sales / unique_invoices if unique_invoices > 0 else 0.0
    
    # 5. Trends & Anomalies
    trends = []
    
    # Check weekend sales percentage
    if 'Weekend' in filtered_df.columns:
        weekend_sales = filtered_df[filtered_df['Weekend'] == True]['Sales'].sum()
        weekday_sales = filtered_df[filtered_df['Weekend'] == False]['Sales'].sum()
        total_sales_we = weekend_sales + weekday_sales
        weekend_pct = (weekend_sales / total_sales_we * 100) if total_sales_we > 0 else 0.0
        
        if weekend_pct > 15:
            trends.append(f"Weekend purchases constitute a notable {weekend_pct:.1f}% of total sales. Targeted weekend promotions could boost revenue.")
        else:
            trends.append(f"Sales are dominated by weekday transactions (Weekend transactions contribute only {weekend_pct:.1f}%). Focus B2B marketing during standard business hours.")
            
    # Check top country concentration
    if not country_sales.empty:
        total_revenue = country_sales.sum()
        top_country_pct = (top_country_revenue / total_revenue * 100) if total_revenue > 0 else 0.0
        if top_country_pct > 70:
            trends.append(f"Revenue is heavily concentrated in the top country ({top_country} accounts for {top_country_pct:.1f}% of total sales), representing a geographic concentration risk.")
        else:
            trends.append(f"Revenue is geographically diversified, with the top country ({top_country}) representing {top_country_pct:.1f}% of total sales.")
            
    # Check for unit price outlier anomaly
    max_price = filtered_df['Price'].max()
    avg_price = filtered_df['Price'].mean()
    if max_price > avg_price * 50:
        trends.append(f"High price variance detected: Maximum unit price (${max_price:,.2f}) is significantly higher than the average price (${avg_price:,.2f}), suggesting high-value items or pricing anomalies.")

    return {
        'top_country': top_country,
        'top_country_revenue': float(top_country_revenue),
        'best_selling_product': best_selling_product,
        'best_selling_units': int(best_selling_units),
        'highest_rev_month': highest_rev_month,
        'highest_rev_month_revenue': float(highest_rev_month_revenue),
        'aov': float(aov),
        'trends': trends
    }
