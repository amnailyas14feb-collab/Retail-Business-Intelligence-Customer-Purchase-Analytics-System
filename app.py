import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging

from typing import Any, Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import modules from src
from src.data_loader import load_and_merge_datasets
from src.preprocessor import clean_retail_data
from src.feature_engineer import engineer_features
from src.analytics import (
    calculate_correlation,
    get_top_correlations,
    generate_business_insights,
    get_causation_explanation,
    get_correlation_strength,
    compile_dataset_insights
)
from src.visualization import (
    plot_heatmap,
    plot_scatter,
    plot_pairplot,
    save_plot_to_png
)
from src.reporting import (
    save_correlation_csv,
    generate_text_report,
    generate_pdf_report,
    md_to_html_bold
)

# Page configuration
st.set_page_config(
    page_title="Retail BI & Customer Purchase Analytics System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Visual layout adjustments (Executive theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header layout */
    .bi-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 2.2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .bi-header h1 {
        font-weight: 700;
        margin: 0;
        font-size: 2.1rem;
        letter-spacing: -0.025em;
    }
    .bi-header p {
        margin: 0.4rem 0 0 0;
        font-size: 1.05rem;
        opacity: 0.85;
        font-weight: 300;
    }
    
    /* KPI grids */
    .kpi-container {
        display: flex;
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        flex: 1;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }
    
    /* Caution callout */
    .caution-card {
        background-color: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 5px solid #d97706;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 2rem;
    }
    .caution-title {
        font-size: 1rem;
        font-weight: 700;
        color: #b45309;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .caution-text {
        font-size: 0.9rem;
        color: #92400e;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- MAIN TITLE -----------------
st.markdown("""
<div class="bi-header">
    <h1>Retail Business Intelligence & Customer Purchase Analytics System</h1>
    <p>Consolidated Online Retail II Multi-Year Cohort Analysis, Pearson Correlation Mining, and Business Intelligence Reporting.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- DATA PIPELINE CACHING -----------------
@st.cache_data(show_spinner="Compiling data: Merging files, cleaning data, and engineering features (takes ~30s on first load)...")
def load_and_prepare_retail_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Coordinates the ingestion, cleaning, and engineering modules. Caches results 
    to CSV and JSON locally for instant subsequent loads.
    """
    processed_path = "data/online_retail_II_processed.csv"
    summary_path = "data/cleaning_summary.json"
    
    # 1. Try to load cached pre-engineered files
    if os.path.exists(processed_path) and os.path.exists(summary_path):
        logger.info("Found cached processed dataset. Loading...")
        try:
            df = pd.read_csv(processed_path, parse_dates=['InvoiceDate'], low_memory=False)
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            return df, summary
        except Exception as e:
            logger.error(f"Error loading cached files: {str(e)}. Re-running pipeline...")
            
    # 2. Ingest original datasets and merge
    df_raw, msg = load_and_merge_datasets(
        file1_path="data/Retail 2009-10.csv",
        file2_path="data/Retail 2010-11.csv",
        merged_output_path="data/online_retail_II_merged.csv"
    )
    if df_raw is None:
        raise FileNotFoundError(
            "Original CSV files 'Retail 2009-10.csv' and 'Retail 2010-11.csv' are missing. "
            "Please ensure they are present in the project root workspace."
        )
        
    # 3. Clean
    logger.info("Starting cleaning phase...")
    cleaned_df, summary = clean_retail_data(df_raw, drop_missing_customer=True)
    
    # 4. Feature Engineer
    logger.info("Starting feature engineering phase...")
    engineered_df = engineer_features(cleaned_df)
    
    # 5. Save cached versions to disk
    logger.info(f"Caching processed data to: {processed_path}")
    engineered_df.to_csv(processed_path, index=False)
    
    logger.info(f"Caching summary audit logs to: {summary_path}")
    with open(summary_path, 'w') as f:
        json.dump(summary, f)
        
    return engineered_df, summary

# Run pipeline
try:
    df, cleaning_summary = load_and_prepare_retail_pipeline()
    pipeline_ok = True
except Exception as e:
    st.error(f"Failed to load or process dataset files: {str(e)}")
    st.info("💡 Please verify that 'Retail 2009-10.csv' and 'Retail 2010-11.csv' are saved in the project directory.")
    pipeline_ok = False

if pipeline_ok:
    # ----------------- SIDEBAR INTERACTIVE FILTERS -----------------
    st.sidebar.markdown("<h3 style='margin:0;font-weight:700;color:#1e293b;'>🔍 Interactive Filters</h3>", unsafe_allow_html=True)
    
    # Filter 1: Country Multiselect
    unique_countries = sorted(df['Country'].unique().tolist())
    # Default select All, or United Kingdom as it's the primary market
    uk_index = unique_countries.index('United Kingdom') if 'United Kingdom' in unique_countries else 0
    selected_countries = st.sidebar.multiselect(
        "Select Countries:",
        options=unique_countries,
        default=[unique_countries[uk_index]] if len(unique_countries) > uk_index else unique_countries
    )
    
    # Filter 2: Month name filter
    months_order = [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ]
    unique_months = [m for m in months_order if m in df['Month_Name'].unique()]
    selected_months = st.sidebar.multiselect(
        "Select Months:",
        options=unique_months,
        default=unique_months
    )
    
    # Filter 3: Product Search Input (Performance safe compared to huge dropdowns)
    st.sidebar.markdown("#### Product Catalog Search")
    product_query = st.sidebar.text_input(
        "Search by SKU Description (e.g. HEART, BAG):",
        value=""
    ).strip().upper()
    
    # Filter 4: Minimum customer order threshold
    min_orders = st.sidebar.slider(
        "Min Orders per Customer:",
        min_value=1,
        max_value=int(df['Customer_Order_Count'].max()),
        value=1
    )
    
    # Apply Filters to dataset
    filtered_df = df.copy()
    
    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
    if selected_months:
        filtered_df = filtered_df[filtered_df['Month_Name'].isin(selected_months)]
    if product_query:
        filtered_df = filtered_df[filtered_df['Description'].str.upper().str.contains(product_query, na=False)]
    if min_orders > 1:
        filtered_df = filtered_df[filtered_df['Customer_Order_Count'] >= min_orders]
        
    # Check if empty
    if filtered_df.empty:
        st.sidebar.error("Warning: Current filters returned 0 rows. Please broaden selections.")
        active_analysis = False
    else:
        active_analysis = True
        
    if active_analysis:
        # ----------------- KPI INDICATORS -----------------
        total_sales = filtered_df['Sales'].sum()
        total_qty = filtered_df['Quantity'].sum()
        unique_invoices = filtered_df['Invoice'].nunique()
        unique_customers = filtered_df['Customer_ID'].nunique()
        avg_basket_value = total_sales / unique_invoices if unique_invoices > 0 else 0.0
        
        # Automatically detect all meaningful numeric features after preprocessing, excluding identifier columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Customer_ID', 'Customer ID', 'Invoice', 'StockCode', 'InvoiceDate']
        corr_features = [col for col in numeric_cols if col not in exclude_cols]
        
        # Compile dataset business insights dynamically based on active filters
        dataset_insights = compile_dataset_insights(filtered_df)
        
        # Display KPI Grid
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-label">Active Total Revenue</div>
                <div class="kpi-value">${total_sales:,.2f}</div>
                <div class="kpi-sub">Total revenue of active selections</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Unique Customers</div>
                <div class="kpi-value">{unique_customers:,}</div>
                <div class="kpi-sub">Count of unique Customer IDs</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Invoices Issued</div>
                <div class="kpi-value">{unique_invoices:,}</div>
                <div class="kpi-sub">Transactions containing {total_qty:,} items</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Average Invoice Value</div>
                <div class="kpi-value">${avg_basket_value:,.2f}</div>
                <div class="kpi-sub">AOV per purchase transaction</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ----------------- EXECUTIVE SUMMARY CARD -----------------
        # Calculate strongest positive and negative correlations globally for summary
        corr_mat_summary = calculate_correlation(filtered_df, corr_features)
        pos_c_sum, neg_c_sum = get_top_correlations(corr_mat_summary, top_n=1)
        
        strong_pos_text = "N/A"
        if not pos_c_sum.empty:
            r_pos = pos_c_sum.iloc[0]
            v1_pos = r_pos['Variable 1'].replace('_', ' ')
            v2_pos = r_pos['Variable 2'].replace('_', ' ')
            strong_pos_text = f"<b>{v1_pos}</b> & <b>{v2_pos}</b> (r = <b>{r_pos['Correlation']:.4f}</b>)"
            
        strong_neg_text = "N/A"
        if not neg_c_sum.empty:
            r_neg = neg_c_sum.iloc[0]
            v1_neg = r_neg['Variable 1'].replace('_', ' ')
            v2_neg = r_neg['Variable 2'].replace('_', ' ')
            strong_neg_text = f"<b>{v1_neg}</b> & <b>{v2_neg}</b> (r = <b>{r_neg['Correlation']:.4f}</b>)"

        # Get anomaly note
        anomaly_note = "No pricing, temporal, or geographic anomalies detected under current filters."
        if dataset_insights['trends']:
            anomaly_note = dataset_insights['trends'][0]

        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 6px solid #1e3a8a; border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);">
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.8rem;">
                <span style="font-size: 1.4rem;">📊</span>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #0f172a;">Executive Analytics Brief</h3>
            </div>
            <p style="margin: 0 0 1rem 0; font-size: 0.85rem; color: #64748b; font-style: italic;">
                ⚡ These findings are generated dynamically based on your active filter selections.
            </p>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.925rem; color: #334155; line-height: 1.6;">
                <li style="margin-bottom: 0.5rem;">💵 <b>Financial Scope:</b> Active total revenue is <b>${total_sales:,.2f}</b> compiled from <b>{unique_invoices:,}</b> completed transactions (Average Invoice Value: <b>${dataset_insights['aov']:,.2f}</b>).</li>
                <li style="margin-bottom: 0.5rem;">🌍 <b>Regional Leader:</b> <b>{dataset_insights['top_country']}</b> is the highest-performing market, generating <b>${dataset_insights['top_country_revenue']:,.2f}</b> in sales volume.</li>
                <li style="margin-bottom: 0.5rem;">📦 <b>Product Demand:</b> The best-selling catalog item is <b>{dataset_insights['best_selling_product']}</b>, shifting <b>{dataset_insights['best_selling_units']:,}</b> units.</li>
                <li style="margin-bottom: 0.5rem;">📅 <b>Peak Window:</b> Sales reached a peak in <b>{dataset_insights['highest_rev_month']}</b> generating <b>${dataset_insights['highest_rev_month_revenue']:,.2f}</b> in revenue.</li>
                <li style="margin-bottom: 0.5rem;">📈 <b>Strongest Positive Relationship:</b> {strong_pos_text}</li>
                <li style="margin-bottom: 0.5rem;">📉 <b>Strongest Negative Relationship:</b> {strong_neg_text}</li>
                <li style="margin-bottom: 0rem;">⚠️ <b>Operational Anomaly:</b> {anomaly_note}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Define Exports paths
        EXPORTS_DIR = "exports"
        os.makedirs(EXPORTS_DIR, exist_ok=True)
        
        # ----------------- TAB STRUCTURE -----------------
        tab_ov, tab_heat, tab_scat, tab_ins, tab_rep = st.tabs([
            "📋 Ingestion Overview",
            "🌡️ Pearson Correlation",
            "📈 Scatter Relationships",
            "💡 Purchase Insights",
            "📄 PDF Executive Exporter"
        ])
        
        # --- TAB 1: INGESTION OVERVIEW & AUDIT LOGS ---
        with tab_ov:
            col_l, col_r = st.columns([2, 1])
            
            with col_l:
                st.markdown("### 📋 Active Transactions Data Preview")
                st.dataframe(filtered_df.head(100), use_container_width=True)
                st.caption(f"Displaying first 100 rows out of {len(filtered_df):,} matching filtered records.")
                
                # Product sales ranking table
                st.markdown("### 🔝 Top 10 Best-Selling SKU Items")
                top_products = filtered_df.groupby('Description').agg(
                    Units_Sold=('Quantity', 'sum'),
                    Total_Revenue=('Sales', 'sum'),
                    Avg_Price=('Price', 'mean')
                ).sort_values(by='Total_Revenue', ascending=False).head(10)
                st.dataframe(top_products.style.format({'Total_Revenue': '${:,.2f}', 'Avg_Price': '${:,.2f}'}), use_container_width=True)
                
            with col_r:
                st.markdown("### 🛠️ Ingestion & Cleaning Audit")
                st.markdown(f"**Duplicates Removed:** `{cleaning_summary.get('duplicates_removed', 0):,}` lines")
                st.markdown(f"**Missing Customer IDs Removed:** `{cleaning_summary.get('null_customers_removed', 0):,}` rows")
                st.markdown(f"**Invalid Prices Filtered (Price <= 0):** `{cleaning_summary.get('invalid_prices_removed', 0):,}` rows")
                
                st.markdown("#### 🚨 Returned Orders Audited")
                st.markdown(f"- **Total Return Lines separated:** `{cleaning_summary.get('returns_count', 0):,}` rows")
                st.markdown(f"- **Total Return Volume:** `-{cleaning_summary.get('returns_items', 0):,}` units")
                st.markdown(f"- **Estimated Return Cost:** `-${cleaning_summary.get('returns_value', 0.0):,.2f}`")
                
                st.markdown("#### 🪵 Data Transformation Logs")
                for log_line in cleaning_summary['logs']:
                    st.text(log_line)
                    
        # --- TAB 2: CORRELATION HEATMAP ---
        with tab_heat:
            st.markdown("### 🌡️ Bivariate Correlation Matrix (Upper masked)")
            
            # Expander for heatmap customizations
            with st.expander("🎨 Visualization Tweaks"):
                col_hp1, col_hp2, col_hp3 = st.columns(3)
                with col_hp1:
                    heatmap_theme = st.selectbox(
                        "Divergent Colormap Theme:",
                        ["coolwarm", "RdBu_r", "vlag", "rocket", "Spectral"],
                        index=0
                    )
                with col_hp2:
                    show_numbers = st.checkbox("Annotate Cells with Values", value=True)
                with col_hp3:
                    h_font_size = st.slider("Cell Text Font Size:", min_value=6, max_value=12, value=9)
            
            selected_corr_vars = st.multiselect(
                "Select variables for the correlation matrix:",
                options=corr_features,
                default=corr_features
            )
            
            if len(selected_corr_vars) < 2:
                st.warning("Select at least 2 numerical variables to map correlations.")
            else:
                corr_matrix = calculate_correlation(filtered_df, selected_corr_vars)
                
                if corr_matrix.empty:
                    st.error("Correlation matrix calculation failed. Ensure columns contain variant data.")
                else:
                    # Highlight strongest positive and negative correlations in KPI cards
                    pos_corrs_kpi, neg_corrs_kpi = get_top_correlations(corr_matrix, top_n=1)
                    
                    col_kpi1, col_kpi2 = st.columns(2)
                    with col_kpi1:
                        if not pos_corrs_kpi.empty:
                            row_pos = pos_corrs_kpi.iloc[0]
                            v1_pos = row_pos['Variable 1'].replace('_', ' ')
                            v2_pos = row_pos['Variable 2'].replace('_', ' ')
                            r_pos = row_pos['Correlation']
                            st.markdown(f"""
                            <div class="kpi-card" style="border-left: 5px solid #10b981; padding: 1.1rem; background-color: #f0fdf4;">
                                <div class="kpi-label" style="color: #15803d; font-size: 0.75rem;">🔥 Strongest Positive Relationship</div>
                                <div class="kpi-value" style="font-size: 1.15rem; margin: 0.2rem 0; font-weight: 700; color: #1e293b;">{v1_pos} & {v2_pos}</div>
                                <div class="kpi-sub" style="color: #166534; font-weight: 600; font-size: 0.85rem;">Pearson r: {r_pos:.4f} ({get_correlation_strength(r_pos)})</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("No positive correlation found.")
                    with col_kpi2:
                        if not neg_corrs_kpi.empty:
                            row_neg = neg_corrs_kpi.iloc[0]
                            v1_neg = row_neg['Variable 1'].replace('_', ' ')
                            v2_neg = row_neg['Variable 2'].replace('_', ' ')
                            r_neg = row_neg['Correlation']
                            st.markdown(f"""
                            <div class="kpi-card" style="border-left: 5px solid #ef4444; padding: 1.1rem; background-color: #fef2f2;">
                                <div class="kpi-label" style="color: #b91c1c; font-size: 0.75rem;">❄️ Strongest Negative Relationship</div>
                                <div class="kpi-value" style="font-size: 1.15rem; margin: 0.2rem 0; font-weight: 700; color: #1e293b;">{v1_neg} & {v2_neg}</div>
                                <div class="kpi-sub" style="color: #991b1b; font-weight: 600; font-size: 0.85rem;">Pearson r: {r_neg:.4f} ({get_correlation_strength(r_neg)})</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("No negative correlation found.")
                    st.markdown("<br/>", unsafe_allow_html=True)
                    
                    fig_heat = plot_heatmap(
                        corr_matrix,
                        cmap_theme=heatmap_theme,
                        annot=show_numbers,
                        font_size=h_font_size,
                        title=f"Pearson Correlation Map: {selected_countries[0] if len(selected_countries)==1 else 'Filtered Regions'}"
                    )
                    st.pyplot(fig_heat, use_container_width=True)
                    
                    # Saves for PDF compiling
                    heat_path = os.path.join(EXPORTS_DIR, "correlation_heatmap.png")
                    save_plot_to_png(fig_heat, heat_path)
                    
                    csv_path = os.path.join(EXPORTS_DIR, "correlation_matrix.csv")
                    save_correlation_csv(corr_matrix, csv_path)
                    
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        with open(heat_path, "rb") as file:
                            st.download_button(
                                label="💾 Download Heatmap Image (PNG)",
                                data=file,
                                file_name="online_retail_heatmap.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    with col_dl2:
                        with open(csv_path, "rb") as file:
                            st.download_button(
                                label="📊 Download Correlation CSV",
                                data=file,
                                file_name="online_retail_correlation_matrix.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
        # --- TAB 3: SCATTER RELATIONSHIPS ---
        with tab_scat:
            st.markdown("### 📈 Bivariate Scatter & Trend Analysis")
            
            col_sc1, col_sc2, col_sc3, col_sc4 = st.columns(4)
            with col_sc1:
                x_param = st.selectbox("X-Axis Feature:", options=corr_features, index=0)
            with col_sc2:
                y_param = st.selectbox("Y-Axis Feature:", options=corr_features, index=2 if len(corr_features) > 2 else 1)
            with col_sc3:
                # Groupings
                grouping_cols = ["None", "Country", "Month_Name", "Order_Size", "Weekend"]
                grouping_var = st.selectbox("Categorical Hue:", options=grouping_cols, index=0)
                grouping_hue = None if grouping_var == "None" else grouping_var
            with col_sc4:
                show_reg_line = st.checkbox("Render Linear Regression Trend", value=True)
                
            if x_param == y_param:
                st.warning("Please choose distinct variables for X and Y.")
            else:
                # Calculate and show Pearson correlation coefficient above the chart
                r_val = filtered_df[x_param].corr(filtered_df[y_param])
                if not np.isnan(r_val):
                    r_strength = get_correlation_strength(r_val)
                    st.markdown(f"**Pearson Correlation Coefficient (r):** `{r_val:.4f}` | Relationship Strength: **{r_strength}**")
                else:
                    st.markdown("**Pearson Correlation Coefficient (r):** `N/A` (Constant Variance)")
                
                fig_scat = plot_scatter(
                    filtered_df,
                    x_col=x_param,
                    y_col=y_param,
                    hue_col=grouping_hue,
                    show_regression=show_reg_line
                )
                st.pyplot(fig_scat, use_container_width=True)
                
                # Save plot
                scat_path = os.path.join(EXPORTS_DIR, "scatter_analysis.png")
                save_plot_to_png(fig_scat, scat_path)
                
                with open(scat_path, "rb") as file:
                    st.download_button(
                        label="💾 Download Scatter Chart (PNG)",
                        data=file,
                        file_name=f"scatter_{x_param}_vs_{y_param}.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            # Scatter matrix
            st.markdown("---")
            st.markdown("### 🔲 Multivariate Scatter Matrix (Grid)")
            
            matrix_vars = st.multiselect(
                "Select features for scatter grid (Limit to 3-5):",
                options=corr_features,
                default=corr_features[:4]
            )
            
            col_mat1, col_mat2 = st.columns([1, 3])
            with col_mat1:
                matrix_hue_var = st.selectbox("Hue grouping column:", options=["None", "Order_Size", "Weekend"], key="matrix_hue")
                matrix_hue = None if matrix_hue_var == "None" else matrix_hue_var
                render_matrix = st.button("🚀 Render Pair Plot Grid", use_container_width=True)
                
            with col_mat2:
                if render_matrix:
                    if len(matrix_vars) < 2:
                        st.warning("Select at least 2 features to compile a matrix grid.")
                    elif len(matrix_vars) > 6:
                        st.error("Limit selections to 6 columns to prevent server timeouts.")
                    else:
                        with st.spinner("Compiling grid (sampling 500 rows to ensure UI speed)..."):
                            fig_matrix = plot_pairplot(filtered_df, matrix_vars, hue_col=matrix_hue)
                            st.pyplot(fig_matrix, use_container_width=True)
                            
                            matrix_export_path = os.path.join(EXPORTS_DIR, "pair_plot_matrix.png")
                            save_plot_to_png(fig_matrix, matrix_export_path)
                            
                            with open(matrix_export_path, "rb") as file:
                                st.download_button(
                                    label="💾 Download Pair Plot Grid (PNG)",
                                    data=file,
                                    file_name="retail_pair_plot_matrix.png",
                                    mime="image/png"
                                )
                                
        # --- TAB 4: BUSINESS INSIGHTS & CAUSATION WARNING ---
        with tab_ins:
            corr_mat_ins = calculate_correlation(filtered_df, corr_features)
            pos_corrs, neg_corrs = get_top_correlations(corr_mat_ins, top_n=5)
            insights = generate_business_insights(pos_corrs, neg_corrs)
            disclaimer = get_causation_explanation()
            
            # Caution Callout Block
            st.markdown(f"""
            <div class="caution-card">
                <div class="caution-title">
                    ⚠️ {disclaimer['headline']}
                </div>
                <div class="caution-text">
                    <b>{disclaimer['core_concept']}</b><br/><br/>
                    {md_to_html_bold(disclaimer['retail_reasons'])}
                    <br/><br/>
                    <b>Actionable Guidance:</b> {disclaimer['actionable_advice']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 High-Level Dataset Business Insights")
            
            col_db1, col_db2, col_db3, col_db4 = st.columns(4)
            with col_db1:
                st.markdown(f"""
                <div class="kpi-card" style="padding: 1.1rem; border-left: 4px solid #3b82f6;">
                    <div class="kpi-label" style="font-size: 0.75rem;">🌍 Top Country by Revenue</div>
                    <div class="kpi-value" style="font-size: 1.15rem; color: #1e293b; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{dataset_insights['top_country']}">{dataset_insights['top_country']}</div>
                    <div class="kpi-sub" style="font-size: 0.8rem; color: #64748b;">Rev: ${dataset_insights['top_country_revenue']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_db2:
                st.markdown(f"""
                <div class="kpi-card" style="padding: 1.1rem; border-left: 4px solid #10b981;">
                    <div class="kpi-label" style="font-size: 0.75rem;">📦 Best-Selling SKU Product</div>
                    <div class="kpi-value" style="font-size: 1.15rem; color: #1e293b; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{dataset_insights['best_selling_product']}">{dataset_insights['best_selling_product']}</div>
                    <div class="kpi-sub" style="font-size: 0.8rem; color: #64748b;">Sold: {dataset_insights['best_selling_units']:,} units</div>
                </div>
                """, unsafe_allow_html=True)
            with col_db3:
                st.markdown(f"""
                <div class="kpi-card" style="padding: 1.1rem; border-left: 4px solid #f59e0b;">
                    <div class="kpi-label" style="font-size: 0.75rem;">📅 Highest Revenue Month</div>
                    <div class="kpi-value" style="font-size: 1.15rem; color: #1e293b; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{dataset_insights['highest_rev_month']}">{dataset_insights['highest_rev_month']}</div>
                    <div class="kpi-sub" style="font-size: 0.8rem; color: #64748b;">Rev: ${dataset_insights['highest_rev_month_revenue']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_db4:
                st.markdown(f"""
                <div class="kpi-card" style="padding: 1.1rem; border-left: 4px solid #8b5cf6;">
                    <div class="kpi-label" style="font-size: 0.75rem;">💵 Average Order Value</div>
                    <div class="kpi-value" style="font-size: 1.15rem; color: #1e293b; font-weight: 700;">${dataset_insights['aov']:,.2f}</div>
                    <div class="kpi-sub" style="font-size: 0.8rem; color: #64748b;">Per customer invoice</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("#### 🚨 Notable Dataset Trends & Anomalies")
            for trend in dataset_insights['trends']:
                st.info(f"💡 {trend}")
                
            st.markdown("---")
            
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                st.markdown("### 📈 Strongest Positive Relationships")
                if pos_corrs.empty:
                    st.info("No positive correlations identified.")
                else:
                    for idx, row in pos_corrs.iterrows():
                        str_label = get_correlation_strength(row['Correlation'])
                        st.markdown(f"""
                        **{idx+1}. {row['Variable 1'].replace('_', ' ')} / {row['Variable 2'].replace('_', ' ')}**  
                        • Pearson coefficient (r): `{row['Correlation']:.4f}` | Strength: **{str_label}**
                        """)
            with col_in2:
                st.markdown("### 📉 Strongest Negative Relationships")
                if neg_corrs.empty:
                    st.info("No negative correlations identified.")
                else:
                    for idx, row in neg_corrs.iterrows():
                        str_label = get_correlation_strength(row['Correlation'])
                        st.markdown(f"""
                        **{idx+1}. {row['Variable 1'].replace('_', ' ')} / {row['Variable 2'].replace('_', ' ')}**  
                        • Pearson coefficient (r): `{row['Correlation']:.4f}` | Strength: **{str_label}**
                        """)
                        
            st.markdown("---")
            st.markdown("### 💡 Domain-Specific Purchase Explanations")
            if not insights:
                st.info("No insights compiled. Verify that features are active.")
            else:
                for idx, insight in enumerate(insights):
                    color_bullet = "🟢" if insight['type'] == 'Positive' else "🔴"
                    with st.expander(f"{color_bullet} {insight['var1'].replace('_', ' ')} vs {insight['var2'].replace('_', ' ')} (r = {insight['correlation']:.2f})", expanded=(idx<2)):
                        st.markdown(f"**Correlation Type:** {insight['type']} | **Strength:** {insight['strength']}")
                        st.write(insight['text'])
                        
        # --- TAB 5: PDF EXECUTIVE EXPORTER ---
        with tab_rep:
            st.markdown("### 📄 Professional Document Exporter")
            st.write(
                "Generate a print-ready executive summary report containing data audit statistics, "
                "correlation matrix, ranked relationships, detailed interpretations, and the scientific causation caveat."
            )
            
            col_rp1, col_rp2 = st.columns([1, 2])
            with col_rp1:
                st.markdown("#### Document Setup")
                doc_title = st.text_input("Document Sub-Header Name:", value="Customer Purchase Analytics Report")
                embed_charts = st.checkbox("Embed Visualizations in PDF", value=True)
                compile_btn = st.button("🚀 Compile Reports (PDF & TXT)", use_container_width=True)
                
            with col_rp2:
                if compile_btn:
                    with st.spinner("Formatting grids and compiling ReportLab canvas..."):
                        # Re-calculate clean matrices
                        corr_mat_final = calculate_correlation(filtered_df, corr_features)
                        pos_c, neg_c = get_top_correlations(corr_mat_final, top_n=5)
                        insights_list = generate_business_insights(pos_c, neg_c)
                        disclaim = get_causation_explanation()
                        
                        # Refresh visual graphs
                        fig_h = plot_heatmap(corr_mat_final, cmap_theme='coolwarm', title=f"Pearson Correlation Map")
                        h_path = os.path.join(EXPORTS_DIR, "correlation_heatmap.png")
                        save_plot_to_png(fig_h, h_path)
                        
                        # Default scatter plot (Sales vs Quantity)
                        fig_s = plot_scatter(filtered_df, "Quantity", "Sales", show_regression=True)
                        s_path = os.path.join(EXPORTS_DIR, "scatter_analysis.png")
                        save_plot_to_png(fig_s, s_path)
                        
                        # Chart paths
                        pdf_h_path = h_path if embed_charts else None
                        pdf_s_path = s_path if embed_charts else None
                        
                        # Generate Plaintext report
                        txt_out_path = os.path.join(EXPORTS_DIR, "executive_summary.txt")
                        txt_ok = generate_text_report(
                            dataset_name=doc_title,
                            shape=filtered_df.shape,
                            cleaning_summary=cleaning_summary,
                            pos_corrs=pos_c,
                            neg_corrs=neg_c,
                            insights=insights_list,
                            disclaimer=disclaim,
                            dataset_insights=dataset_insights,
                            output_path=txt_out_path
                        )
                        
                        # Generate PDF Report
                        pdf_out_path = os.path.join(EXPORTS_DIR, "executive_summary.pdf")
                        pdf_ok = generate_pdf_report(
                            dataset_name=doc_title,
                            shape=filtered_df.shape,
                            cleaning_summary=cleaning_summary,
                            pos_corrs=pos_c,
                            neg_corrs=neg_c,
                            insights=insights_list,
                            disclaimer=disclaim,
                            dataset_insights=dataset_insights,
                            heatmap_path=pdf_h_path,
                            scatter_path=pdf_s_path,
                            output_path=pdf_out_path
                        )
                        
                        if txt_ok and pdf_ok:
                            st.success("✅ Plaintext Summary and PDF Executive Report generated successfully!")
                            
                            with open(pdf_out_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Executive PDF Report",
                                    data=f,
                                    file_name="executive_summary.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            with open(txt_out_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Plaintext Summary (TXT)",
                                    data=f,
                                    file_name="executive_summary.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                        else:
                            st.error("An error occurred during document compilation. Check console logs.")
                            
                st.info("💡 Clicking 'Compile Reports' will output structured files to the 'exports/' folder and enable the download prompts.")
