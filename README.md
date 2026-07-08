# Retail BI & Customer Purchase Analytics System

A Streamlit dashboard that ingests, cleans, feature-engineers, and analyzes the **Online Retail II** dataset (UK e-commerce transactions, 2009–2011). It surfaces Pearson correlations between engineered retail metrics, generates plain-English business interpretations, and compiles print-ready PDF/TXT executive reports.

## 🚀 Key Features

- **Two-file ingestion & caching**: Merges `Retail 2009-10.csv` and `Retail 2010-11.csv` into a single dataset, with cached merged/processed outputs so subsequent runs skip re-processing.
- **Automated cleaning**: Standardizes headers, parses `InvoiceDate`, removes duplicates, separates returns/cancellations from genuine purchases, drops rows with missing `Customer ID`/`Description`, filters invalid (≤0) prices, and logs every step in an audit trail.
- **Feature engineering**: Computes transactional (`Sales`), time-based (Year, Month, Quarter, Day, Hour, Weekend), invoice-level (`Invoice_Total`, `Items_Per_Invoice`), customer-level (`Customer_Total_Spend`, `Customer_Order_Count`, `Customer_Average_Order_Value`), product-level, and country-level metrics, plus an `Order_Size` segment (Small/Medium/Large/Bulk).
- **Interactive filtering**: Sidebar filters by country, month, product/SKU search, and minimum customer order count.
- **KPI dashboard & executive brief**: Auto-generated summary of revenue, top country, best-selling product, peak month, AOV, and detected anomalies (weekend sales concentration, geographic concentration, price outliers).
- **Pearson correlation analysis**: Masked upper-triangle heatmap, ranked top positive/negative variable pairs, and domain-specific retail interpretations (price elasticity, CLV drivers, basket-size effects, etc.).
- **Scatter & pairplot visualizations**: Regression-line scatter plots with optional hue grouping, and downsampled scatter matrices for multi-variable exploration.
- **Correlation-vs-causation caveat**: Built-in explanation of confounding variables, reverse causality, and spurious correlation, with actionable guidance (A/B testing, econometric methods).
- **Executive report export**: One-click PDF and TXT executive reports with embedded charts, correlation tables, and insights.

---

## 📁 Repository Structure

```
retail-bi-analytics/
│
├── data/
│   ├── Retail 2009-10.csv                # Raw source file (not included — see Setup)
│   ├── Retail 2010-11.csv                # Raw source file (not included — see Setup)
│   ├── online_retail_II_merged.csv       # Cached merged raw data (auto-generated)
│   ├── online_retail_II_processed.csv    # Cached cleaned + engineered data (auto-generated)
│   └── cleaning_summary.json             # Cached audit log (auto-generated)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # CSV loading, merging, and caching
│   ├── preprocessor.py                   # Cleaning, deduplication, returns handling
│   ├── feature_engineer.py               # Time, invoice, customer, product, country features
│   ├── analytics.py                      # Correlation calculations and business insight generation
│   ├── visualization.py                  # Heatmap, scatter, and pairplot generation
│   └── reporting.py                      # TXT and ReportLab PDF report generation
│
├── exports/                              # Output folder for generated charts, CSVs, and PDFs
├── app.py                                # Streamlit entrypoint and dashboard interface
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Python 3.8+ is required.

### 2. Get the dataset
Download the **Online Retail II** dataset (UCI Machine Learning Repository) and place the two yearly CSVs — `Retail 2009-10.csv` and `Retail 2010-11.csv` — in the project root or a `data/` folder.

### 3. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
streamlit run app.py
```
On first launch, the app will merge the raw files, clean them, engineer features, and cache the results (~30 seconds). Subsequent launches load instantly from cache.

The app opens at `http://localhost:8501`.

---

## 📊 Module Summary

- **data_loader.py** — Checks for a cached merged dataset; if absent, loads and concatenates the two raw yearly CSVs and saves the consolidated file.
- **preprocessor.py** — Cleans column names, parses dates, drops duplicates and invalid rows, separates returns from purchases, and returns a full audit summary dict.
- **feature_engineer.py** — Vectorized feature generation: sales value, time attributes, invoice/customer/product/country aggregates, and order-size segmentation.
- **analytics.py** — Pearson correlation matrix computation, top pair ranking, correlation-strength labeling, retail-specific business insight text, and dataset-level trend detection.
- **visualization.py** — Matplotlib/Seaborn heatmap, scatter (with regression), and pairplot generation with a consistent visual theme.
- **reporting.py** — Compiles the CSV correlation export, plaintext executive summary, and multi-page PDF report with embedded charts.

## ⚠️ Note on Correlation

All correlation findings in this tool are descriptive, not causal. See the in-app "Correlation Does Not Imply Causation" panel for details on confounding variables, reverse causality, and spurious correlation risks specific to retail data.
"# Retail-Business-Intelligence-Customer-Purchase-Analytics-System" 
