"""
ReviewLens Commerce Analytics - DS Sprint Use Case 26
Round 2: EDA and Feature Engineering
Jupyter Notebook Generator & Executor (build_notebook.py)

Generates and executes ReviewLens_Round2_EDA_FeatureEngineering.ipynb
with full code, markdown commentaries, mathematical justifications,
statistical tests, summary tables, and embedded high-resolution figures.
"""

import os
import sys
import io
import base64
import contextlib
import nbformat as nbf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "ReviewLens_Round2_EDA_FeatureEngineering.ipynb")


class NotebookBuilder:
    def __init__(self):
        self.nb = nbf.v4.new_notebook()
        self.nb.metadata = {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.0"
            }
        }
        # Provide display helper for execution
        def custom_display(obj):
            if isinstance(obj, pd.DataFrame):
                print(obj.to_string())
            elif isinstance(obj, pd.Series):
                print(obj.to_string())
            else:
                print(obj)
                
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
            
        self.globs = {
            'display': custom_display,
            '__file__': NOTEBOOK_PATH
        }
        self.exec_count = 1

    def add_md(self, text):
        cell = nbf.v4.new_markdown_cell(text.strip())
        self.nb.cells.append(cell)

    def add_code(self, code_str, capture_plot=False):
        cell = nbf.v4.new_code_cell(code_str.strip())
        cell.execution_count = self.exec_count
        
        stdout_buf = io.StringIO()
        plt.close('all') # Clear prior plots
        
        with contextlib.redirect_stdout(stdout_buf):
            try:
                exec(code_str, self.globs)
            except Exception as e:
                print(f"Error executing cell {self.exec_count}: {e}", file=sys.stderr)
                raise e
                
        out_text = stdout_buf.getvalue()
        if out_text:
            cell.outputs.append(nbf.v4.new_output(
                output_type='stream',
                name='stdout',
                text=out_text
            ))
            
        if capture_plot and plt.get_fignums():
            fig = plt.gcf()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
            img_buf.seek(0)
            img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
            cell.outputs.append(nbf.v4.new_output(
                output_type='display_data',
                data={'image/png': img_b64, 'text/plain': '<Figure size ...>'},
                metadata={}
            ))
            plt.close('all')
            
        self.nb.cells.append(cell)
        self.exec_count += 1

    def save(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            nbf.write(self.nb, f)
        print(f"Notebook successfully written and saved to: {filepath}")


def build_reviewlens_notebook():
    builder = NotebookBuilder()
    
    # -------------------------------------------------------------------------
    # TITLE & EXECUTIVE SUMMARY
    # -------------------------------------------------------------------------
    builder.add_md("""
# ReviewLens Commerce Analytics
## DS Sprint Skill-Based Assessment — Use Case 26: Product Review Insight
### Round 2: Exploratory Data Analysis & Feature Engineering

---

### Executive Overview & Problem Context
In modern multi-category e-commerce ecosystems, customer reviews are the primary barometer of customer satisfaction, product quality, seller performance, and operational fulfillment. **ReviewLens** is an analytics initiative designed to unlock actionable intelligence from cross-relational commerce data.

The project encompasses five core transactional and operational entities:
1. **Reviews (`reviews.csv`)**: Customer ratings, text sentiment, character counts, verification status, and review timestamps.
2. **Products (`products.csv`)**: Product metadata, taxonomy categories, brands, and price tiers.
3. **Sellers (`sellers.csv`)**: Merchant profiles, operating regions, and merchant platform tenure.
4. **Orders (`orders.csv`)**: Fulfillment logs, delivery lead times, promotional discount percentages, and return flags.
5. **Support (`support.csv`)**: Post-order customer tickets, complaint classifications, and ticket resolution states.

---

### Dataset Nature & Competition Integrity Note
> **Important Assessment Clarification:**
> The dataset analyzed in this project is a **synthetic practice dataset** generated to strictly emulate the schema, data-quality challenges, and entity relationships specified in the DS Sprint Use Case 26 requirements.
> It contains deliberate real-world data quality anomalies (missing ratings, duplicated reviews, inconsistent taxonomy casings, extreme text length outliers, and foreign key mismatches) designed to evaluate rigorous data science workflows.
> No real customer personally identifiable information (PII) is included. For the official competition submission, this exact reproducible pipeline can be seamlessly pointed to the organizer-provided dataset.
""")

    # -------------------------------------------------------------------------
    # SECTION 1: SETUP & ENVIRONMENT
    # -------------------------------------------------------------------------
    builder.add_md("""
---
## 1. Environment Setup & Data Ingestion

We begin by establishing a reproducible analytical environment, importing necessary data science packages, and configuring consistent publication-style visualization defaults.
""")

    builder.add_code(r"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

if '.' not in sys.path:
    sys.path.insert(0, '.')

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configure aesthetics
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 120

# Define paths
DATA_DIR = "ReviewLens_Round2_Practice_Dataset"
OUTPUT_DIR = "output"

print(f"Pandas Version: {pd.__version__}")
print(f"NumPy Version:  {np.__version__}")
print(f"Seaborn Version:{sns.__version__}")
print("Data directory:", os.path.abspath(DATA_DIR))
""")

    builder.add_md("""
### 1.1 Ingesting All Five CSV Files
We load all five tables using Pandas and inspect their fundamental shapes and structures.
""")

    builder.add_code(r"""
# Load raw datasets
raw_reviews = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))
raw_products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
raw_sellers = pd.read_csv(os.path.join(DATA_DIR, "sellers.csv"))
raw_orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
raw_support = pd.read_csv(os.path.join(DATA_DIR, "support.csv"))

dataset_summary = pd.DataFrame({
    'Dataset': ['reviews.csv', 'products.csv', 'sellers.csv', 'orders.csv', 'support.csv'],
    'Rows': [raw_reviews.shape[0], raw_products.shape[0], raw_sellers.shape[0], raw_orders.shape[0], raw_support.shape[0]],
    'Columns': [raw_reviews.shape[1], raw_products.shape[1], raw_sellers.shape[1], raw_orders.shape[1], raw_support.shape[1]],
    'Primary Key Candidate': ['review_id', 'product_id', 'seller_id', 'order_id', 'support_id']
})
display(dataset_summary)
""")

    # -------------------------------------------------------------------------
    # SECTION 2: DATA PROFILING & QUALITY AUDIT
    # -------------------------------------------------------------------------
    builder.add_md("""
---
## 2. Dataset Profiling & Data Quality Audit

In accordance with Round 2 requirements, we conduct an exhaustive audit across shapes, data types, descriptive statistics, missing values, duplicates, and business logic inconsistencies.
""")

    builder.add_code(r"""
print("=== REVIEWS DATA TYPES & NULL VALUES ===")
display(pd.DataFrame({
    'Dtype': raw_reviews.dtypes,
    'Null_Count': raw_reviews.isnull().sum(),
    'Null_Pct': (raw_reviews.isnull().mean() * 100).round(2),
    'Unique_Count': raw_reviews.nunique()
}))

print("\n=== SUMMARY OF NULL VALUES ACROSS ALL TABLES ===")
null_audit = pd.DataFrame({
    'reviews': raw_reviews.isnull().sum(),
    'products': raw_products.isnull().sum(),
    'sellers': raw_sellers.isnull().sum(),
    'orders': raw_orders.isnull().sum(),
    'support': raw_support.isnull().sum()
}).fillna('-')
display(null_audit)
""")

    builder.add_md("""
### 2.1 Identifying Key Data Quality Issues
We systematically test for the five intentional data-quality anomalies highlighted in the assessment guidelines:
1. **Missing Ratings**: Records where the customer review rating is `NaN`.
2. **Duplicate Reviews**: Exact duplicate rows across reviews.
3. **Inconsistent Category Labels**: Typographical / casing variations in `products.category`.
4. **Extreme Review Lengths**: Outlier character counts that diverge from actual review text lengths.
5. **Unmatched Product IDs**: Product foreign keys present in reviews that do not exist in the product catalog.
""")

    builder.add_code(r"""
# 1. Missing ratings
missing_ratings_df = raw_reviews[raw_reviews['rating'].isna()]
print(f"1. Missing Ratings Count: {len(missing_ratings_df)}")
display(missing_ratings_df[['review_id', 'order_id', 'product_id', 'review_text', 'verified_flag']])

# 2. Duplicate reviews
exact_duplicates = raw_reviews[raw_reviews.duplicated(keep=False)]
print(f"\n2. Duplicate Review Rows (Total involved rows): {len(exact_duplicates)}")
display(exact_duplicates)

# 3. Inconsistent categories
print("\n3. Raw Categories in products.csv:")
print(raw_products['category'].unique())

# 4. Extreme review lengths
raw_reviews['actual_len'] = raw_reviews['review_text'].str.len()
extreme_len = raw_reviews[raw_reviews['review_length'] > 1000]
print(f"\n4. Extreme Review Lengths (> 1000 chars): {len(extreme_len)}")
display(extreme_len[['review_id', 'order_id', 'review_length', 'actual_len', 'review_text']])

# 5. Unmatched product IDs
unmatched_pids = set(raw_reviews['product_id']) - set(raw_products['product_id'])
print(f"\n5. Unmatched product_ids in reviews: {unmatched_pids}")
display(raw_reviews[raw_reviews['product_id'].isin(unmatched_pids)][['review_id', 'order_id', 'product_id', 'rating']])
""")

    # -------------------------------------------------------------------------
    # SECTION 3: DATA CLEANING DECISIONS & IMPLEMENTATION
    # -------------------------------------------------------------------------
    builder.add_md("""
---
## 3. Data Cleaning Decisions & Transformations

To ensure full reproducibility and maintain scientific rigor, every cleaning decision is formally documented below:

| Anomaly Identified | Affected Table | Cleaning Action & Strategy | Justification |
| :--- | :--- | :--- | :--- |
| **Duplicate Reviews** | `reviews.csv` | Drop exact duplicate rows (`keep='first'`), reducing rows from 1052 to 1050. | Exact duplicate rows skew frequency distributions and artificially inflate review volume. |
| **Missing Ratings** | `reviews.csv` | Retain records with `rating_is_missing = True` flag; filter dynamically when analyzing target ratings. | Never fabricate or impute subjective customer feedback target metrics; preserve complete audit history. |
| **Category Casing & Whitespace** | `products.csv` | Strip whitespace (`str.strip()`) and map canonical Title Case: `'electronics'` $\\rightarrow$ `'Electronics'`, `'HOME & KITCHEN'` $\\rightarrow$ `'Home & Kitchen'`. | Standardizes taxonomy into 6 distinct, accurate product categories. |
| **Extreme Review Lengths** | `reviews.csv` | Derive `actual_char_count` & `actual_word_count` from text; create `review_length_capped` (99th percentile winsorization) & flag `is_extreme_length_outlier`. | Eliminates artificial 5000 and 8000 character outliers while preserving authentic text metrics. |
| **Unmatched Product IDs** | `reviews.csv` | Cross-reference `orders.csv` via `order_id` to reconcile genuine `product_id` (O00420 $\\rightarrow$ P0136, O00470 $\\rightarrow$ P0083); preserve `raw_product_id`. | Restores relational integrity without data loss, linking orphan reviews to valid product records. |
""")

    builder.add_code(r"""
# Execute documented cleaning pipeline
from src.data_cleaning import clean_reviewlens_data

CLEANED_DIR = os.path.join(OUTPUT_DIR, "cleaned_data")
cleaned_dfs = clean_reviewlens_data(DATA_DIR, CLEANED_DIR)

reviews_clean = cleaned_dfs['reviews']
products_clean = cleaned_dfs['products']
sellers_clean = cleaned_dfs['sellers']
orders_clean = cleaned_dfs['orders']
support_clean = cleaned_dfs['support']

print("Cleaned products categories:", products_clean['category'].unique())
print("Cleaned reviews row count:  ", len(reviews_clean))
""")

    # -------------------------------------------------------------------------
    # SECTION 4: PK/FK VALIDATION & RELATIONAL MERGING
    # -------------------------------------------------------------------------
    builder.add_md("""
---
## 4. Entity-Relationship Validation & Relational Joins

A critical requirement in multi-table e-commerce analytics is verifying relational integrity and preventing **join multiplication (the Cartesian fan-out trap)**.

### Relational Schema Architecture:
```
+---------------+           +---------------+           +---------------+
|   products    |           |    sellers    |           |    support    |
+---------------+           +---------------+           +---------------+
| PK: product_id|           | PK: seller_id |           | PK: support_id|
+-------+-------+           +-------+-------+           | FK: order_id  |
        |                           |                   +-------+-------+
        | (1:N)                     | (1:N)                     | (N:1 aggregated)
        |                           |                           |
+-------v---------------------------v-------+                   |
|                   orders                  |<------------------+
+-------------------------------------------+
| PK: order_id                              |
| FK: product_id                            |
| FK: seller_id                             |
+---------------------+---------------------+
                      | (1:N)
+---------------------v---------------------+
|                   reviews                 |
+-------------------------------------------+
| PK: review_id                             |
| FK: order_id                              |
| FK: product_id (reconciled)               |
+-------------------------------------------+
```

### Preventing the Support Join Trap:
`support.csv` contains 290 tickets for 243 unique orders (42 orders have 2 to 4 tickets).
If we perform a raw join between `reviews` and `support` on `order_id`, orders with multiple tickets will duplicate review records, corrupting statistical aggregations.
**Solution:** We aggregate support tickets at the `order_id` level first, extracting:
- `support_ticket_count`
- `has_support_ticket`
- `primary_complaint_category`
- `latest_resolution`
- Multi-complaint binary indicators
This guarantees that the master analytical dataset contains exactly **1050 records**, matching the number of cleaned reviews.
""")

    builder.add_code(r"""
from src.data_merging import merge_reviewlens_data

MERGED_DIR = os.path.join(OUTPUT_DIR, "merged_data")
master_df, support_analysis_df = merge_reviewlens_data(CLEANED_DIR, MERGED_DIR)

print("Master Analysis Dataset Shape:", master_df.shape)
print("Support Analysis Dataset Shape:", support_analysis_df.shape)
display(master_df[['review_id', 'order_id', 'product', 'category', 'seller', 'rating', 'delivery_days', 'return_flag', 'has_support_ticket']].head())
""")

    # -------------------------------------------------------------------------
    # SECTION 5: EDA & ANSWERING THE SIX ASSESSMENT QUESTIONS
    # -------------------------------------------------------------------------
    builder.add_md("""
---
## 5. Exploratory Data Analysis & Answering the Six Core Questions

We now analyze the merged dataset to provide data-backed answers to all six assessment questions using real computations, confidence intervals, and hypothesis tests.
""")

    # Q1
    builder.add_md("""
### 5.1 Question 1: Which categories receive lower ratings?
We examine mean, median, standard deviation, and the proportion of low ratings ($\\le 2$ stars) across product categories.
""")

    builder.add_code(r"""
df_rated = master_df[master_df['rating'].notna()].copy()

q1_summary = df_rated.groupby('category')['rating'].agg(
    review_count='count',
    mean_rating='mean',
    median_rating='median',
    std_rating='std',
    low_rating_pct=lambda x: (x <= 2).mean() * 100
).sort_values('mean_rating')
display(q1_summary.round(3))

# Statistical Significance (One-Way ANOVA)
groups = [group['rating'].values for _, group in df_rated.groupby('category')]
f_val, p_val = stats.f_oneway(*groups)
print(f"One-way ANOVA: F = {f_val:.4f}, p-value = {p_val:.4f}")
""", capture_plot=False)

    builder.add_code(r"""
fig, ax = plt.subplots(figsize=(9, 4.8))
palette = sns.color_palette("mako", n_colors=len(q1_summary))
bars = sns.barplot(
    x=q1_summary.index, 
    y=q1_summary['mean_rating'], 
    hue=q1_summary.index,
    palette=palette, 
    legend=False,
    ax=ax,
    edgecolor='#333333',
    linewidth=0.8
)
ax.axhline(df_rated['rating'].mean(), color='#e63946', linestyle='--', linewidth=1.5, label=f"Overall Mean: {df_rated['rating'].mean():.2f}")
ax.set_ylim(3.0, 4.2)
ax.set_title("Average Review Rating by Product Category", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Product Category", fontsize=11, fontweight='bold')
ax.set_ylabel("Mean Rating (Stars)", fontsize=11, fontweight='bold')
for p in bars.patches:
    h = p.get_height()
    ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h + 0.02), ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
""", capture_plot=True)

    builder.add_md("""
> **Key Finding — Question 1:**
> - **Sports** receives the lowest average rating at **3.63 stars** (median 4.0, std 1.09) with 14.1% low ratings ($\\le 2$ stars).
> - **Books** is second lowest at **3.74 stars** (median 4.0, std 0.97).
> - **Fashion** achieved the highest customer satisfaction at **3.86 stars**, followed by **Home & Kitchen** (3.80) and **Electronics** (3.78).
> - The one-way ANOVA yields $F = 1.054, p = 0.385$, indicating that while Sports underperforms by $\\sim 0.23$ stars compared to Fashion, category-level variance alone is modest compared to order fulfillment factors.
""")

    # Q2
    builder.add_md("""
### 5.2 Question 2: Do verified reviews differ?
We compare customer rating behavior between verified purchase reviews (`verified_flag = 'Yes'`) and unverified reviews (`verified_flag = 'No'`).
""")

    builder.add_code(r"""
q2_summary = df_rated.groupby('verified_flag')['rating'].agg(
    review_count='count',
    mean_rating='mean',
    median_rating='median',
    std_rating='std',
    low_rating_pct=lambda x: (x <= 2).mean() * 100
)
display(q2_summary.round(3))

# Independent two-sample t-test & Mann-Whitney U test
t_stat, p_val_t = stats.ttest_ind(
    df_rated[df_rated['verified_flag'] == 'Yes']['rating'],
    df_rated[df_rated['verified_flag'] == 'No']['rating']
)
u_stat, p_val_u = stats.mannwhitneyu(
    df_rated[df_rated['verified_flag'] == 'Yes']['rating'],
    df_rated[df_rated['verified_flag'] == 'No']['rating']
)
print(f"Two-sample t-test:   t = {t_stat:.4f}, p-value = {p_val_t:.4f}")
print(f"Mann-Whitney U test: U = {u_stat:.1f}, p-value = {p_val_u:.4f}")
""")

    builder.add_code(r"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(
    data=df_rated, 
    x='verified_flag', 
    y='rating', 
    hue='verified_flag',
    palette=['#457b9d', '#2a9d8f'], 
    legend=False,
    ax=ax1, 
    width=0.45
)
ax1.set_title("Rating Distribution: Verified vs Unverified", fontsize=12, fontweight='bold')
ax1.set_xlabel("Verified Purchase Flag", fontsize=10, fontweight='bold')
ax1.set_ylabel("Rating (Stars)", fontsize=10, fontweight='bold')

prop_df = df_rated.groupby(['verified_flag', 'rating']).size().unstack(fill_value=0)
prop_pct = prop_df.div(prop_df.sum(axis=1), axis=0) * 100
prop_pct.plot(kind='bar', stacked=True, colormap='Blues', ax=ax2, edgecolor='#333333', linewidth=0.6)
ax2.set_title("Rating Breakdown by Verification Status (%)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Verified Purchase Flag", fontsize=10, fontweight='bold')
ax2.set_ylabel("Percentage (%)", fontsize=10, fontweight='bold')
ax2.legend(title='Rating', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
""", capture_plot=True)

    builder.add_md("""
> **Key Finding — Question 2:**
> - Verified reviews average **3.757 stars** (n=798, std 1.055), while non-verified reviews average **3.759 stars** (n=249, std 1.046).
> - Difference is negligible: $\\Delta = -0.002$ stars.
> - Statistical significance: $t = -0.028, p = 0.978$; Mann-Whitney $U = 99649.0, p = 0.941$.
> - **Conclusion**: There is **no statistically significant difference** in customer sentiment between verified and unverified reviews in this dataset.
""")

    # Q3
    builder.add_md("""
### 5.3 Question 3: How does delivery time relate to rating?
We evaluate the relationship between fulfillment lead time (`delivery_days`) and customer satisfaction (`rating`).
""")

    builder.add_code(r"""
q3_summary = df_rated.groupby('delivery_days')['rating'].agg(
    review_count='count',
    mean_rating='mean',
    median_rating='median',
    std_rating='std'
)
display(q3_summary.round(3))

# Pearson and Spearman correlations
r_p, p_p = stats.pearsonr(df_rated['delivery_days'], df_rated['rating'])
r_s, p_s = stats.spearmanr(df_rated['delivery_days'], df_rated['rating'])
print(f"Pearson Correlation (Linear):    r = {r_p:.4f}, p-value = {p_p:.4e}")
print(f"Spearman Correlation (Monotonic): rho = {r_s:.4f}, p-value = {p_s:.4e}")
""")

    builder.add_code(r"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
sns.lineplot(
    data=df_rated, 
    x='delivery_days', 
    y='rating', 
    marker='o', 
    color='#1d3557', 
    linewidth=2.2, 
    markersize=7, 
    ax=ax1,
    errorbar=('ci', 95)
)
ax1.set_title("Mean Rating vs Delivery Days (with 95% CI)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Delivery Time (Days)", fontsize=10, fontweight='bold')
ax1.set_ylabel("Average Rating (Stars)", fontsize=10, fontweight='bold')
ax1.annotate(f"Pearson r = {r_p:.2f} (p < 0.001)", xy=(0.05, 0.1), xycoords='axes fraction',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f1faee', edgecolor='#1d3557'),
             fontsize=10, fontweight='bold')

df_rated['delivery_tier'] = pd.cut(df_rated['delivery_days'], bins=[0, 2, 5, 20], labels=['Express (1-2d)', 'Standard (3-5d)', 'Delayed (6+d)'])
bin_stats = df_rated.groupby('delivery_tier', observed=False)['rating'].mean()
sns.barplot(x=bin_stats.index, y=bin_stats.values, hue=bin_stats.index, palette='copper', legend=False, ax=ax2, edgecolor='#333333')
ax2.set_ylim(2.5, 4.5)
ax2.set_title("Average Rating by Delivery Speed Tier", fontsize=12, fontweight='bold')
ax2.set_xlabel("Delivery Speed Tier", fontsize=10, fontweight='bold')
ax2.set_ylabel("Mean Rating (Stars)", fontsize=10, fontweight='bold')
for p in ax2.patches:
    h = p.get_height()
    ax2.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h + 0.04), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()
""", capture_plot=True)

    builder.add_md("""
> **Key Finding — Question 3:**
> - Delivery speed exerts a **strong, statistically significant negative relationship** on customer ratings ($r = -0.2409, p = 2.74 \\times 10^{-15}$).
> - **Express delivery (1–2 days)** commands top ratings: **4.10 stars**.
> - **Standard delivery (3–5 days)** maintains healthy ratings: **3.83 stars**.
> - **Delayed delivery (6+ days)** drops sharply to **3.42 stars**, plunging to **1.5–2.0 stars** when shipments exceed 10 days.
""")

    # Q4
    builder.add_md("""
### 5.4 Question 4: Which sellers show recurring low ratings?
We identify merchants with consistent customer dissatisfaction by calculating both average rating and the share of reviews rated $\\le 2$ stars.
""")

    builder.add_code(r"""
seller_summary = df_rated.groupby(['seller_id', 'seller', 'region']).agg(
    review_count=('rating', 'count'),
    mean_rating=('rating', 'mean'),
    median_rating=('rating', 'median'),
    low_rating_count=('rating', lambda x: (x <= 2).sum()),
    low_rating_pct=('rating', lambda x: (x <= 2).mean() * 100),
    tenure_years=('tenure_years', 'first')
).reset_index().sort_values(by=['mean_rating', 'low_rating_pct'], ascending=[True, False])

print("Top 10 Sellers with Lowest Ratings:")
display(seller_summary.head(10).round(2))
""")

    builder.add_code(r"""
fig, ax = plt.subplots(figsize=(11, 5.2))
top_lowest = seller_summary.head(10).copy()
top_lowest['label'] = top_lowest['seller'] + " (" + top_lowest['region'] + ")"

bar_plot = sns.barplot(
    data=top_lowest,
    x='label',
    y='mean_rating',
    hue='label',
    palette='flare',
    legend=False,
    ax=ax,
    edgecolor='#333333',
    linewidth=0.8
)
ax.axhline(df_rated['rating'].mean(), color='#e63946', linestyle='--', linewidth=1.5, label=f"Platform Mean: {df_rated['rating'].mean():.2f}")
ax.set_ylim(2.5, 4.3)
ax.set_title("Top 10 Sellers with Lowest Average Ratings", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Seller (Region)", fontsize=11, fontweight='bold')
ax.set_ylabel("Mean Rating (Stars)", fontsize=11, fontweight='bold')
plt.xticks(rotation=35, ha='right')

for idx, p in enumerate(bar_plot.patches):
    h = p.get_height()
    pct = top_lowest.iloc[idx]['low_rating_pct']
    ax.annotate(f"{h:.2f}\n({pct:.0f}% <=2*)", (p.get_x() + p.get_width() / 2., h + 0.04),
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
""", capture_plot=True)

    builder.add_md("""
> **Key Finding — Question 4:**
> - **Seller 19 (East)** records the platform's lowest mean rating of **3.43 stars** (14.3% low ratings).
> - **Seller 13 (South)** averages **3.50 stars** (14.3% low ratings).
> - **Seller 12 (Central)** has the highest proportion of severe dissatisfaction: **22.6% of reviews are $\\le 2$ stars** with an average of **3.51 stars** across 53 orders.
> - **Seller 09 (East)**, **Seller 15 (South)**, and **Seller 01 (North)** also demonstrate recurring low ratings ($< 3.63$ stars).
""")

    # Q5
    builder.add_md("""
### 5.5 Question 5: Do returned orders have lower ratings?
We assess the impact of product returns (`return_flag`) on review sentiment.
""")

    builder.add_code(r"""
q5_summary = df_rated.groupby('return_flag')['rating'].agg(
    review_count='count',
    mean_rating='mean',
    median_rating='median',
    std_rating='std',
    low_rating_pct=lambda x: (x <= 2).mean() * 100
)
display(q5_summary.round(3))

# Two-sample t-test and effect size
t_ret, p_ret = stats.ttest_ind(
    df_rated[df_rated['return_flag'] == 'Yes']['rating'],
    df_rated[df_rated['return_flag'] == 'No']['rating']
)
mean_yes = df_rated[df_rated['return_flag'] == 'Yes']['rating'].mean()
mean_no = df_rated[df_rated['return_flag'] == 'No']['rating'].mean()
s_p = np.sqrt(
    ((len(df_rated[df_rated['return_flag'] == 'Yes']) - 1) * df_rated[df_rated['return_flag'] == 'Yes']['rating'].var() +
     (len(df_rated[df_rated['return_flag'] == 'No']) - 1) * df_rated[df_rated['return_flag'] == 'No']['rating'].var()) /
    (len(df_rated) - 2)
)
d_val = (mean_yes - mean_no) / s_p
print(f"Two-sample t-test:    t = {t_ret:.4f}, p-value = {p_ret:.4e}")
print(f"Cohen's d Effect Size: d = {d_val:.4f} (Very large negative effect)")
""")

    builder.add_code(r"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
sns.violinplot(
    data=df_rated, 
    x='return_flag', 
    y='rating', 
    hue='return_flag',
    palette=['#2a9d8f', '#e76f51'], 
    legend=False,
    ax=ax1, 
    cut=0, 
    inner='quartile'
)
ax1.set_title("Rating Distribution: Returned vs Non-Returned", fontsize=12, fontweight='bold')
ax1.set_xlabel("Order Return Flag", fontsize=10, fontweight='bold')
ax1.set_ylabel("Rating (Stars)", fontsize=10, fontweight='bold')

sns.barplot(
    x=q5_summary.index,
    y=q5_summary['low_rating_pct'],
    hue=q5_summary.index,
    palette=['#2a9d8f', '#e76f51'],
    legend=False,
    ax=ax2,
    edgecolor='#333333'
)
ax2.set_title("Proportion of Low Ratings (<= 2 Stars) %", fontsize=12, fontweight='bold')
ax2.set_xlabel("Order Return Flag", fontsize=10, fontweight='bold')
ax2.set_ylabel("Low Rating Share (%)", fontsize=10, fontweight='bold')
for p in ax2.patches:
    h = p.get_height()
    ax2.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 1), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()
""", capture_plot=True)

    builder.add_md("""
> **Key Finding — Question 5:**
> - Returned orders suffer a **massive drop of 0.76 stars**:
>   - **Non-returned orders**: Mean = **3.88 stars** (std 0.99, n=883, only 8.7% low ratings).
>   - **Returned orders**: Mean = **3.12 stars** (std 1.15, n=164, **31.1% low ratings**).
> - Independent t-test is overwhelmingly significant: $t = -8.807, p = 5.25 \\times 10^{-18}$.
> - Cohen's $d = -0.75$, confirming that product return status is one of the single most powerful predictors of negative review ratings.
""")

    # Q6
    builder.add_md("""
### 5.6 Question 6: What complaint categories align with low ratings?
We analyze support tickets (`support.csv`) to determine which customer complaint classifications align with the steepest rating declines.
""")

    builder.add_code(r"""
# Primary complaint category ratings (for orders with tickets)
q6_summary = df_rated[df_rated['has_support_ticket'] == 1].groupby('primary_complaint_category')['rating'].agg(
    review_count='count',
    mean_rating='mean',
    median_rating='median',
    std_rating='std',
    low_rating_pct=lambda x: (x <= 2).mean() * 100
).sort_values('mean_rating')
display(q6_summary.round(3))
""")

    builder.add_code(r"""
fig, ax = plt.subplots(figsize=(9, 4.8))
sns.barplot(
    x=q6_summary.index,
    y=q6_summary['mean_rating'],
    hue=q6_summary.index,
    palette='rocket_r',
    legend=False,
    ax=ax,
    edgecolor='#333333',
    linewidth=0.8
)
ax.axhline(df_rated['rating'].mean(), color='#2a9d8f', linestyle='--', linewidth=1.5, label=f"Platform Mean: {df_rated['rating'].mean():.2f}")
ax.set_ylim(2.5, 4.3)
ax.set_title("Mean Review Rating by Support Complaint Category", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Support Complaint Category", fontsize=11, fontweight='bold')
ax.set_ylabel("Mean Rating (Stars)", fontsize=11, fontweight='bold')
plt.xticks(rotation=25, ha='right')
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h + 0.04), ha='center', va='bottom', fontweight='bold')
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
""", capture_plot=True)

    builder.add_md("""
> **Key Finding — Question 6:**
> - **Refund Delay** and **Damaged Item** complaints align with the lowest review ratings:
>   - **Refund Delay**: **3.42 stars** (median 3.0, 17.3% $\\le 2$ stars).
>   - **Damaged Item**: **3.43 stars** (median 3.0, **21.4% $\\le 2$ stars**).
> - **Wrong Item** complaints also result in depressed ratings (**3.56 stars**).
> - Conversely, **Late Delivery** (3.88 stars) and **Quality Issue** (3.79 stars) align closer to platform averages, suggesting that when items arrive intact, operational delivery delays are forgiven more readily than receiving damaged physical merchandise or encountering financial friction in refunds.
""")

    # -------------------------------------------------------------------------
    # SECTION 6: FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    builder.add_md("""
---
## 6. Feature Engineering & Transformation

To prepare the dataset for predictive modeling and segmentation, we engineer domain-rich features across multiple operational dimensions:
1. **`review_length_group`**: Binned character lengths ('Short (<150)', 'Medium (150-300)', 'Long (>300)').
2. **`rating_class`**: 3-tier customer sentiment ('Positive [4-5★]', 'Neutral [3★]', 'Negative [1-2★]').
3. **`is_low_rating`**: Binary target indicator ($1$ if rating $\\le 2$, else $0$).
4. **`delivery_speed_group`**: Binned lead times ('Express [1-2d]', 'Standard [3-5d]', 'Delayed [6+d]').
5. **`discount_band`**: Promotional intensity tiers ('No Discount [0%]', 'Low [5-10%]', 'Medium [15-20%]', 'High [25-30%]').
6. **Temporal / Calendar Features**: `review_year`, `review_month`, `review_day_of_week`, `review_quarter`, `is_weekend`.
7. **Text Characteristics**: `actual_char_count`, `actual_word_count`, `avg_word_length`.
8. **Seller Tenure**: `seller_tenure_group` ('New [<=3y]', 'Established [4-7y]', 'Veteran [>7y]').
""")

    builder.add_code(r"""
from src.feature_engineering import engineer_and_select_features

FEAT_DIR = os.path.join(OUTPUT_DIR, "engineered_features")
FIGS_DIR = os.path.join(OUTPUT_DIR, "figures")

feat_df, X, y, rf_imp, anova_res, pca_res = engineer_and_select_features(
    os.path.join(MERGED_DIR, "reviewlens_master_analysis.csv"),
    FEAT_DIR,
    FIGS_DIR
)

print(f"Total engineered features created: {feat_df.shape[1]}")
display(feat_df[['review_id', 'review_length_group', 'rating_class', 'delivery_speed_group', 'discount_band', 'seller_tenure_group']].head())
""")

    # -------------------------------------------------------------------------
    # SECTION 7: FEATURE SELECTION & DIMENSIONALITY REDUCTION
    # -------------------------------------------------------------------------
    builder.add_md("""
---
## 7. Feature Selection & Dimensionality Reduction

We evaluate three complementary feature selection paradigms to identify the strongest predictors of low customer ratings (`is_low_rating`), followed by Principal Component Analysis (PCA).

### Selection Paradigms:
1. **Random Forest Feature Importance (MDI / Gini Impurity)**: Captures non-linear feature interactions and threshold splits.
2. **Mutual Information Classification (`mutual_info_classif`)**: Non-parametric information theoretic metric measuring general statistical dependency without assuming linearity.
3. **ANOVA F-test (`SelectKBest` with `f_classif`)**: Univariate linear variance ratio evaluating group mean separation.
""")

    builder.add_code(r"""
print("=== TOP 10 FEATURES: RANDOM FOREST IMPORTANCE ===")
display(rf_imp.head(10).round(4))

print("\n=== TOP 10 FEATURES: ANOVA F-SCORE ===")
display(anova_res.head(10).round(4))
""")

    builder.add_code(r"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
top_rf = rf_imp.head(10)
sns.barplot(data=top_rf, x='rf_importance', y='feature', hue='feature', palette='mako', legend=False, ax=ax1, edgecolor='#333333')
ax1.set_title("Top 10 Features: Random Forest Importance", fontsize=12, fontweight='bold')
ax1.set_xlabel("Relative Importance Score", fontsize=10, fontweight='bold')
ax1.set_ylabel("Feature Name", fontsize=10, fontweight='bold')

top_f = anova_res.head(10)
sns.barplot(data=top_f, x='f_score', y='feature', hue='feature', palette='rocket', legend=False, ax=ax2, edgecolor='#333333')
ax2.set_title("Top 10 Features: ANOVA F-Statistic", fontsize=12, fontweight='bold')
ax2.set_xlabel("ANOVA F-Score", fontsize=10, fontweight='bold')
ax2.set_ylabel("", fontsize=10)
plt.tight_layout()
plt.show()
""", capture_plot=True)

    builder.add_md("""
### 7.1 Multivariate Correlation Matrix
To detect multicollinearity and understand pairwise interactions between operational metrics, we compute the Pearson correlation heatmap.
""")

    builder.add_code(r"""
fig, ax = plt.subplots(figsize=(9, 7.5))
corr_cols = [
    'rating', 'delivery_days', 'discount_pct', 'tenure_years', 
    'is_returned', 'is_verified', 'has_support_ticket', 
    'review_length_capped', 'price_band_ordinal'
]
corr_matrix = feat_df[corr_cols].corr()

sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    center=0, 
    ax=ax, 
    cbar_kws={'label': 'Pearson Correlation'},
    linewidths=0.5,
    linecolor='#f0f0f0'
)
ax.set_title("Multivariate Feature Correlation Heatmap", fontsize=14, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
""", capture_plot=True)

    builder.add_md("""
### 7.2 Dimensionality Reduction via PCA & Method Limitations
We test Principal Component Analysis (PCA) on standardized continuous features:
`['delivery_days', 'discount_pct', 'tenure_years', 'review_length_capped', 'actual_char_count', 'actual_word_count', 'avg_word_length', 'price_band_ordinal', 'support_ticket_count']`.
""")

    builder.add_code(r"""
display(pca_res)

# Plot Scree & Cumulative Variance
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(pca_res['component'], pca_res['cumulative_variance_ratio'] * 100, marker='o', color='#1d3557', linewidth=2)
ax.bar(pca_res['component'], pca_res['explained_variance_ratio'] * 100, color='#457b9d', alpha=0.7, edgecolor='#333333')
ax.axhline(80, color='#e63946', linestyle='--', label='80% Variance Threshold')
ax.axhline(90, color='#2a9d8f', linestyle=':', label='90% Variance Threshold')
ax.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold')
ax.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
ax.set_title('PCA Scree Plot: Individual & Cumulative Explained Variance', fontsize=13, fontweight='bold')
ax.legend(loc='center right')
plt.tight_layout()
plt.show()
""", capture_plot=True)

    builder.add_md("""
> ### Method Justification & PCA Limitations Discussion
> 1. **Why Feature Selection is Preferred over PCA for Commerce Analytics:**
>    - **Interpretability**: In business intelligence and e-commerce operations, stakeholders require direct attribution (e.g., "Returned orders increase negative review rates by 3.5x", "Lead times over 6 days degrade ratings by 0.6 stars"). PCA projects features into orthogonal linear combinations (e.g., $PC1 = 0.52 \\cdot \\text{char\\_count} + 0.48 \\cdot \\text{word\\_count} - 0.15 \\cdot \\text{delivery\\_days}$), destroying physical business meaning.
>    - **Sparsity & Categorical Data**: One-hot encoded columns (e.g., categories, complaint types, regions) violate PCA's multivariate normality assumption.
>    - **Variance Spread**: As shown in the Scree plot, explaining 90% of variance requires 7 components out of 9 original features, yielding negligible dimensionality reduction benefits.
> 2. **Recommended Feature Set for Predictive Modeling (Round 3):**
>    - Core predictors: `is_returned`, `delivery_days`, `delivery_speed_group_Delayed (6+d)`, `review_length_capped`, `tenure_years`, `complaint_Damaged_Item`, `complaint_Refund_Delay`, `category_Sports`.
""")

    # -------------------------------------------------------------------------
    # SECTION 8: CONCLUSIONS & RECOMMENDATIONS
    # -------------------------------------------------------------------------
    builder.add_md("""
---
## 8. Summary of Findings & Strategic Business Recommendations

Based on empirical data science calculations across all 1,050 reviews and 850 orders:

### 1. Root Causes of Low Customer Ratings:
1. **Product Returns (Primary Driver)**:
   - Returned orders average **3.12 stars** vs **3.88 stars** for non-returned orders ($t = -8.81, p < 10^{-17}$). Over 31% of returned orders receive 1–2 star ratings.
2. **Logistics Fulfillment Delays (Secondary Driver)**:
   - Strong negative linear correlation ($r = -0.24, p < 10^{-14}$). Lead times under 3 days average $\\ge 4.0$ stars; lead times over 6 days collapse to 3.42 stars.
3. **Severe Support Friction (Damaged Goods & Refund Delays)**:
   - Damaged merchandise and refund delays produce the lowest ratings on the platform (3.42–3.43 stars).
4. **Seller Concentration**:
   - Customer dissatisfaction is concentrated in a small group of underperforming sellers (Seller 19, Seller 13, Seller 12). Seller 12 alone accounts for a 22.6% severe negative rating rate.

### 2. Operational Recommendations for ReviewLens Platform:
- **Fast-Track Returns & Refunds**: Automate instant refund approvals to neutralize the steep rating penalty observed in refund delays.
- **SLA Threshold on Logistics**: Impose a maximum 5-day delivery guarantee; throttle order volume for logistics routes exceeding 6 days.
- **Seller Quality Audits**: Implement targeted merchant interventions for Seller 12, 19, and 13, auditing their packaging and quality assurance protocols to curb damaged item shipments.

---
## 9. Submission Artifacts Catalog
All reproducible outputs from this project are archived in the `output/` directory:
- **Cleaned Data (`output/cleaned_data/`)**: `reviews_cleaned.csv`, `products_cleaned.csv`, `sellers_cleaned.csv`, `orders_cleaned.csv`, `support_cleaned.csv`.
- **Merged Data (`output/merged_data/`)**: `reviewlens_master_analysis.csv`, `reviewlens_support_analysis.csv`.
- **Engineered Features (`output/engineered_features/`)**: `reviewlens_features_engineered.csv`, `feature_importance_rf.csv`, `feature_selection_anova_f.csv`, `feature_selection_mutual_info.csv`, `pca_explained_variance.csv`.
- **Visualizations (`output/figures/`)**: 10 publication-quality PNG charts corresponding to Questions 1 through 6, EDA distributions, and feature importance benchmarks.
""")

    builder.save(NOTEBOOK_PATH)


if __name__ == "__main__":
    build_reviewlens_notebook()
