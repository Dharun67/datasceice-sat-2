# ReviewLens Commerce Analytics
## DS Sprint Skill-Based Assessment — Use Case 26: Product Review Insight
### Round 2: Exploratory Data Analysis & Feature Engineering

---

## 📌 Executive Summary

**ReviewLens** is an end-to-end commerce intelligence pipeline designed to extract operational and customer satisfaction insights from multi-relational e-commerce data. 

In this Round 2 assessment, we analyze the relationship between customer reviews, catalog metadata, merchant operations, delivery fulfillment, and post-purchase customer support tickets across five interconnected datasets:
1. `reviews.csv` (1,052 raw records $\rightarrow$ 1,050 cleaned records)
2. `products.csv` (180 products across 6 standardized categories)
3. `sellers.csv` (28 merchants across 4 geographic regions)
4. `orders.csv` (850 customer orders)
5. `support.csv` (290 customer support tickets across 243 unique orders)

---

## ⚠️ Synthetic Practice Data Notice

> **Official Competition & Assessment Disclaimer:**
> The dataset in `ReviewLens_Round2_Practice_Dataset/` is a **synthetic practice dataset** generated to reflect the schema, data-quality challenges, and entity relationships outlined in the DS Sprint Use Case 26 specification.
> - It contains intentional real-world anomalies (missing ratings, duplicated reviews, casing inconsistencies, extreme length outliers, and foreign key mismatches).
> - It contains no real customer personally identifiable information (PII).
> - All analytical workflows, scripts, and notebooks in this project are **100% modular and reproducible**. When official competition data is provided, simply point the pipeline to the new directory to execute the exact same analysis.

---

## 📂 Repository Structure

```
DATASAT2/
├── ReviewLens_Round2_EDA_FeatureEngineering.ipynb  # Primary Assessment Jupyter Notebook (Executed with all outputs)
├── README.md                                       # Comprehensive project documentation (This file)
├── ReviewLens_Round2_Practice_Dataset/             # Raw practice datasets (Preserved untouched)
│   ├── reviews.csv                                 # Customer ratings, review texts, lengths, dates
│   ├── products.csv                                # Catalog taxonomy, brands, price bands
│   ├── sellers.csv                                 # Merchant names, regions, tenure
│   ├── orders.csv                                  # Delivery days, discounts, return flags
│   ├── support.csv                                 # Complaint categories, resolutions
│   └── README.txt                                  # Organizer dataset note
├── src/                                            # Modular Python pipeline source code
│   ├── __init__.py                                 # Package marker
│   ├── data_cleaning.py                            # Data audit & documented cleaning transformations
│   ├── data_merging.py                             # PK/FK validation & join trap prevention
│   ├── eda_and_analysis.py                         # Statistical tests & visualization generator
│   ├── feature_engineering.py                      # Feature creation, encodings, selection & PCA
│   ├── build_notebook.py                           # Automated notebook generator & execution runner
│   └── run_pipeline.py                             # End-to-end pipeline execution runner
└── output/                                         # Generated assessment artifacts
    ├── cleaned_data/                               # Documented cleaned datasets
    │   ├── reviews_cleaned.csv                     # De-duplicated, reconciled product IDs, text metrics
    │   ├── products_cleaned.csv                    # Normalized categories and trimmed strings
    │   ├── sellers_cleaned.csv                     # Trimmed merchant names and regions
    │   ├── orders_cleaned.csv                      # Standardized return flags and numeric fields
    │   └── support_cleaned.csv                     # Standardized complaint categories and resolutions
    ├── merged_data/                                # Relational datasets (0 duplicate rows)
    │   ├── reviewlens_master_analysis.csv          # 1-row-per-review merged dataset (1,050 rows)
    │   └── reviewlens_support_analysis.csv         # Ticket-level merged dataset (290 rows)
    ├── engineered_features/                        # Feature engineering & selection tables
    │   ├── reviewlens_features_engineered.csv      # Master dataset with 57 engineered columns
    │   ├── feature_importance_rf.csv               # Random Forest feature importance scores
    │   ├── feature_selection_anova_f.csv           # Univariate ANOVA F-scores and p-values
    │   ├── feature_selection_mutual_info.csv       # Non-linear Mutual Information classification scores
    │   └── pca_explained_variance.csv              # PCA explained & cumulative variance table
    └── figures/                                    # 10 publication-quality visualization charts (PNG)
        ├── 01_category_rating_comparison.png       # Q1: Category vs Rating bar chart
        ├── 02_verified_vs_unverified_ratings.png   # Q2: Verified vs Unverified boxplot & distribution
        ├── 03_delivery_time_vs_rating_trend.png    # Q3: Delivery time vs Rating regression trend
        ├── 04_seller_recurring_low_ratings.png     # Q4: Sellers with recurring low ratings
        ├── 05_returned_orders_rating_impact.png    # Q5: Return flag violin plot & low rating %
        ├── 06_complaint_categories_low_ratings.png # Q6: Support complaint ratings bar chart
        ├── 07_rating_distribution_and_classes.png  # EDA: Overall rating histogram & sentiment share
        ├── 08_review_length_distribution_and_outliers.png # EDA: Review length & outlier scatter plot
        ├── 09_feature_importances_comparison.png   # Selection: RF importance vs ANOVA F-score
        └── 10_correlation_matrix_heatmap.png       # Multivariate Pearson correlation heatmap
```

---

## 🛠️ Data Quality Audit & Documented Cleaning Decisions

| Issue Identified | Affected Dataset | Cleaning Action Taken | Rationale & Justification |
| :--- | :--- | :--- | :--- |
| **Exact Duplicate Rows** | `reviews.csv` | Dropped 2 exact duplicate rows (`R00006`, `R00016` duplicated at indices 1050 and 1051), retaining the first occurrence. Row count adjusted from 1,052 to 1,050. | Duplicate reviews artificially inflate sample counts and distort variance estimates. |
| **Missing Ratings** | `reviews.csv` | Retained 3 reviews (`R00012`, `R00038`, `R00102`) with an explicit boolean flag `rating_is_missing = True`. When computing target rating statistics or training models, filtered dynamically using `rating.notna()`. | Never fabricate or invent ground truth rating scores. Retaining them preserves complete transactional auditability. |
| **Inconsistent Taxonomy Labels** | `products.csv` | Stripped leading/trailing whitespace (`str.strip()`) and mapped canonical Title Case: `'electronics '` $\rightarrow$ `'Electronics'`, `'HOME & KITCHEN'` $\rightarrow$ `'Home & Kitchen'`. Result: Exactly 6 distinct categories. | Fixes categorization errors that would otherwise fragment category aggregations into spurious duplicate buckets. |
| **Extreme Length Outliers** | `reviews.csv` | Identified artificial character counts of 5,000 (`R00023`) and 8,000 (`R00073`), where actual texts were only 35 and 42 characters. Derived `actual_char_count` and `actual_word_count`, created `review_length_capped` (99th percentile winsorization at 414.5 chars), and flagged `is_extreme_length_outlier`. | Prevents artificial scale distortion in distance-based algorithms and regression weights while capturing authentic text surface properties. |
| **Unmatched Foreign Keys** | `reviews.csv` | Discovered product IDs `P9998` (order `O00420`) and `P9999` (order `O00470`) not present in `products.csv`. Cross-referenced `orders.csv` to resolve true products: `O00420` $\rightarrow$ `P0136`, `O00470` $\rightarrow$ `P0083`. Preserved `raw_product_id` and flagged `is_reconciled_product_id`. | Restores relational integrity without data loss, linking reviews to their verified product records. |

---

## 🔗 Entity-Relationship Integrity & Preventing Join Traps

### Primary Key Validation
- `reviews.review_id`: 1,050 unique values (100% unique post-deduplication)
- `products.product_id`: 180 unique values (100% unique)
- `sellers.seller_id`: 28 unique values (100% unique)
- `orders.order_id`: 850 unique values (100% unique)
- `support.support_id`: 290 unique values (100% unique)

### Foreign Key Integrity
- `orders.product_id` $\rightarrow$ `products.product_id`: 100% match (0 unmatched)
- `orders.seller_id` $\rightarrow$ `sellers.seller_id`: 100% match (0 unmatched)
- `reviews.order_id` $\rightarrow$ `orders.order_id`: 100% match (0 unmatched)
- `reviews.product_id` $\rightarrow$ `products.product_id`: 100% match (0 unmatched after reconciliation)
- `support.order_id` $\rightarrow$ `orders.order_id`: 100% match (0 unmatched)

### The Join Trap: Preventing Cartesian Row Multiplication
In `support.csv`, 243 unique orders generated 290 tickets:
- **42 orders have multiple support tickets** (up to 4 tickets for a single order).
- **500+ orders have zero support tickets**.

**The Join Trap:** If a naive `merge(reviews, support, on='order_id')` is executed:
- An inner join eliminates >70% of customer reviews.
- A left join multiplies reviews for multi-ticket orders, corrupting rating distributions.

**The Solution:**
1. **Aggregated Order-Level Support Summary**: In `data_merging.py`, support is aggregated per `order_id` before merging:
   - `support_ticket_count` (0 if no ticket)
   - `has_support_ticket` (0 or 1)
   - `primary_complaint_category`
   - `latest_resolution` ('Resolved', 'Pending', 'Escalated')
   - Binary indicator columns for individual complaint types
2. **Master Analysis Dataset**: Exactly **1,050 records** (retaining 1 row per review, 0 duplicated rows).
3. **Dedicated Ticket-Level Dataset**: `reviewlens_support_analysis.csv` (290 rows) is preserved separately for ticket-specific complaint analyses.

---

## 📊 Answers to the Six Core Assessment Questions

All findings are derived from real calculations on the cleaned datasets:

### Question 1: Which categories receive lower ratings?
- **Sports** receives the lowest average rating at **3.63 stars** (median 4.0, std 1.09, 14.1% ratings $\le 2$ stars).
- **Books** is second lowest at **3.74 stars** (median 4.0, std 0.97, 10.4% ratings $\le 2$ stars).
- **Beauty**: 3.76 stars (median 4.0, std 1.11).
- **Electronics**: 3.78 stars (median 4.0, std 1.10).
- **Home & Kitchen**: 3.80 stars (median 4.0, std 1.04).
- **Fashion**: **3.86 stars** (median 4.0, std 0.94, highest satisfaction).
- **Statistical Significance**: One-way ANOVA yields $F = 1.054, p = 0.385$, indicating that while category-level differences exist, category type alone is not the primary driver of customer dissatisfaction.

### Question 2: Do verified reviews differ?
- **Verified Purchase Reviews (`Yes`)**: Mean = **3.757 stars** (median 4.0, std 1.055, n = 798).
- **Unverified Reviews (`No`)**: Mean = **3.759 stars** (median 4.0, std 1.046, n = 249).
- **Difference**: $\Delta = -0.002$ stars.
- **Statistical Significance**:
  - Independent Two-Sample t-test: $t = -0.0281, p = 0.9776$.
  - Mann-Whitney U test: $U = 99,649.0, p = 0.9407$.
- **Conclusion**: There is **no statistically significant difference** in review ratings between verified and unverified purchases in this dataset.

### Question 3: How does delivery time relate to rating?
- **Correlation**:
  - Pearson correlation: $r = -0.2409, p = 2.74 \times 10^{-15}$ (Strong negative linear relationship).
  - Spearman rank correlation: $\rho = -0.2311, p = 3.73 \times 10^{-14}$.
- **Performance by Delivery Speed Tier**:
  - **Express (1–2 days)**: Mean = **4.10 stars** (std 0.85, n = 165).
  - **Standard (3–5 days)**: Mean = **3.83 stars** (std 1.03, n = 568).
  - **Delayed (6+ days)**: Mean = **3.42 stars** (std 1.09, n = 314).
  - Extreme lead times (11–12 days) drop to **1.50–2.00 stars**.
- **Conclusion**: Extended delivery times have a severe, compounding negative effect on customer reviews.

### Question 4: Which sellers show recurring low ratings?
Sellers ranked by lowest average rating and highest low-rating share ($\le 2$ stars):
1. **Seller 19 (East)**: Mean = **3.43 stars** (median 4.0, 14.3% low ratings, tenure 9y).
2. **Seller 13 (South)**: Mean = **3.50 stars** (median 3.0, 14.3% low ratings, tenure 5y).
3. **Seller 12 (Central)**: Mean = **3.51 stars** (median 4.0, **22.6% low ratings**, tenure 8y).
4. **Seller 09 (East)**: Mean = **3.58 stars** (median 4.0, 17.8% low ratings, tenure 9y).
5. **Seller 15 (South)**: Mean = **3.59 stars** (median 4.0, 17.9% low ratings, tenure 10y).
6. **Seller 01 (North)**: Mean = **3.62 stars** (median 4.0, 17.8% low ratings, tenure 5y).
7. **Seller 02 (Central)**: Mean = **3.63 stars** (median 4.0, 14.3% low ratings, tenure 12y).
- **Takeaway**: Seller 12 requires urgent quality intervention, as nearly 1 in 4 orders results in a 1 or 2-star rating.

### Question 5: Do returned orders have lower ratings?
- **Non-Returned Orders (`No`)**: Mean = **3.88 stars** (median 4.0, std 0.99, n = 883, only 8.7% low ratings).
- **Returned Orders (`Yes`)**: Mean = **3.12 stars** (median 3.0, std 1.15, n = 164, **31.1% low ratings**).
- **Rating Penalty**: **$-0.76$ stars**.
- **Statistical Significance**:
  - Independent Two-Sample t-test: $t = -8.8069, p = 5.25 \times 10^{-18}$.
  - Effect Size: **Cohen's $d = -0.7489$** (Large negative effect).
- **Conclusion**: Product return status is the single most decisive factor predicting low review ratings.

### Question 6: What complaint categories align with low ratings?
Customer dissatisfaction is sharply concentrated in post-delivery operational failures:
1. **Refund Delay**: Mean = **3.42 stars** (median 3.0, std 1.16, n = 52, 17.3% low ratings).
2. **Damaged Item**: Mean = **3.43 stars** (median 3.0, std 1.09, n = 42, **21.4% low ratings**).
3. **Wrong Item**: Mean = **3.56 stars** (median 4.0, std 1.16, n = 25, 16.0% low ratings).
4. **Quality Issue**: Mean = **3.79 stars** (median 4.0, std 1.06, n = 77, 10.4% low ratings).
5. **Late Delivery**: Mean = **3.88 stars** (median 4.0, std 1.01, n = 58, 8.6% low ratings).
6. **Other**: Mean = **4.06 stars** (median 4.0, std 0.89, n = 34, 8.8% low ratings).
- **Takeaway**: Financial friction (**Refund Delay**) and physical damage (**Damaged Item**) cause the deepest brand damage and lowest customer sentiment.

---

## ⚙️ Feature Engineering & Selection

### 1. Engineered Features
- `review_length_group`: Binned into `'Short (<150)'`, `'Medium (150-300)'`, `'Long (>300)'`.
- `rating_class`: 3-tier customer sentiment (`'Positive'`, `'Neutral'`, `'Negative'`).
- `is_low_rating`: Binary target indicator ($1$ if rating $\le 2$, else $0$).
- `delivery_speed_group`: Binned lead times (`'Express (1-2d)'`, `'Standard (3-5d)'`, `'Delayed (6+d)'`).
- `discount_band`: Promotional tiers (`'No Discount (0%)'`, `'Low (5-10%)'`, `'Medium (15-20%)'`, `'High (25-30%)'`).
- Temporal features: `review_year`, `review_month`, `review_month_name`, `review_day_of_week`, `review_quarter`, `is_weekend`.
- Text surface characteristics: `actual_char_count`, `actual_word_count`, `avg_word_length`, `exclamation_count`.
- Merchant tenure group: `seller_tenure_group` (`'New (<=3y)'`, `'Established (4-7y)'`, `'Veteran (>7y)'`).

### 2. Categorical Encoding
- **One-Hot Encoding**: Applied to nominal variables (`category`, `region`, `delivery_speed_group`, `discount_band`, `primary_complaint_category`, `seller_tenure_group`) using `pd.get_dummies(drop_first=True)`.
- **Ordinal Encoding**: Applied to ordered categories (`price_band_ordinal`: Budget = 1, Mid-range = 2, Premium = 3).
- **Binary Encoding**: Converted flags to numeric 0/1 (`is_returned`, `is_verified`, `has_support_ticket`, `is_weekend`).

### 3. Feature Selection Results
Target: Predicting severe customer dissatisfaction (`is_low_rating = 1`).

| Rank | Random Forest Importance (MDI) | ANOVA F-Statistic (Linear) | Mutual Information (Non-Linear) |
| :---: | :--- | :--- | :--- |
| **1** | `review_length_capped` (0.190) | `is_returned` ($F = 68.65, p < 10^{-15}$) | `is_returned` (0.033) |
| **2** | `actual_char_count` (0.089) | `delivery_days` ($F = 41.35, p < 10^{-9}$) | `category_Books` (0.023) |
| **3** | `tenure_years` (0.081) | `delivery_speed_group_Delayed` ($F = 24.13$) | `delivery_days` (0.021) |
| **4** | `delivery_days` (0.081) | `category_Sports` ($F = 3.50, p = 0.062$) | `delivery_speed_group_Delayed` (0.019) |
| **5** | `is_returned` (0.068) | `category_Fashion` ($F = 3.45, p = 0.063$) | `first_complaint_Other` (0.018) |

### 4. Dimensionality Reduction (PCA) & Method Limitations
We evaluated Principal Component Analysis on standardized continuous metrics:
- Component 1: 25.16% variance explained
- Component 2: 13.16% variance explained
- Cumulative variance across first 6 components: 82.52%
- Reaching the standard 90% variance threshold requires **7 components out of 9 original features**.

**Why Tree & Filter Feature Selection is Preferred over PCA:**
1. **Business Interpretability**: PCA constructs abstract linear combinations of features ($PC1 = 0.52 \cdot \text{length} - 0.15 \cdot \text{delivery\_days}$), obscuring actionable levers for merchants.
2. **Sparsity & Categorical Data**: One-hot encoded business flags violate PCA's multivariate normality assumption.
3. **Actionability**: Tree-based and ANOVA selection explicitly identify `is_returned` and `delivery_days` as the dominant drivers of customer churn, enabling direct SLA improvements.

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Required packages:
  ```bash
  pip install pandas numpy matplotlib seaborn scipy scikit-learn nbformat
  ```

### Step 1: Run Full Modular Pipeline
Execute all cleaning, merging, statistical analysis, feature engineering, and verification stages with a single command:
```bash
python src/run_pipeline.py
```
This runs in ~10 seconds and verifies all 22 output files in `output/`.

### Step 2: Open and Run the Jupyter Notebook
Open the notebook in JupyterLab, VS Code, or Google Colab:
```bash
jupyter notebook ReviewLens_Round2_EDA_FeatureEngineering.ipynb
```
The notebook is already fully pre-computed with all tables, markdown commentary, and embedded charts.

---

## 📋 Assessment Submission Checklist

- [x] **Requirement 1**: Load all 5 datasets using Pandas.
- [x] **Requirement 2**: Inspect shapes, data types, descriptive statistics, missing values, and duplicate records.
- [x] **Requirement 3**: Identify missing ratings, duplicate reviews, inconsistent categories, extreme review lengths, and unmatched product IDs.
- [x] **Requirement 4**: Clean datasets with documented cleaning decisions; preserve raw data files untouched.
- [x] **Requirement 5**: Validate primary and foreign keys; eliminate join duplication traps.
- [x] **Requirement 6**: Create merged analysis dataset (1,050 rows, 0 duplicate rows).
- [x] **Requirement 7**: Perform descriptive analysis with clear Matplotlib & Seaborn visualizations.
- [x] **Requirement 8**: Answer all six assessment questions with actual computations and statistical tests.
- [x] **Requirement 9**: Engineer domain features (length groups, rating classes, delivery tiers, calendar metrics, discount bands).
- [x] **Requirement 10**: Encode categorical variables (One-Hot, Ordinal, Binary).
- [x] **Requirement 11**: Perform feature selection (Random Forest, Mutual Info, ANOVA F-test) and PCA with limitation discussion.
- [x] **Requirement 12**: Professional Jupyter Notebook with headings, outputs, visualizations, and written interpretations.
- [x] **Requirement 13**: Save all cleaned data, merged tables, engineered features, figures, and selection results in `output/`.
- [x] **Requirement 14**: Detailed, reproducible README documentation.
- [x] **Requirement 15**: Verified end-to-end execution with zero runtime errors.
