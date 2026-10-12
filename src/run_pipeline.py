"""
ReviewLens Commerce Analytics - DS Sprint Use Case 26
Round 2: EDA and Feature Engineering
Master Pipeline Runner (run_pipeline.py)

Executes all pipeline stages:
1. Data Cleaning
2. Data Merging & Key Validation
3. Exploratory Data Analysis & Statistical Testing (Q1-Q6)
4. Feature Engineering, Encoding, Selection & PCA
5. Output Verification
"""

import os
import sys
import time

# Add src directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from data_cleaning import clean_reviewlens_data
from data_merging import merge_reviewlens_data
from eda_and_analysis import run_eda_and_answer_questions
from feature_engineering import engineer_and_select_features


def main():
    start_time = time.time()
    print("================================================================================")
    print("      STARTING REVIEWLENS ROUND 2: EDA & FEATURE ENGINEERING PIPELINE           ")
    print("================================================================================\n")
    
    raw_data_dir = os.path.join(BASE_DIR, "ReviewLens_Round2_Practice_Dataset")
    output_dir = os.path.join(BASE_DIR, "output")
    cleaned_dir = os.path.join(output_dir, "cleaned_data")
    merged_dir = os.path.join(output_dir, "merged_data")
    features_dir = os.path.join(output_dir, "engineered_features")
    figures_dir = os.path.join(output_dir, "figures")
    
    # Stage 1: Data Cleaning
    print("\n--- STAGE 1: DATA CLEANING ---")
    clean_reviewlens_data(raw_data_dir, cleaned_dir)
    
    # Stage 2: Data Merging
    print("\n--- STAGE 2: DATA MERGING & KEY VALIDATION ---")
    master_df, support_df = merge_reviewlens_data(cleaned_dir, merged_dir)
    
    # Stage 3: EDA & Hypothesis Testing (Questions 1 to 6)
    print("\n--- STAGE 3: EXPLORATORY DATA ANALYSIS & HYPOTHESIS TESTING ---")
    master_csv_path = os.path.join(merged_dir, "reviewlens_master_analysis.csv")
    eda_results = run_eda_and_answer_questions(master_csv_path, figures_dir)
    
    # Stage 4: Feature Engineering, Encoding & Selection
    print("\n--- STAGE 4: FEATURE ENGINEERING & SELECTION ---")
    engineer_and_select_features(master_csv_path, features_dir, figures_dir)
    
    # Stage 5: Verification & Quality Assurance
    print("\n--- STAGE 5: ARTIFACT & OUTPUT VERIFICATION ---")
    expected_files = [
        os.path.join(cleaned_dir, "reviews_cleaned.csv"),
        os.path.join(cleaned_dir, "products_cleaned.csv"),
        os.path.join(cleaned_dir, "sellers_cleaned.csv"),
        os.path.join(cleaned_dir, "orders_cleaned.csv"),
        os.path.join(cleaned_dir, "support_cleaned.csv"),
        os.path.join(merged_dir, "reviewlens_master_analysis.csv"),
        os.path.join(merged_dir, "reviewlens_support_analysis.csv"),
        os.path.join(features_dir, "reviewlens_features_engineered.csv"),
        os.path.join(features_dir, "feature_importance_rf.csv"),
        os.path.join(features_dir, "feature_selection_anova_f.csv"),
        os.path.join(features_dir, "feature_selection_mutual_info.csv"),
        os.path.join(features_dir, "pca_explained_variance.csv"),
        os.path.join(figures_dir, "01_category_rating_comparison.png"),
        os.path.join(figures_dir, "02_verified_vs_unverified_ratings.png"),
        os.path.join(figures_dir, "03_delivery_time_vs_rating_trend.png"),
        os.path.join(figures_dir, "04_seller_recurring_low_ratings.png"),
        os.path.join(figures_dir, "05_returned_orders_rating_impact.png"),
        os.path.join(figures_dir, "06_complaint_categories_low_ratings.png"),
        os.path.join(figures_dir, "07_rating_distribution_and_classes.png"),
        os.path.join(figures_dir, "08_review_length_distribution_and_outliers.png"),
        os.path.join(figures_dir, "09_feature_importances_comparison.png"),
        os.path.join(figures_dir, "10_correlation_matrix_heatmap.png")
    ]
    
    all_ok = True
    for f in expected_files:
        if os.path.exists(f) and os.path.getsize(f) > 0:
            print(f"[OK] {os.path.relpath(f, BASE_DIR)} ({os.path.getsize(f)} bytes)")
        else:
            print(f"[FAILED] Missing or empty: {os.path.relpath(f, BASE_DIR)}")
            all_ok = False
            
    elapsed = time.time() - start_time
    print(f"\nPipeline finished in {elapsed:.2f} seconds. All artifacts verified: {all_ok}")
    return all_ok


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
