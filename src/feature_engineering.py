"""
ReviewLens Commerce Analytics - DS Sprint Use Case 26
Round 2: EDA and Feature Engineering
Module: Feature Engineering, Encoding & Feature Selection (feature_engineering.py)

This module engineers domain-specific features, performs categorical encoding,
applies multiple feature selection techniques (Random Forest, Mutual Information, ANOVA F-test),
evaluates dimensionality reduction (PCA), and saves all engineered outputs and figures.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def engineer_and_select_features(master_csv_path: str, output_dir: str, figures_dir: str):
    """
    Builds engineered features, performs encodings, runs feature selection and PCA.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    df = pd.read_csv(master_csv_path)
    df['review_date'] = pd.to_datetime(df['review_date'])
    
    print("======================================================================")
    print("        FEATURE ENGINEERING & DIMENSIONALITY REDUCTION PIPELINE        ")
    print("======================================================================\n")
    
    # -------------------------------------------------------------------------
    # 1. ENGINEER MEANINGFUL FEATURES
    # -------------------------------------------------------------------------
    # A. Review Length Group
    df['review_length_group'] = pd.cut(
        df['review_length_capped'],
        bins=[-1, 150, 300, 10000],
        labels=['Short (<150)', 'Medium (150-300)', 'Long (>300)']
    )
    
    # B. Rating Class & Binary Target
    def get_rating_class(r):
        if pd.isna(r):
            return np.nan
        elif r <= 2:
            return 'Negative'
        elif r == 3:
            return 'Neutral'
        else:
            return 'Positive'
            
    df['rating_class'] = df['rating'].apply(get_rating_class)
    df['is_low_rating'] = df['rating'].apply(lambda r: 1 if (pd.notna(r) and r <= 2) else (np.nan if pd.isna(r) else 0))
    
    # C. Delivery Speed Group
    df['delivery_speed_group'] = pd.cut(
        df['delivery_days'],
        bins=[0, 2, 5, 50],
        labels=['Express (1-2d)', 'Standard (3-5d)', 'Delayed (6+d)']
    )
    
    # D. Review Calendar / Temporal Features
    df['review_year'] = df['review_date'].dt.year
    df['review_month'] = df['review_date'].dt.month
    df['review_month_name'] = df['review_date'].dt.strftime('%b')
    df['review_day_of_week'] = df['review_date'].dt.day_name()
    df['review_quarter'] = df['review_date'].dt.quarter
    df['is_weekend'] = df['review_date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # E. Discount Bands
    df['discount_band'] = pd.cut(
        df['discount_pct'],
        bins=[-1, 0, 10, 20, 100],
        labels=['No Discount (0%)', 'Low (5-10%)', 'Medium (15-20%)', 'High (25-30%)']
    )
    
    # F. Seller Tenure Group
    df['seller_tenure_group'] = pd.cut(
        df['tenure_years'],
        bins=[-1, 3, 7, 50],
        labels=['New (<=3y)', 'Established (4-7y)', 'Veteran (>7y)']
    )
    
    # G. Ordinal Price Band
    price_order = {'Budget': 1, 'Mid-range': 2, 'Premium': 3}
    df['price_band_ordinal'] = df['price_band'].map(price_order)
    
    # H. Text Surface Characteristics
    df['avg_word_length'] = (df['actual_char_count'] / df['actual_word_count'].replace(0, 1)).round(2)
    df['exclamation_count'] = df['review_text'].astype(str).str.count('!')
    
    print("Features engineered successfully.")
    print(f"Total columns in feature dataset: {df.shape[1]}")
    
    # Save Full Engineered Dataset
    df.to_csv(os.path.join(output_dir, "reviewlens_features_engineered.csv"), index=False)
    
    # -------------------------------------------------------------------------
    # 2. ENCODING CATEGORICAL VARIABLES FOR MODELING / SELECTION
    # -------------------------------------------------------------------------
    # Filter records with non-null target rating
    modeling_df = df[df['rating'].notna()].copy()
    modeling_df['is_low_rating'] = modeling_df['is_low_rating'].astype(int)
    
    nominal_cols = [
        'category', 
        'delivery_speed_group', 
        'discount_band', 
        'region', 
        'primary_complaint_category',
        'seller_tenure_group'
    ]
    
    encoded_dummies = pd.get_dummies(modeling_df[nominal_cols], drop_first=True, dtype=int)
    
    numerical_feature_cols = [
        'delivery_days', 
        'discount_pct', 
        'tenure_years', 
        'is_returned', 
        'is_verified', 
        'review_length_capped', 
        'actual_char_count', 
        'actual_word_count', 
        'avg_word_length',
        'price_band_ordinal',
        'support_ticket_count', 
        'has_support_ticket', 
        'has_escalated_support', 
        'has_pending_support',
        'is_weekend'
    ]
    
    X = pd.concat([modeling_df[numerical_feature_cols], encoded_dummies], axis=1)
    y = modeling_df['is_low_rating']
    
    print(f"Prepared Feature Matrix X shape: {X.shape}")
    print(f"Target distribution (is_low_rating): {np.bincount(y)}")
    
    # -------------------------------------------------------------------------
    # 3. FEATURE SELECTION
    # -------------------------------------------------------------------------
    # Technique 1: Random Forest Feature Importance (MDI / Gini Importance)
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    rf_importance.to_csv(os.path.join(output_dir, "feature_importance_rf.csv"), index=False)
    
    # Technique 2: Mutual Information Classification (Non-linear dependencies)
    mi = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mutual_info_score': mi
    }).sort_values('mutual_info_score', ascending=False)
    mi_df.to_csv(os.path.join(output_dir, "feature_selection_mutual_info.csv"), index=False)
    
    # Technique 3: ANOVA F-test (Linear variance between groups)
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    anova_df = pd.DataFrame({
        'feature': X.columns,
        'f_score': selector.scores_,
        'p_value': selector.pvalues_
    }).sort_values('f_score', ascending=False)
    anova_df.to_csv(os.path.join(output_dir, "feature_selection_anova_f.csv"), index=False)
    
    # -------------------------------------------------------------------------
    # 4. DIMENSIONALITY REDUCTION (PCA) & LIMITATION ANALYSIS
    # -------------------------------------------------------------------------
    # Scale continuous numerical features
    scaler = StandardScaler()
    continuous_features = [
        'delivery_days', 'discount_pct', 'tenure_years', 
        'review_length_capped', 'actual_char_count', 'actual_word_count', 
        'avg_word_length', 'price_band_ordinal', 'support_ticket_count'
    ]
    X_cont_scaled = scaler.fit_transform(modeling_df[continuous_features])
    
    pca = PCA()
    pca.fit(X_cont_scaled)
    
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    
    pca_df = pd.DataFrame({
        'component': [f"PC{i+1}" for i in range(len(explained_var))],
        'explained_variance_ratio': explained_var,
        'cumulative_variance_ratio': cum_var
    })
    pca_df.to_csv(os.path.join(output_dir, "pca_explained_variance.csv"), index=False)
    
    print("\nPCA on Continuous Features:")
    print(pca_df.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 5. VISUALIZATIONS: FEATURE SELECTION & CORRELATION HEATMAP
    # -------------------------------------------------------------------------
    # Figure 09: Top 10 Feature Importances Comparison (RF vs ANOVA F-Score)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    top_rf = rf_importance.head(10)
    sns.barplot(data=top_rf, x='rf_importance', y='feature', hue='feature', palette='mako', legend=False, ax=ax1, edgecolor='#333333')
    ax1.set_title("Top 10 Features: Random Forest Importance", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Relative Importance Score", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Feature Name", fontsize=10, fontweight='bold')
    
    top_f = anova_df.head(10)
    sns.barplot(data=top_f, x='f_score', y='feature', hue='feature', palette='rocket', legend=False, ax=ax2, edgecolor='#333333')
    ax2.set_title("Top 10 Features: ANOVA F-Statistic", fontsize=12, fontweight='bold')
    ax2.set_xlabel("ANOVA F-Score", fontsize=10, fontweight='bold')
    ax2.set_ylabel("", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "09_feature_importances_comparison.png"))
    plt.close()
    
    # Figure 10: Correlation Matrix Heatmap
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    corr_cols = [
        'rating', 'delivery_days', 'discount_pct', 'tenure_years', 
        'is_returned', 'is_verified', 'has_support_ticket', 
        'review_length_capped', 'price_band_ordinal'
    ]
    corr_matrix = modeling_df[corr_cols].corr()
    
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
    plt.savefig(os.path.join(figures_dir, "10_correlation_matrix_heatmap.png"))
    plt.close()
    
    print("\nFeature engineering and selection outputs saved successfully!")
    return df, X, y, rf_importance, anova_df, pca_df


if __name__ == "__main__":
    master_csv = r"c:\Users\dharu\OneDrive\Desktop\DATASAT2\output\merged_data\reviewlens_master_analysis.csv"
    feat_out = r"c:\Users\dharu\OneDrive\Desktop\DATASAT2\output\engineered_features"
    figs_out = r"c:\Users\dharu\OneDrive\Desktop\DATASAT2\output\figures"
    engineer_and_select_features(master_csv, feat_out, figs_out)
