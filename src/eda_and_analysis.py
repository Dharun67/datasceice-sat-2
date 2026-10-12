"""
ReviewLens Commerce Analytics - DS Sprint Use Case 26
Round 2: EDA and Feature Engineering
Module: Exploratory Data Analysis & Question Answering (eda_and_analysis.py)

This module performs rigorous descriptive and statistical analysis on the merged dataset,
answers the six core business questions using actual data calculations and statistical tests,
and exports publication-quality Matplotlib/Seaborn visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure visualization aesthetic
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def run_eda_and_answer_questions(master_csv_path: str, figures_output_dir: str):
    """
    Executes full EDA, tests hypotheses, and produces saved figures.
    """
    os.makedirs(figures_output_dir, exist_ok=True)
    df = pd.read_csv(master_csv_path)
    df['review_date'] = pd.to_datetime(df['review_date'])
    
    # Exclude missing ratings for rating-specific calculations
    df_rated = df[df['rating'].notna()].copy()
    
    results = {}
    print("======================================================================")
    print("           REVIEWLENS ROUND 2: EDA & STATISTICAL ANALYSIS             ")
    print("======================================================================\n")
    
    # -------------------------------------------------------------------------
    # QUESTION 1: Which categories receive lower ratings?
    # -------------------------------------------------------------------------
    print(">>> QUESTION 1: Which categories receive lower ratings? <<<")
    cat_summary = df_rated.groupby('category')['rating'].agg(
        review_count='count',
        mean_rating='mean',
        median_rating='median',
        std_rating='std',
        low_rating_pct=lambda x: (x <= 2).mean() * 100
    ).sort_values('mean_rating')
    print(cat_summary.to_string())
    
    
    cat_groups = [group['rating'].values for _, group in df_rated.groupby('category')]
    f_stat, f_pval = stats.f_oneway(*cat_groups)
    print(f"\nOne-way ANOVA across product categories: F = {f_stat:.4f}, p = {f_pval:.4f}")
    
    results['Q1_table'] = cat_summary
    results['Q1_anova'] = (f_stat, f_pval)
    
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    palette = sns.color_palette("mako", n_colors=len(cat_summary))
    bars = sns.barplot(
        x=cat_summary.index, 
        y=cat_summary['mean_rating'], 
        hue=cat_summary.index,
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
        height = p.get_height()
        ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height + 0.02),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_output_dir, "01_category_rating_comparison.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # QUESTION 2: Do verified reviews differ?
    # -------------------------------------------------------------------------
    print("\n>>> QUESTION 2: Do verified reviews differ? <<<")
    ver_summary = df_rated.groupby('verified_flag')['rating'].agg(
        review_count='count',
        mean_rating='mean',
        median_rating='median',
        std_rating='std',
        low_rating_pct=lambda x: (x <= 2).mean() * 100
    )
    print(ver_summary.to_string())
    
    t_stat_ver, p_val_ver = stats.ttest_ind(
        df_rated[df_rated['verified_flag'] == 'Yes']['rating'],
        df_rated[df_rated['verified_flag'] == 'No']['rating']
    )
    u_stat_ver, p_val_u_ver = stats.mannwhitneyu(
        df_rated[df_rated['verified_flag'] == 'Yes']['rating'],
        df_rated[df_rated['verified_flag'] == 'No']['rating']
    )
    print(f"\nTwo-sample Independent t-test: t = {t_stat_ver:.4f}, p = {p_val_ver:.4f}")
    print(f"Mann-Whitney U Test: U = {u_stat_ver:.1f}, p = {p_val_u_ver:.4f}")
    
    results['Q2_table'] = ver_summary
    results['Q2_ttest'] = (t_stat_ver, p_val_ver)
    
    # Plot Q2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    sns.boxplot(
        data=df_rated, 
        x='verified_flag', 
        y='rating', 
        palette=['#457b9d', '#2a9d8f'], 
        ax=ax1, 
        width=0.45,
        boxprops=dict(alpha=0.85)
    )
    ax1.set_title("Rating Distribution: Verified vs Unverified", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Verified Purchase Flag", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Rating (Stars)", fontsize=10, fontweight='bold')
    
    # Rating proportions per flag
    prop_df = df_rated.groupby(['verified_flag', 'rating']).size().unstack(fill_value=0)
    prop_pct = prop_df.div(prop_df.sum(axis=1), axis=0) * 100
    prop_pct.plot(kind='bar', stacked=True, colormap='Blues', ax=ax2, edgecolor='#333333', linewidth=0.6)
    ax2.set_title("Rating Breakdown by Verification Status (%)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Verified Purchase Flag", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Percentage (%)", fontsize=10, fontweight='bold')
    ax2.legend(title='Rating', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_output_dir, "02_verified_vs_unverified_ratings.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # QUESTION 3: How does delivery time relate to rating?
    # -------------------------------------------------------------------------
    print("\n>>> QUESTION 3: How does delivery time relate to rating? <<<")
    del_summary = df_rated.groupby('delivery_days')['rating'].agg(
        review_count='count',
        mean_rating='mean',
        median_rating='median',
        std_rating='std'
    )
    print(del_summary.to_string())
    
    corr_p, p_val_p = stats.pearsonr(df_rated['delivery_days'], df_rated['rating'])
    corr_s, p_val_s = stats.spearmanr(df_rated['delivery_days'], df_rated['rating'])
    print(f"\nPearson correlation (delivery_days vs rating):  r = {corr_p:.4f}, p = {p_val_p:.4e}")
    print(f"Spearman correlation (delivery_days vs rating): rho = {corr_s:.4f}, p = {p_val_s:.4e}")
    
    results['Q3_table'] = del_summary
    results['Q3_corr'] = (corr_p, corr_s)
    
    # Plot Q3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    # Line plot with mean & confidence interval
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
    ax1.annotate(f"Pearson r = {corr_p:.2f} (p < 0.001)", xy=(0.05, 0.1), xycoords='axes fraction',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f1faee', edgecolor='#1d3557'),
                 fontsize=10, fontweight='bold')
    
    # Binned delivery speed bar plot
    df_rated['delivery_bin'] = pd.cut(df_rated['delivery_days'], bins=[0, 2, 5, 20], labels=['Express (1-2d)', 'Standard (3-5d)', 'Delayed (6+d)'])
    bin_stats = df_rated.groupby('delivery_bin', observed=False)['rating'].mean()
    sns.barplot(x=bin_stats.index, y=bin_stats.values, hue=bin_stats.index, palette='copper', legend=False, ax=ax2, edgecolor='#333333')
    ax2.set_ylim(2.5, 4.5)
    ax2.set_title("Average Rating by Delivery Speed Tier", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Delivery Speed Tier", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Mean Rating (Stars)", fontsize=10, fontweight='bold')
    for p in ax2.patches:
        h = p.get_height()
        ax2.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h + 0.04), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_output_dir, "03_delivery_time_vs_rating_trend.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # QUESTION 4: Which sellers show recurring low ratings?
    # -------------------------------------------------------------------------
    print("\n>>> QUESTION 4: Which sellers show recurring low ratings? <<<")
    seller_summary = df_rated.groupby(['seller_id', 'seller', 'region']).agg(
        review_count=('rating', 'count'),
        mean_rating=('rating', 'mean'),
        median_rating=('rating', 'median'),
        low_rating_count=('rating', lambda x: (x <= 2).sum()),
        low_rating_pct=('rating', lambda x: (x <= 2).mean() * 100),
        tenure_years=('tenure_years', 'first')
    ).reset_index().sort_values(by=['mean_rating', 'low_rating_pct'], ascending=[True, False])
    print(seller_summary.head(10).to_string(index=False))
    
    results['Q4_table'] = seller_summary
    
    # Plot Q4
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    top_lowest_sellers = seller_summary.head(10).copy()
    top_lowest_sellers['label'] = top_lowest_sellers['seller'] + " (" + top_lowest_sellers['region'] + ")"
    
    bar_plot = sns.barplot(
        data=top_lowest_sellers,
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
        pct = top_lowest_sellers.iloc[idx]['low_rating_pct']
        ax.annotate(f"{h:.2f}\n({pct:.0f}% ≤2★)", (p.get_x() + p.get_width() / 2., h + 0.04),
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_output_dir, "04_seller_recurring_low_ratings.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # QUESTION 5: Do returned orders have lower ratings?
    # -------------------------------------------------------------------------
    print("\n>>> QUESTION 5: Do returned orders have lower ratings? <<<")
    ret_summary = df_rated.groupby('return_flag')['rating'].agg(
        review_count='count',
        mean_rating='mean',
        median_rating='median',
        std_rating='std',
        low_rating_pct=lambda x: (x <= 2).mean() * 100
    )
    print(ret_summary.to_string())
    
    t_stat_ret, p_val_ret = stats.ttest_ind(
        df_rated[df_rated['return_flag'] == 'Yes']['rating'],
        df_rated[df_rated['return_flag'] == 'No']['rating']
    )
    # Cohen's d effect size
    mean_yes = df_rated[df_rated['return_flag'] == 'Yes']['rating'].mean()
    mean_no = df_rated[df_rated['return_flag'] == 'No']['rating'].mean()
    s_pooled = np.sqrt(
        ((len(df_rated[df_rated['return_flag'] == 'Yes']) - 1) * df_rated[df_rated['return_flag'] == 'Yes']['rating'].var() +
         (len(df_rated[df_rated['return_flag'] == 'No']) - 1) * df_rated[df_rated['return_flag'] == 'No']['rating'].var()) /
        (len(df_rated) - 2)
    )
    cohens_d = (mean_yes - mean_no) / s_pooled
    print(f"\nTwo-sample Independent t-test: t = {t_stat_ret:.4f}, p = {p_val_ret:.4e}")
    print(f"Cohen's d Effect Size: d = {cohens_d:.4f} (Substantial negative impact)")
    
    results['Q5_table'] = ret_summary
    results['Q5_ttest'] = (t_stat_ret, p_val_ret, cohens_d)
    
    # Plot Q5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300)
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
    
    # Low rating percentage bar
    sns.barplot(
        x=ret_summary.index,
        y=ret_summary['low_rating_pct'],
        hue=ret_summary.index,
        palette=['#2a9d8f', '#e76f51'],
        legend=False,
        ax=ax2,
        edgecolor='#333333'
    )
    ax2.set_title("Proportion of Low Ratings (≤ 2 Stars) %", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Order Return Flag", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Low Rating Share (%)", fontsize=10, fontweight='bold')
    for p in ax2.patches:
        h = p.get_height()
        ax2.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 1), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_output_dir, "05_returned_orders_rating_impact.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # QUESTION 6: What complaint categories align with low ratings?
    # -------------------------------------------------------------------------
    print("\n>>> QUESTION 6: What complaint categories align with low ratings? <<<")
    # Method 1: Compare orders with tickets vs without tickets
    sup_status = df_rated.groupby('has_support_ticket')['rating'].agg(
        review_count='count',
        mean_rating='mean',
        median_rating='median',
        std_rating='std',
        low_rating_pct=lambda x: (x <= 2).mean() * 100
    )
    print("Ticket Status Comparison:")
    print(sup_status.to_string())
    
    # Method 2: Analysis of primary complaint categories (for orders with tickets)
    comp_summary = df_rated[df_rated['has_support_ticket'] == 1].groupby('primary_complaint_category')['rating'].agg(
        review_count='count',
        mean_rating='mean',
        median_rating='median',
        std_rating='std',
        low_rating_pct=lambda x: (x <= 2).mean() * 100
    ).sort_values('mean_rating')
    print("\nPrimary Complaint Category Ratings:")
    print(comp_summary.to_string())
    
    results['Q6_status'] = sup_status
    results['Q6_categories'] = comp_summary
    
    # Plot Q6
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.barplot(
        x=comp_summary.index,
        y=comp_summary['mean_rating'],
        hue=comp_summary.index,
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
    plt.savefig(os.path.join(figures_output_dir, "06_complaint_categories_low_ratings.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # ADDITIONAL EDA VISUALIZATIONS: Rating & Length Distributions
    # -------------------------------------------------------------------------
    # 07: Rating distribution & rating classes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    rating_counts = df['rating'].value_counts(dropna=False).sort_index()
    labels_x = [str(x) for x in rating_counts.index]
    sns.barplot(x=labels_x, y=rating_counts.values, hue=labels_x, palette='viridis', legend=False, ax=ax1, edgecolor='#333333')
    ax1.set_title("Overall Rating Count Distribution (Including Missing)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Rating Stars", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Count", fontsize=10, fontweight='bold')
    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2., h + 5), ha='center', va='bottom', fontweight='bold')
        
    # Rating class breakdown (Positive, Neutral, Negative)
    def to_class(r):
        if pd.isna(r):
            return 'Missing (3)'
        elif r <= 2:
            return 'Negative (1-2★)'
        elif r == 3:
            return 'Neutral (3★)'
        else:
            return 'Positive (4-5★)'
            
    classes = df['rating'].apply(to_class).value_counts()
    colors = ['#2a9d8f', '#e9c46a', '#e76f51', '#adb5bd']
    ax2.pie(classes.values, labels=classes.index, autopct='%1.1f%%', colors=colors, startangle=140,
            wedgeprops=dict(edgecolor='#333333', linewidth=1.2))
    ax2.set_title("Customer Sentiment Tier Share", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_output_dir, "07_rating_distribution_and_classes.png"))
    plt.close()
    
    # 08: Review Length Distribution & Extreme Outliers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    # Boxplot of raw review length vs capped review length
    sns.histplot(df['review_length_capped'], bins=30, kde=True, color='#264653', ax=ax1, edgecolor='#1d3557')
    ax1.set_title("Review Length Distribution (Capped at 99th Pct)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Review Length (Characters)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Frequency", fontsize=10, fontweight='bold')
    
    # Scatter plot: review_length vs actual_char_count showing artificial outliers
    ax2.scatter(df['actual_char_count'], df['review_length'], color='#e76f51', alpha=0.6, edgecolors='none', label='Reviews')
    ax2.set_yscale('log')
    ax2.set_title("Simulated vs Actual Text Length (Log Scale)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Actual Text Character Count", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Recorded review_length (Log Scale)", fontsize=10, fontweight='bold')
    ax2.annotate("Extreme Outliers\n(5000 & 8000 chars)", xy=(38, 5500), xytext=(80, 2000),
                 arrowprops=dict(facecolor='#d62828', shrink=0.05, width=1.5, headwidth=6),
                 fontweight='bold', color='#d62828')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_output_dir, "08_review_length_distribution_and_outliers.png"))
    plt.close()
    
    print("\nAll 8 core EDA and question figures generated and saved successfully!")
    return results


if __name__ == "__main__":
    master_path = r"c:\Users\dharu\OneDrive\Desktop\DATASAT2\output\merged_data\reviewlens_master_analysis.csv"
    figs_path = r"c:\Users\dharu\OneDrive\Desktop\DATASAT2\output\figures"
    run_eda_and_answer_questions(master_path, figs_path)
