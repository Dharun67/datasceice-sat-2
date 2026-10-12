"""
ReviewLens Commerce Analytics - DS Sprint Use Case 26
Round 2: EDA and Feature Engineering
Module: Data Merging (data_merging.py)

This module validates primary keys and foreign keys across all cleaned datasets,
prevents duplicate rows caused by one-to-many relationship traps, creates an aggregated
order-support view, and produces a clean 1-row-per-review master analysis dataset
as well as a dedicated ticket-level complaint analysis dataset.
"""

import os
import pandas as pd
import numpy as np


def merge_reviewlens_data(cleaned_data_dir: str, merged_output_dir: str):
    """
    Validates keys, merges datasets, and prevents join duplication.
    """
    os.makedirs(merged_output_dir, exist_ok=True)
    
    # 1. Load Cleaned Datasets
    reviews = pd.read_csv(os.path.join(cleaned_data_dir, "reviews_cleaned.csv"))
    products = pd.read_csv(os.path.join(cleaned_data_dir, "products_cleaned.csv"))
    sellers = pd.read_csv(os.path.join(cleaned_data_dir, "sellers_cleaned.csv"))
    orders = pd.read_csv(os.path.join(cleaned_data_dir, "orders_cleaned.csv"))
    support = pd.read_csv(os.path.join(cleaned_data_dir, "support_cleaned.csv"))
    
    print("=== Primary Key & Foreign Key Validation ===")
    
    # Validate Primary Keys
    pk_checks = {
        'reviews.review_id': reviews['review_id'].is_unique,
        'products.product_id': products['product_id'].is_unique,
        'sellers.seller_id': sellers['seller_id'].is_unique,
        'orders.order_id': orders['order_id'].is_unique,
        'support.support_id': support['support_id'].is_unique
    }
    for pk_name, is_unique in pk_checks.items():
        print(f"Primary Key {pk_name} unique: {is_unique}")
        if not is_unique:
            raise ValueError(f"Primary key {pk_name} is not unique!")
            
    # Foreign Key Integrity Checks
    fk_orders_prod = set(orders['product_id']) - set(products['product_id'])
    fk_orders_seller = set(orders['seller_id']) - set(sellers['seller_id'])
    fk_reviews_order = set(reviews['order_id']) - set(orders['order_id'])
    fk_reviews_prod = set(reviews['product_id']) - set(products['product_id'])
    fk_support_order = set(support['order_id']) - set(orders['order_id'])
    
    print(f"Unmatched orders -> products: {len(fk_orders_prod)}")
    print(f"Unmatched orders -> sellers:  {len(fk_orders_seller)}")
    print(f"Unmatched reviews -> orders:  {len(fk_reviews_order)}")
    print(f"Unmatched reviews -> products (after reconciliation): {len(fk_reviews_prod)}")
    print(f"Unmatched support -> orders:  {len(fk_support_order)}")
    
    # 2. Support Aggregation Strategy to Prevent Join Traps
    # Support has 290 tickets across 243 unique orders (42 orders have multiple tickets).
    # Joining raw support directly to reviews will cause a Cartesian product / row multiplication.
    # Therefore, we aggregate support per order_id before merging with reviews:
    
    # Complaint categories list
    categories = ['Damaged Item', 'Late Delivery', 'Quality Issue', 'Refund Delay', 'Wrong Item', 'Other']
    
    support_cat_dummies = pd.get_dummies(support.set_index('order_id')['complaint_category'], prefix='complaint').groupby('order_id').max()
    
    support_agg = support.groupby('order_id').agg(
        support_ticket_count=('support_id', 'count'),
        primary_complaint_category=('complaint_category', 'first'),
        complaint_categories_list=('complaint_category', lambda x: '; '.join(x.unique())),
        latest_resolution=('resolution', 'last'),
        has_pending_support=('resolution', lambda x: int((x == 'Pending').any())),
        has_escalated_support=('resolution', lambda x: int((x == 'Escalated').any())),
        has_resolved_support=('resolution', lambda x: int((x == 'Resolved').all()))
    ).reset_index()
    
    # Merge dummy complaint indicators
    support_summary = support_agg.merge(support_cat_dummies.reset_index(), on='order_id', how='left')
    support_summary['has_support_ticket'] = 1
    
    # 3. Master Analysis Dataset Merge
    # Base: reviews (1050 rows)
    # Step A: Merge orders (order_id)
    # Note: orders has product_id and seller_id. reviews has product_id (reconciled).
    master_df = reviews.merge(
        orders[['order_id', 'seller_id', 'delivery_days', 'discount_pct', 'return_flag', 'is_returned']], 
        on='order_id', 
        how='left'
    )
    
    # Step B: Merge products (product_id)
    master_df = master_df.merge(products, on='product_id', how='left')
    
    # Step C: Merge sellers (seller_id)
    master_df = master_df.merge(sellers, on='seller_id', how='left')
    
    # Step D: Merge support_summary (order_id)
    master_df = master_df.merge(support_summary, on='order_id', how='left')
    
    # Fill defaults for orders without support tickets
    master_df['has_support_ticket'] = master_df['has_support_ticket'].fillna(0).astype(int)
    master_df['support_ticket_count'] = master_df['support_ticket_count'].fillna(0).astype(int)
    master_df['primary_complaint_category'] = master_df['primary_complaint_category'].fillna('None')
    master_df['complaint_categories_list'] = master_df['complaint_categories_list'].fillna('None')
    master_df['latest_resolution'] = master_df['latest_resolution'].fillna('None')
    master_df['has_pending_support'] = master_df['has_pending_support'].fillna(0).astype(int)
    master_df['has_escalated_support'] = master_df['has_escalated_support'].fillna(0).astype(int)
    master_df['has_resolved_support'] = master_df['has_resolved_support'].fillna(0).astype(int)
    
    for cat in categories:
        col = f"complaint_{cat}"
        if col in master_df.columns:
            master_df[col] = master_df[col].fillna(0).astype(int)
        else:
            master_df[col] = 0
            
    # Verify no row explosion occurred
    print(f"\n[Validation] Master Analysis Dataset rows: {len(master_df)} (Original cleaned reviews: {len(reviews)})")
    if len(master_df) != len(reviews):
        raise ValueError("Join multiplication detected! Review row count changed.")
        
    # 4. Create Ticket-Level Analysis Dataset (for Question 6 and granular support deep dives)
    # Each row is a support ticket, enriched with order, product, seller, and average review rating
    order_review_stats = reviews.groupby('order_id').agg(
        order_review_count=('review_id', 'count'),
        order_avg_rating=('rating', 'mean'),
        order_has_low_rating=('rating', lambda x: int((x <= 2).any()))
    ).reset_index()
    
    support_analysis_df = support.merge(orders, on='order_id', how='left')
    support_analysis_df = support_analysis_df.merge(products, on='product_id', how='left')
    support_analysis_df = support_analysis_df.merge(sellers, on='seller_id', how='left')
    support_analysis_df = support_analysis_df.merge(order_review_stats, on='order_id', how='left')
    
    # Save datasets
    master_df.to_csv(os.path.join(merged_output_dir, "reviewlens_master_analysis.csv"), index=False)
    support_analysis_df.to_csv(os.path.join(merged_output_dir, "reviewlens_support_analysis.csv"), index=False)
    
    print("\n=== Merged Datasets Successfully Saved ===")
    print(f"reviewlens_master_analysis.csv:  {master_df.shape}")
    print(f"reviewlens_support_analysis.csv: {support_analysis_df.shape}")
    
    return master_df, support_analysis_df


if __name__ == "__main__":
    cleaned_dir = r"c:\Users\dharu\OneDrive\Desktop\DATASAT2\output\cleaned_data"
    merged_dir = r"c:\Users\dharu\OneDrive\Desktop\DATASAT2\output\merged_data"
    merge_reviewlens_data(cleaned_dir, merged_dir)
