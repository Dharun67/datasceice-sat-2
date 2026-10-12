ReviewLens Round 2 Practice Dataset
====================================
This is a SYNTHETIC practice dataset created to match the schema described in the user's DS Sprint Use Case 26 PDF.
It is not the official competition dataset and contains no real customer information.

Files:
- reviews.csv: review ratings, text, lengths, verified flag, date, product_id, order_id
- products.csv: product, category, brand, price band
- sellers.csv: seller, region, tenure
- orders.csv: delivery days, discount, return flag, product_id, seller_id
- support.csv: complaint category, resolution, order_id

Intentional practice data-quality issues:
- Missing ratings
- Exact duplicate review rows
- Inconsistent category labels
- Extreme review lengths
- Unmatched product IDs

Use this dataset to practise cleaning, merging, EDA, visualization, feature engineering and feature selection.
For official contest submission, use the organizer-provided dataset if one becomes available.
