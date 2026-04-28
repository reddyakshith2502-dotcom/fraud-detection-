import pandas as pd

# Load the full dataset
df = pd.read_csv('dataset.csv')

# Get all 492 fraud transactions
fraud_df = df[df['Class'] == '"1"']
if len(fraud_df) == 0:
    fraud_df = df[df['Class'] == 1]
if len(fraud_df) == 0:
    fraud_df = df[df['Class'] == '1']

print(f"Found {len(fraud_df)} fraud rows.")

# Get 5000 safe transactions
safe_df = df[df['Class'] == '"0"']
if len(safe_df) == 0:
    safe_df = df[df['Class'] == 0]
if len(safe_df) == 0:
    safe_df = df[df['Class'] == '0']
    
safe_sample = safe_df.sample(5000, random_state=42)

# Combine and shuffle
demo_df = pd.concat([fraud_df, safe_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new file
demo_df.to_csv('demo_dataset.csv', index=False)
print("demo_dataset.csv created successfully with", len(demo_df), "rows.")
