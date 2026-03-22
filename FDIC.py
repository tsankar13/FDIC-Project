import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. DATA LOADING
# Updating filenames to match your specific 'cleaned' and interest files
files = {
    'assets': 'Total Assets (3)_cleaned.csv',
    'cap': 'Total Liabilities and Capital (3)_cleaned.csv',
    'income': 'Total Interest Income (3)_cleaned.csv',
    'expense': 'Total Interest Expense.csv',
    'past_due': 'Past Due and Nonaccrual Assets (3)_cleaned.csv',
    'charge_offs': 'Net Charge-Offs (3)_cleaned.csv'
}

def load_and_merge_data(file_dict):
    print("Starting ETL process...")
    # Load all files
    dfs = {k: pd.read_csv(v) for k, v in file_dict.items()}
    
    # Merge sequentially on 'CERT' (the unique bank identifier)
    master_df = dfs['assets']
    for key in ['cap', 'income', 'expense', 'past_due', 'charge_offs']:
        master_df = master_df.merge(dfs[key], on='CERT', how='left', suffixes=('', f'_{key}'))
    
    # Clean up duplicate columns if any exist after the merge
    master_df = master_df.loc[:, ~master_df.columns.str.contains('_drop')]
    return master_df

df_raw = load_and_merge_data(files)

# 2. FEATURE ENGINEERING
# We transform raw data into ratios to normalize for bank size
def engineer_risk_features(df):
    f = pd.DataFrame(index=df.index)
    f['CERT'] = df['CERT']
    
    # Capital Adequacy: Is the bank's cushion large enough?
    f['capital_ratio'] = df['Total Tier 1 Capital'] / df['Total Assets']
    
    # Net Interest Margin (NIM): The primary 'Duration Trap' proxy
    # (Income - Expense) / Assets
    f['nim'] = (df['Total Interest Income'] - df['Total Interest Expense']) / df['Total Assets']
    
    # Asset Quality: The Texas Ratio
    # Non-performing assets relative to capital cushion
    f['texas_ratio'] = df['Total Past Due and Nonaccrual Assets'] / df['Total Tier 1 Capital']
    
    # Efficiency: Net charge-offs relative to assets
    f['charge_off_ratio'] = df['Total Net Charge-Offs'] / df['Total Assets']
    
    return f.dropna()

features_df = engineer_risk_features(df_raw)

# 3. SCALING & CLUSTERING
# K-Means requires scaling because 'Assets' and 'NIM' have different magnitudes
X = features_df.drop('CERT', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Running K-Means with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
features_df['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. DIMENSIONALITY REDUCTION (PCA) FOR VISUALIZATION
# Compressing 4 features into 2D coordinates
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
features_df['PCA1'] = components[:, 0]
features_df['PCA2'] = components[:, 1]

# 5. VISUALIZATION
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")
sns.scatterplot(data=features_df, x='PCA1', y='PCA2', hue='Cluster', palette='RdYlGn_r', s=100, alpha=0.8)

plt.title("2
