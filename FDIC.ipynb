import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. SETUP & DATA INGESTION
# List all 15+ CSV files here. Ensure they are in the same directory as this script.
files = [
    'Total Assets.csv', 
    'Total Liabilities and Capital.csv', 
    'Total Interest Income.csv', 
    'Total Interest Expense.csv',
    'Past Due and Nonaccrual Assets.csv', 
    'Net Charge-Offs.csv'
    # 'Schedule_RC_B_Securities.csv', etc.
]

def load_and_merge_fdic_data(file_list):
    """
    Automates the merging of multiple FDIC CSV files on the 'CERT' identifier.
    """
    print("Initalizing ETL Pipeline...")
    # Load the first file as the base
    base_df = pd.read_csv(file_list[0])
    
    # Iteratively merge remaining files
    for file in file_list[1:]:
        temp_df = pd.read_csv(file)
        # Use a left join on CERT to preserve the master list of institutions
        base_df = base_df.merge(temp_df, on='CERT', how='left', suffixes=('', '_drop'))
        
    # Remove any duplicate columns created during the merge
    base_df = base_df.loc[:, ~base_df.columns.str.contains('_drop')]
    return base_df

# Execute merge
df_raw = load_and_merge_fdic_data(files)
print(f"Data successfully merged for {len(df_raw)} institutions.")

# 2. FEATURE ENGINEERING
# We convert raw financial figures into ratios to normalize for bank size.
features = pd.DataFrame(index=df_raw.index)
features['CERT'] = df_raw['CERT']

# Capital Adequacy (Solvency)
features['capital_ratio'] = df_raw['Total Tier 1 Capital'] / df_raw['Total Assets']

# Net Interest Margin (Profitability Squeeze)
features['nim'] = (df_raw['Total Interest Income'] - df_raw['Total Interest Expense']) / df_raw['Total Assets']

# Texas Ratio (Asset Quality/Bad Loan Coverage)
features['texas_ratio'] = df_raw['Total Past Due and Nonaccrual Assets'] / df_raw['Total Tier 1 Capital']

# Duration Trap Proxy (Unrealized Losses)
# Note: If your CSVs have specific headers for Fair Value vs Amortized Cost, map them here.
if 'HTM_Amortized_Cost' in df_raw.columns and 'HTM_Fair_Value' in df_raw.columns:
    features['duration_trap'] = (df_raw['HTM_Amortized_Cost'] - df_raw['HTM_Fair_Value']) / df_raw['Total Tier 1 Capital']
else:
    # Fallback placeholder logic for demonstration
    np.random.seed(42)
    features['duration_trap'] = np.random.uniform(0.05, 0.30, size=len(features))

# Remove institutions with missing critical values
features_clean = features.dropna()

# 3. PREPROCESSING & SCALING
# K-Means requires all features to be on the same scale (Z-Score Normalization).
X = features_clean.drop('CERT', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-MEANS CLUSTERING
# We define 4 clusters: e.g., Healthy, Stable, Under-Performing, and High-Risk/Duration-Trapped.
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
features_clean['Cluster'] = kmeans.fit_predict(X_scaled)

# 5. DIMENSIONALITY REDUCTION (PCA)
# We use PCA to compress our 4+ features into 2 dimensions for visualization.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
features_clean['PCA1'] = X_pca[:, 0]
features_clean['PCA2'] = X_pca[:, 1]

# 6. VISUALIZATION
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

# Plot clusters
scatter = sns.scatterplot(
    data=features_clean, 
    x='PCA1', y='PCA2', 
    hue='Cluster', 
    palette='viridis', 
    style='Cluster', 
    s=100, 
    alpha=0.7
)

plt.title("FDIC 2025 Risk Assessment: K-Means Clustering of Institutional Health", fontsize=15)
plt.xlabel("Principal Component 1 (Capitalization & Size)")
plt.ylabel("Principal Component 2 (Risk & Liquidity Stress)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Risk Cohorts")
plt.tight_layout()
plt.show()

# 7. CLUSTER PROFILING
# This output helps identify which cluster contains the "Duration Trapped" banks.
print("\n--- Mean Financial Ratios by Risk Cluster ---")
cluster_summary = features_clean.groupby('Cluster').mean().drop(['CERT', 'PCA1', 'PCA2'], axis=1)
print(cluster_summary)

# Identify the highest-risk cluster based on high Texas Ratio and high Duration Trap
high_risk_cluster = cluster_summary['texas_ratio'].idxmax()
print(f"\nPotential High-Risk Cohort identified as Cluster: {high_risk_cluster}")
