import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. FILE MAPPING
# Filenames updated to match your uploaded cleaned datasets exactly
files = {
    'assets': 'Total Assets (3)_cleaned.csv',
    'cap': 'Total Liabilities and Capital (3)_cleaned.csv',
    'income': 'Total Interest Income (3)_cleaned.csv',
    'expense': 'Total Interest Expense_cleaned.csv',
    'past_due': 'Past Due and Nonaccrual Assets (3)_cleaned.csv',
    'charge_offs': 'Net Charge-Offs (3)_cleaned.csv'
}

def load_and_merge_data(file_dict):
    print("🚀 Initializing 6-file ETL Pipeline...")
    # Load all files into a dictionary
    dfs = {k: pd.read_csv(v) for k, v in file_dict.items()}
    
    # Merge sequentially on 'CERT' (the unique bank identifier)
    master_df = dfs['assets']
    for key in ['cap', 'income', 'expense', 'past_due', 'charge_offs']:
        master_df = master_df.merge(dfs[key], on='CERT', how='left', suffixes=('', f'_{key}'))
    
    # Drop any duplicate columns created during the merge
    master_df = master_df.loc[:, ~master_df.columns.str.contains('_drop')]
    return master_df

# Execute ETL
df_raw = load_and_merge_data(files)
print(f"✅ Data successfully merged for {len(df_raw)} unique institutions.")

# 2. FEATURE ENGINEERING
def engineer_risk_features(df):
    f = pd.DataFrame(index=df.index)
    f['CERT'] = df['CERT']
    
    # Solvency: Tier 1 Capital Ratio
    f['capital_ratio'] = df['Total Tier 1 Capital'] / df['Total Assets']
    
    # Profitability: Net Interest Margin (NIM)
    f['nim'] = (df['Total Interest Income'] - df['Total Interest Expense']) / df['Total Assets']
    
    # Asset Quality: Texas Ratio
    f['texas_ratio'] = df['Total Past Due and Nonaccrual Assets'] / df['Total Tier 1 Capital']
    
    # Realized Risk: Charge-Off Ratio
    f['charge_off_ratio'] = df['Total Net Charge-Offs'] / df['Total Assets']
    
    return f.dropna()

features_df = engineer_risk_features(df_raw)

# 3. K-MEANS CLUSTERING
# Standardizing features is critical for K-Means distance calculations
X = features_df.drop('CERT', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fitting 4 clusters to identify distinct health profiles
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
features_df['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. DIMENSIONALITY REDUCTION (PCA)
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
features_df['PCA1'] = components[:, 0]
features_df['PCA2'] = components[:, 1]

# 5. VISUALIZATION
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")
scatter = sns.scatterplot(
    data=features_df, x='PCA1', y='PCA2', 
    hue='Cluster', palette='RdYlGn_r', s=100, alpha=0.8, style='Cluster'
)

plt.title("2025 FDIC Risk Assessment: Multi-Factor Cluster Analysis", fontsize=16)
plt.xlabel("PCA 1: Capital Strength & Size")
plt.ylabel("PCA 2: Profitability & Credit Risk")
plt.legend(title="Risk Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 6. RESULTS INTERPRETATION
print("\n--- Mean Financial Ratios by Risk Cluster ---")
summary = features_df.groupby('Cluster').mean().drop(['CERT', 'PCA1', 'PCA2'], axis=1)
print(summary)
