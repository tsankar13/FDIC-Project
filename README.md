# FDIC Risk Assessment: 2025 Institutional Liquidity & Capital Adequacy
Authors:

### Tharun Sankar – Data Science, Computer Science, Economics third-year at University of California, Davis

### Philip Rybkin – Managerial Economics, Statistics third-year at University of California, Davis

## Project Overview

This project performs a comprehensive Top-Down Risk Assessment of the U.S. banking sector using 2025 FDIC Call Report data. By processing and merging 15+ disparate datasets, we identified a systemic "Duration Trap" affecting mid-cap financial institutions ($10B–$50B in assets). Our analysis utilizes unsupervised machine learning to segment the 4,000+ FDIC-insured institutions into distinct risk cohorts based on capital adequacy, liquidity, and asset performance.

## Motivation: The "Duration Trap"

Following the high-interest-rate environment of 2023-2024, many banks entered 2025 with significant unrealized losses on their Held-to-Maturity (HTM) and Available-for-Sale (AFS) securities portfolios. We aimed to quantify how these "paper losses" impacted Tier 1 Capital and restricted the ability of regional banks to rotate into higher-yield assets, creating a measurable squeeze on Net Interest Margins (NIM).

## Technical Pipeline

1. Data Engineering (ETL)
Multi-Source Ingestion: Automated the ingestion of 15+ bulk CSV schedules, including Balance Sheets (RC), Income Statements (RI), and Asset Quality (RC-N).

Schema Standardization: Resolved complex header inconsistencies and handled "CERT" (Certificate Number) join conflicts to maintain a traceable master dataset for 4,000+ institutions.

Feature Engineering: Derived critical financial ratios, including the Texas Ratio, Tier 1 Capital Ratio, and a custom Duration Trap Sensitivity Index.

2. Unsupervised Machine Learning
K-Means Clustering: Applied K-Means to partition the banking sector into risk-based clusters.

Feature Scaling: Utilized StandardScaler to normalize features across vastly different magnitudes (e.g., Total Assets vs. NIM percentages).

Optimal K-Selection: Used the Elbow Method and Silhouette Analysis to determine the most statistically significant grouping of bank health profiles.

## Key Insights

Mid-Cap Vulnerability: Our model identified a specific cluster of mid-sized banks holding 18% higher unrealized losses relative to Tier 1 Capital compared to Global Systemically Important Banks (G-SIBs).

NIM Squeeze: Identified a strong correlation between "Duration Trapped" institutions and a 14% average decline in Net Interest Margin, as high deposit costs outpaced returns from low-yield legacy bond portfolios.

Geographic Risk: Visualized regional "hotspots" where Commercial Real Estate (CRE) exposure overlapped with poor liquidity scores.

## Technologies

Language: Python

Data Libraries: Pandas, NumPy

Machine Learning: Scikit-learn (K-Means, PCA)

Visualization: Matplotlib, Seaborn, Plotly
