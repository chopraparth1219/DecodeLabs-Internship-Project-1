import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

df=pd.read_csv('data1.csv', parse_dates=['Date'])
print("Original datraset shape:", df.shape)
np.random.seed(42)
for col in ['Quantity','UnitPrice']:
    mask=np.random.random(len(df))< 0.05
    df.loc[mask, col]=np.nan

df.loc[0, 'TotalPrice'] = 50000
print(df.isnull().sum())

df['Quantitym'] = df['Quantity'].fillna(df['Quantity'].median())
df['UnitPricem'] = df['UnitPrice'].fillna(df['UnitPrice'].median())

knn_imputer=KNNImputer(n_neighbors=2)
df_knn=df.copy()
df_knn[['Quantity', 'UnitPrice']] = knn_imputer.fit_transform(df_knn[['Quantity', 'UnitPrice']])

df_clean = df.copy()
df_clean['Quantity'] = df_clean['Quantitym']
df_clean['UnitPrice'] = df_clean['UnitPricem']
df_clean.drop(['Quantitym', 'UnitPricem'], axis=1, inplace=True)

print("\nMissing values after median imputation:")
print(df_clean.isnull().sum())

Q1 = df_clean['TotalPrice'].quantile(0.25)
Q3 = df_clean['TotalPrice'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df_clean[(df_clean['TotalPrice'] < lower_bound) | (df_clean['TotalPrice'] > upper_bound)]
print(f"\nOutliers detected by IQR in 'TotalPrice': {len(outliers_iqr)}")

df_clean['TotalPrice_capped'] = df_clean['TotalPrice'].clip(lower=lower_bound, upper=upper_bound)


z_scores = np.abs((df_clean['UnitPrice'] - df_clean['UnitPrice'].mean()) / df_clean['UnitPrice'].std())
outliers_z = df_clean[z_scores > 3]
print(f"Outliers detected by Z‑score in 'UnitPrice': {len(outliers_z)}")

unit_mean = df_clean['UnitPrice'].mean()
unit_std = df_clean['UnitPrice'].std()
df_clean['UnitPrice_capped'] = df_clean['UnitPrice'].clip(unit_mean - 3*unit_std, unit_mean + 3*unit_std)

df_final = df_clean.copy()
df_final['TotalPrice'] = df_final['TotalPrice_capped']
df_final['UnitPrice'] = df_final['UnitPrice_capped']
df_final.drop(['TotalPrice_capped', 'UnitPrice_capped'], axis=1, inplace=True)

print("\nAfter capping, no rows were deleted. Outliers are now within statistical bounds.")

df_final['TotalPrice_per_Quantity'] = df_final['TotalPrice'] / df_final['Quantity'].replace(0, np.nan)
df_final['TotalPrice_per_Quantity'].fillna(df_final['TotalPrice_per_Quantity'].median(), inplace=True)

median_total = df_final['TotalPrice'].median()
df_final['IsHighValue'] = (df_final['TotalPrice'] > median_total).astype(int)

if 'Date' in df_final.columns:
    df_final['Year'] = df_final['Date'].dt.year
    df_final['Month'] = df_final['Date'].dt.month
    df_final['DayOfWeek'] = df_final['Date'].dt.dayofweek

print("\n" + "="*60)
print("FINAL CLEANED DATASET STATISTICS")
print("="*60)
print(df_final.describe())

print("\nMissing values after full pipeline:")
print(df_final.isnull().sum())

df_final.to_csv('data1_cleaned.csv', index=False)
print("\nCleaned dataset saved as 'data1_cleaned.csv'")
