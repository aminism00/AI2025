
# question1.py
# Exploratory Data Analysis for Iris.csv (Q1 - Python)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('Iris.csv')  # adjust path if needed

print('Shape:', df.shape)
print('\\nColumns:', df.columns.tolist())
print('\\nHead:')
print(df.head())

print('\\nInfo:')
print(df.info())

print('\\nMissing values per column:')
print(df.isnull().sum())

print('\\nDuplicate rows count:', df.duplicated().sum())

print('\\nValue counts (Species):')
print(df['Species'].value_counts())

print('\\nDescriptive statistics:')
print(df.describe())

# Correlation matrix and heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr()
plt.figure()
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title('Correlation matrix (Iris numeric features)')
plt.tight_layout()
plt.savefig('fig_corr.png', dpi=200)
plt.close()

# PCA (2D) after standardization
X = df[numeric_cols].values
Xstd = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
Xproj = pca.fit_transform(Xstd)

plt.figure()
species = df['Species'].unique()
for sp in species:
    mask = df['Species'] == sp
    plt.scatter(Xproj[mask,0], Xproj[mask,1], label=str(sp))
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA (2D) projection of Iris (standardized features)')
plt.legend()
plt.tight_layout()
plt.savefig('fig_pca.png', dpi=200)
plt.close()
