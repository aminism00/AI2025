# kmeans_iris.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')
numeric_cols = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = df[numeric_cols].values
Xstd = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=0)
Xproj = pca.fit_transform(Xstd)

for k in [2,3,4]:
    km = KMeans(n_clusters=k, n_init=20, random_state=0)
    labels = km.fit_predict(Xstd)
    print(f'k={k}: inertia={km.inertia_:.4f}, silhouette={(silhouette_score(Xstd,labels) if k>1 else float("nan")):.4f}')
    centers_proj = pca.transform(km.cluster_centers_)
    plt.figure()
    for lab in range(k):
        mask = labels==lab
        plt.scatter(Xproj[mask,0], Xproj[mask,1], label=f'Cluster {lab}')
    plt.scatter(centers_proj[:,0], centers_proj[:,1], marker='X', s=100)
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.title(f'K-Means k={k} (PCA projection)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'fig_kmeans_k{k}.png', dpi=200)
    plt.close()
