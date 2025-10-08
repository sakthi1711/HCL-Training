import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

X, y_true = make_blobs(n_samples=300, centers=4, n_features=6, random_state=42)  #X -> conatins (300,6) array y_true -> contains lables for each data point

pca = PCA(n_components = 2)    #reduce to 2 dimensions
x_reduced = pca.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(x_reduced)

y_kmeans = kmeans.predict(x_reduced)

plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title("KMeans Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
