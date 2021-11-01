# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 17:22:27 2021

@author: Nikhil J
@github: ML-Nikhil
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Import Dataset

df = pd.read_csv('D:/SEML/Dataset/wine.csv')

# Data for Clustering

X = df.iloc[:,[0,9]].values

# linkage:groups pair objects into clusters based on their similarity
# ward Minimum Variance : overall minimises the total within cluster variance
# each pair with minimum cluster distance will be merged

dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Wine')
plt.ylabel('Euclidean distances')

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# Plotting Clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'orange', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'black', label = 'Cluster 4')

# Standardising Dataset Based on Z score

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
x = scaler.transform(X)

dend = sch.dendrogram(sch.linkage(x,method='ward'))
hcs = AgglomerativeClustering(n_clusters=4,affinity ='euclidean',linkage = 'ward')
y_sh = hcs.fit_predict(x)

plt.scatter(x[:,0],x[:,1],c=y_sh,s=100,cmap='rainbow')
plt.hist(x_n)













