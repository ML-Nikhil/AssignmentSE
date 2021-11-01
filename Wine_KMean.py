# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:27:47 2021

@author: Nikhil J
@github: ML-Nikhil

Problem Statement : Clustering wine with Alcohol and Color Intensity
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Importing Wine
df = pd.read_csv('D:/SEML/Dataset/wine.csv')
df1 = df[[' Alcohol','Color intensity']]
# Creating an Array
X  =df.iloc[:,[0,9]].values
df1.describe()
plt.plot(df['Color intensity'])
# Without Scaling the Input

# Applying Elbow Method

wcss =[]  
for i in range(1,11):
    kmeans = KMeans(n_clusters =i,init ='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss,marker='x')

# Performing Cluster with optimal K -4
kmeans = KMeans(n_clusters =4, init ='k-means++',random_state =0)
y_kmeans = kmeans.fit_predict(X)

# Concatenating with dataset
km = pd.DataFrame(y_kmeans)
df1 = pd.concat([df1,km],axis=1)
# to find the centroids of clusters
cent = kmeans.cluster_centers_

# Plotting Cluster 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'orange',label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue',  label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'black', label = 'Cluster 4')
plt.scatter(cent[:,0],cent[:,1],c ='red',marker ='o',s=300)
plt.title('Cluster of Wine _without Scaling')
plt.xlabel('Alcohol')
plt.ylabel('Color intensity')
#____________________________________________________________

#_________________Scaling Input_______________________________

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()

scaler.fit(X)

x =scaler.transform(X)
# Changes after Transformation
plt.plot(df['Color intensity'])
plt.plot(x[:,1])
# Changes in Alcohol
plt.plot(df[' Alcohol'])
plt.plot(x[:,0])
wss =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters =i,init ='k-means++',random_state=0)
    kmeans.fit(x)
    wss.append(kmeans.inertia_)
plt.plot(range(1,11),wss,marker='x')


# Performing Clustering
kmeans = KMeans(n_clusters =4, init ='k-means++',random_state =0)
y_kmeanss = kmeans.fit_predict(x)
centr = kmeans.cluster_centers_
# Plotting Clusters
plt.scatter(x[y_kmeanss == 0, 0], x[y_kmeanss == 0, 1], s = 100, c = 'orange', label = 'Cluster 1')
plt.scatter(x[y_kmeanss == 1, 0], x[y_kmeanss == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeanss == 2, 0], x[y_kmeanss == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeanss == 3, 0], x[y_kmeanss == 3, 1], s = 100, c = 'black', label = 'Cluster 4')
plt.scatter(centr[:,0],centr[:,1],c ='red',marker ='o',s=300)
plt.title('Cluster of Wine_ with Scaling')
plt.xlabel('Alcohol')
plt.ylabel('Color intensity')

# Comparative plot

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.scatter(X[:,0],X[:,1],c =y_kmeans,s=100,cmap='viridis')
ax1.scatter(cent[:,0],cent[:,1],c ='red',marker ='o',s=150)
ax1.set_title('Cluster without Scaling Inputs')
ax2.scatter(x[:,0],x[:,1],c =y_kmeanss,s=100,cmap='viridis')
ax2.scatter(centr[:,0],centr[:,1],c ='red',marker ='o',s=150)
ax2.set_title('Cluster with Scaling Inputs')
#___________________________Silhoutte Scores
# Optimal Numbers Can Also be selected using Silhoutte Scores
# For More Precision Silhoutte Score is used

from sklearn.metrics import silhouette_score
sil =[]
for i in range(2,11):
     kmeans = KMeans(n_clusters = i,init='k-means++',random_state=0)
     kmeans.fit(x)
     cluster_labels = kmeans.labels_
     sil.append(silhouette_score(x,cluster_labels))
     
kmeans = KMeans(n_clusters =3,init = 'k-means++',random_state=0)
y_sil = kmeans.fit_predict(x)    
cents = kmeans.cluster_centers_
plt.scatter(x[:,0],x[:,1],c =y_sil,s=100,cmap='viridis')
plt.scatter(cents[:,0],cents[:,1],c ='red',marker ='o',s=300)








