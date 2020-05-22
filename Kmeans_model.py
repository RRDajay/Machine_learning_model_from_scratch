# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:00:00 2020

@author: ryan_
"""
import numpy as np
import random as rd
import math
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs # for data cluster generation only

# Fake Data
# x.shape -> (m,n)
x, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=100, cluster_std=2)
plt.scatter(x[:,0],x[:,1],c='black',label='unclustered data')
plt.title('Plot of data points')
plt.legend()
plt.show()

def calc_distance(x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a*a + b*b))
    return d

def optimalK(wcss_array, K_array):
    
    a = wcss_array[0] - wcss_array[-1]
    b = K_array[-1] - K_array[0]
    c1 = K_array[0] * wcss_array[-1]
    c2 = K_array[-1] * wcss_array[0]
    c = c1 - c2
    
    dist_of_points_from_line = []
    
    for k in range(len(K_array)):
        dist_of_points_from_line.append(calc_distance(K_array[k], wcss_array[k], a, b, c))
        
    plt.plot(K_array,wcss_array)
    plt.plot([K_array[0],K_array[-1]], [wcss_array[0],wcss_array[-1]], '-ro')
    plt.xlabel('Number of Clusters')
    plt.ylabel('within-cluster sums of squares')
    plt.title('Elbow method')
    plt.show()
    
    plt.plot(K_array, dist_of_points_from_line)
    plt.show()
    
    print('Optimum K is {}'.format(np.argmax(dist_of_points_from_line)+1))
    return (np.argmax(dist_of_points_from_line)+1)

def centroid_initialization(data, K): 
    i=rd.randint(0,data.shape[0])-1
    Centroid=np.array([data[i]])
    for k in range(1,K):
        D=np.array([]) 
        for x in data:
            D=np.append(D,np.min(np.sum((x-Centroid)**2)))
        prob=D/np.sum(D)
        cummulative_prob=np.cumsum(prob)
        r=rd.random()
        i=0
        for j,p in enumerate(cummulative_prob):
            if r<p:
                i=j
                break
        Centroid=np.append(Centroid,[data[i]],axis=0)
        
    return Centroid.transpose()

def kmeans(data, *, optimumK=3): #actual k-means model
  
    m, n, n_iter, K = x.shape[0], x.shape[1], 100, optimumK
    centroids=np.array([]).reshape(n,0)
    eucldistance = np.array([]).reshape(m,0)
    
    for i in range(n_iter): #optimization  
          eucldistance=np.array([]).reshape(m,0)
          for k in range(K):
              if i == 0: 
                  # rand = rd.randint(0,m-1) #
                  # centroids = np.c_[centroids,x[rand]] #random centroid at first
                   centroids = centroid_initialization(data, K)
                  
              elif i != 0:
                  for j in range(K):
                      centroids[:,j]=np.mean(Y[j+1],axis=0)
                
              eucldistance=np.c_[eucldistance,np.sum((x-centroids[:,k])**2, axis=1)]
          C=np.argmin(eucldistance,axis=1)+1
          
          Y={} #dictionary containing points at their nearest centroids
          
          for j in range(K):
            Y[j+1]=np.array([]).reshape(2,0)
          for i in range(m):
            Y[C[i]]=np.c_[Y[C[i]],x[i]]
             
          for j in range(K):
            Y[j+1]=Y[j+1].T
            centroids[:,j]=np.mean(Y[j+1],axis=0)

    return Y, K, centroids

def kmeans_iter(data): # code to get optimum value for K clusters
    
    wcss_array=np.array([])
    
    for K in range(1,6):
        Y, K, centroids = kmeans(data, optimumK=K)    
        
        wcss=0

        for j in range(K):
            wcss+=np.sum((Y[j+1]-centroids[:,j])**2) # get within-cluster sums of square
                
        wcss_array=np.append(wcss_array,wcss)
        
    
    return Y, K, centroids, wcss_array

#find optimum K
K_array=np.arange(1,6,1)
output, K, centroids, wcss_array = kmeans_iter(x)

#Kmeans model with optimum K clusters
output, K, centroids = kmeans(x, optimumK=optimalK(wcss_array, K_array))

#output -> dictionary that contains points belonging to cluster
#K -> number of clusters
#centroids -> data points of centroids for clusters

color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(K):
    plt.scatter(output[k+1][:,0],output[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(centroids[0,:],centroids[1,:],s=300,c='yellow',label='Centroids')
plt.legend()
plt.show()