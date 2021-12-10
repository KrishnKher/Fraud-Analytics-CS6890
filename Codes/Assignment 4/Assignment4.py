#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Bharat Chandra Mukkavalli (ES19BTECH11016), Krishn Vishwas Kher (ES19BTECH11015), Sayantan Biswas (AI19BTECH11015).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
import seaborn as sns


# Reading the data.
data = pd.read_csv("spectral.csv")

N = len(data)
col_header = data.columns
Laplacian = []
D = []

# Initialising the matrix.
for row_index in range(N):
    Laplacian.append([0 for _ in range(N)])
    D.append([0 for _ in range(N)])

for row_index in range(N):
    diag_ele = 0
    curr_point = [data[col_header[index]][row_index] for index in range(len(col_header))]
    for another_row_index in range(N):
        if another_row_index != row_index:
            
            another_point = [data[col_header[index]][another_row_index] for index in range(len(col_header))]
            distance = math.dist(curr_point, another_point)
            Laplacian[another_row_index][row_index] = Laplacian[row_index][another_row_index] = (-1.0)*distance
            

    for another_row_index in range(N):
        if row_index != another_row_index: # Not necessary actually, but for the correctness 
            diag_ele += Laplacian[row_index][another_row_index]
    diag_ele *= (-1.0)
    Laplacian[row_index][row_index] = diag_ele
    D[row_index][row_index] = 1.0/((diag_ele)**(0.5))
    

Laplacian = np.array(Laplacian)
D = np.array(D)
Laplacian = D @ (Laplacian @ D)
evals, evecs = np.linalg.eig(Laplacian)
evecvals = []
evecs = np.transpose(evecs)

for vector in range(len(evecs)):
    e_pair = [evecs[vector], evals[vector]]
    evecvals.append(e_pair)

evecvals = sorted(evecvals, key=lambda element: element[1])


sorted_evals = evals
sorted_evals.sort()


plt.title("Sorted eigen values of the Laplacian.")
plt.plot(range(len(sorted_evals)), sorted_evals)
plt.show()

principal_eigen_vectors = []

for vec_index in range(2):
   principal_eigen_vectors.append(principal_eigen_vectors[vec_index][0])
    
principal_eigen_vectors = np.transpose(principal_eigen_vectors )
data = np.array(data)

print(principal_eigen_vectors.shape())
transformed_data = data @ principal_eigen_vectors 

kmeans_clusterer = KMeans(n_clusters = 2, random_state=16).fit_predict(transformed_data)
sns.scatterplot(transformed_data[0], transformed_data[1], hue=kmeans_clusterer)
plt.title("Detecing outliers through clusters.")
plt.show()

