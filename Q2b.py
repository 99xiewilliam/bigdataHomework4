#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[34]:


data = pd.read_csv('trainImage.txt', header=None)
print(data.shape)
df = data.values


# In[49]:


def pca_my(n_components, df):
#     X = StandardScaler().fit_transform(data)
#     pca = PCA(n_components)
#     X_pca = pca.fit_transform(X)
    Mat_new = df - np.mean(df, axis=0)
    C = np.dot(Mat_new.T, Mat_new) / (Mat_new.shape[0] - 1)
    print(C)
    eig_vals, eig_vecs = np.linalg.eig(C)
    #largestIndexes = map(list(eig_vals).index, heapq.nlargest(n_components, range(eig_vals), eig_vals))
    largestIndexes = np.argsort(eig_vals)[::-1][:n_components]
    print(largestIndexes)
    X_pca = np.dot(Mat_new, eig_vecs[:,largestIndexes])
    return eig_vecs[:,largestIndexes], X_pca


# In[50]:


eig_vecs, X_pca = pca_my(20, df)


np.savetxt('123.txt', eig_vecs.T)


# In[51]:


np.savetxt('trainImageNew.txt', X_pca)


# In[ ]:




