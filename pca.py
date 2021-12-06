#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
from sklearn.decomposition import PCA


# In[16]:


def PCA_my(matrix):
    Mat = np.array(matrix, dtype='float64')
    Mat_new = Mat - np.mean(Mat, axis=0)
    C = np.dot(Mat_new.T, Mat_new) / (Mat_new.shape[0] - 1)
    eig_vals, eig_vecs = np.linalg.eig(C)
    print(eig_vals)
    X_pca = np.dot(Mat_new, eig_vecs[:,[0,2]])
    print(X_pca)
    


# In[17]:


matrix = [[7,4,3,4],[4,1,8,3],[6,3,5,2],[8,3,2,10],[4,5,0,9],[1,3,2,5],[6,6,3,2],[8,3,3,6]]

PCA_my(matrix)


# In[18]:


pca = PCA(2)
pca.fit_transform(matrix)


# In[23]:


def PCA_SVD(matrix):
    Mat = np.array(matrix, dtype='float64')
    Mat_new = Mat - np.mean(Mat, axis=0)
    U, Sigma, Vh = np.linalg.svd(Mat_new, full_matrices=False, compute_uv=True)
    print(np.square(Sigma) / (Mat_new.shape[0] - 1))
    X_pca_svd = np.dot(Mat_new, Vh.T[:, [0, 1]])
    print(X_pca_svd)
    
    


# In[24]:


PCA_SVD(matrix)


# In[ ]:




