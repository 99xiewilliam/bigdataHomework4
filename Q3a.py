#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np


# In[11]:


def similarity(list1, list2, common_list1, common_list2):
    a_list1 = np.array(list1)
    a_list2 = np.array(list2)
    common_list1 = np.array(common_list1)
    common_list2 = np.array(common_list2)
    
    sum1 = np.sum((common_list1 - np.mean(a_list1))*(common_list2 - np.mean(a_list2)))
    sum2 = np.sqrt(np.sum((a_list1 - np.mean(a_list1))*(a_list1 - np.mean(a_list1)))) * np.sqrt(np.sum((a_list2 - np.mean(a_list2))*(a_list2 - np.mean(a_list2))))
    result = sum1 / sum2
    return result


# In[12]:


A = [2,5,2]
B = [1,2,3,4]
C = [5,4,4,1]
D = [4,3,1,5]
E = [3,5,4,3]
F = [4,2,2]

common_list1 = [3,4]
common_list2 = [2,5]

result = similarity(E, A, common_list1, common_list2)
print(result)


# In[13]:


common_list1 = [3,5,3]
common_list2 = [1,2,4]

result = similarity(E, B, common_list1, common_list2)
print(result)


# In[14]:


common_list1 = [3,4,3]
common_list2 = [5,4,1]

result = similarity(E, C, common_list1, common_list2)
print(result)


# In[15]:


common_list1 = [3,5,4]
common_list2 = [4,3,1]

result = similarity(E, D, common_list1, common_list2)
print(result)


# In[16]:


common_list1 = [5,4,3]
common_list2 = [4,2,2]

result = similarity(E, F, common_list1, common_list2)
print(result)


# In[17]:


def forcastStar(similar, stars):
    stars = np.array(stars)
    similar = np.array(similar)
    
    sum1 = np.sum(stars * similar)
    sum2 = np.sum(similar)
    
    r = sum1 / sum2
    
    return r


# In[19]:


similar = [0.3077287274483319,0.17588161767036212]
stars = [2, 4]

r = forcastStar(similar, stars)
print(r)


# In[20]:


U1 = [2,1,5,4,3]
U2 = [2,3,5,4]
U3 = [5,4,1,4,2]
U4 = [2,3,4,5]
U5 = [4,1,3,2]

common_list1 = [2,3,4,5]
common_list2 = [2,1,5,4]

result = similarity(U4, U1, common_list1, common_list2)
print(result)


# In[21]:


common_list1 = [3,5]
common_list2 = [2,3]

result = similarity(U4, U2, common_list1, common_list2)
print(result)


# In[22]:


common_list1 = [2,4,5]
common_list2 = [5,4,1]

result = similarity(U4, U3, common_list1, common_list2)
print(result)


# In[23]:


common_list1 = [3,4]
common_list2 = [4,1]

result = similarity(U4, U5, common_list1, common_list2)
print(result)


# In[24]:


similar = [0.7071067811865475,0.0]
stars = [3, 5]

r = forcastStar(similar, stars)
print(r)


# In[ ]:




