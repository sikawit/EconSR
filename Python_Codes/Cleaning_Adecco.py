#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', -1)


# In[2]:


df_adec = pd.read_csv("/Users/sikawit/OneDrive/Chula/2018-2/Research/Code_upload/adecco.csv")


# In[19]:


df_adec


# In[20]:


df_adec["newgrad_d"] = np.where(df_adec['newgrad']>=0, 1, 0)
df_adec["junior_d"] = np.where(df_adec['junior']>=0, 1, 0)
df_adec["senior_d"] = np.where(df_adec['senior']>=0, 1, 0)


# In[21]:


df_adec


# In[22]:


main_df = df_adec.drop(["newgrad", "junior", "senior", "newgrad_d", 'junior_d', "senior_d", "wage"], axis =1)
wage_df = df_adec[["newgrad", "junior", "senior"]]


# In[24]:


pd.melt(wage_df)


# In[25]:


df2 = pd.melt(df_adec[["newgrad","junior","senior"]])


# In[26]:


df1 =pd.concat([main_df]*3, ignore_index=True)


# In[28]:


clustered_df =pd.concat([df1, df2], axis=1, join='inner')


# In[47]:


clustered_df.to_csv("clustered_adecco.csv")


# In[30]:


clustered_df.head()


# In[32]:


clustered_df=clustered_df[clustered_df.value != -1]


# In[36]:


clustered_df=clustered_df.reset_index()


# In[37]:


clustered_df


# In[40]:


dummy = pd.get_dummies(clustered_df.variable, drop_first=True)


# In[42]:


clustered_df = pd.concat([clustered_df, dummy], axis =1)


# In[46]:


clustered_df


# In[ ]:




