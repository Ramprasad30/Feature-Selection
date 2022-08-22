#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("mobile_dataset.csv")


# In[2]:


df.head()


# #### Univariate Selection

# In[3]:


x = df.iloc[:,:-1]
y = df['price_range']


# In[5]:


x.head()


# In[6]:


y.head()


# In[7]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[9]:


### Apply SelectKBest Algorithm
ordered_rank_features=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_features.fit(x,y)


# In[11]:


dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(x.columns)


# In[12]:


features_rank=pd.concat([dfcolumns,dfscores],axis=1)


# In[13]:


features_rank.columns=['Features','Score']
features_rank


# In[14]:


features_rank.nlargest(10,'Score')


# #### Feature Importance
# #### This technique gives you a score for each feature of your data,the higher the score mor relevant it is

# In[16]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(x,y)


# In[17]:


print(model.feature_importances_)


# In[19]:


ranked_features=pd.Series(model.feature_importances_,index=x.columns)
ranked_features.nlargest(10).plot(kind='barh')
plt.show()


# #### Correlation

# In[20]:


df.corr()


# In[21]:


import seaborn as sns
corr=df.iloc[:,:-1].corr()
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(df[top_features].corr(),annot=True)


# In[22]:


threshold=0.8


# In[23]:


# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[24]:


correlation(df.iloc[:,:-1],threshold)


# #### Information Gain

# In[25]:


from sklearn.feature_selection import mutual_info_classif


# In[26]:


mutual_info=mutual_info_classif(x,y)


# In[28]:


mutual_data=pd.Series(mutual_info,index=x.columns)
mutual_data.sort_values(ascending=False)


# In[ ]:




