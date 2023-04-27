#!/usr/bin/env python
# coding: utf-8

# In[18]:


#im using the heart disease database instead of diabetes
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 


# In[2]:


df = pd.read_csv("heart.csv")


# In[5]:


df.shape 


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info


# In[9]:


df. describe()


# In[10]:


df.isnull()


# In[11]:


df.isnull().sum()



# In[12]:


df["target"].value_counts()


# In[13]:


#so 526 people HAVE heart disease 


# In[14]:


#and 499 people dont


# In[16]:


sns.countplot(df["target"])
plt.xlabel("target")
plt.ylabel("count of target")
plt.title("target variable count plot")
plt.show()


# In[17]:


# it should show two graphs but it isnt please help 


# In[21]:


#splitting the dataframe
 
X= df.iloc[:, :-1] #taking all the columns except the target column

Y= df.iloc[:, -1]# taking only the target column 


# In[23]:


X.shape


# In[24]:


Y.shape


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=99)


# In[28]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion = "gini",
                             max_depth = 8,
                             min_samples_split= 10,
                            random_state= 5)


# In[29]:


clf.fit(X_train, Y_train)


# In[30]:


clf.feature_importances_


# In[31]:


df.columns


# In[32]:


Y_pred = clf.predict(X_test)


# In[33]:


Y_pred


# In[34]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)


# In[35]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred)


# In[38]:


from sklearn.model_selection import cross_val_score 
cross_val_score(clf, X_train, Y_train, cv=10)


# In[39]:


#very high accuracy


# In[40]:


from sklearn.metrics import classification_report 
print(classification_report(Y_pred, Y_test))


# In[42]:


features = df.columns
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.title('Features importances')
plt.barh(range(len(indices)), importances[indices], color='b', )
plt.yticks(range(len(indices)),[features[i] for i in indices])
plt.xlabel('relative importance')
plt.show()


# In[43]:


#cp holds the most importance 


# In[ ]:




