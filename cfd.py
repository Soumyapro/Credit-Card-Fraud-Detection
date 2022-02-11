#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("creditcard.csv")
df.head(10) 


# In[3]:


#printing all information about our dataset
df.info()


# In[4]:


#All the cloumn has non-null values.So we don't need to modify it


# In[5]:


#printing all the columns of dataset
df.columns


# In[6]:


#distribution of legit and fraud transactions.
df['Class'].value_counts()


# In[7]:


# 0---->legit transactions
# 1---->Fraud transactions


# In[8]:


#separating the two classes
legit_trans = df[df.Class==0]
fraud_trans = df[df.Class==1]


# In[9]:


print(legit_trans.shape)
print(fraud_trans.shape)


# In[10]:


legit_trans.describe()


# In[11]:


fraud_trans.describe()


# In[12]:


#As we can see that our dataset is highly unbalanced.So, we will use under-sampling.
legit_new = legit_trans.sample(n=492)
new_df = pd.concat([legit_new,fraud_trans],axis=0) #axis=0 because we want to add thses two row wise not column wise.


# In[13]:


print(new_df.shape)


# In[14]:


new_df.head(10)


# In[15]:


print(new_df['Class'].value_counts())


# In[16]:


#we will split the data into fetures and target
X = new_df.drop(columns='Class',axis=1)
y = new_df['Class']


# In[17]:


#printing X
print(X)


# In[18]:


#printing y
print(y)


# In[19]:


#Using the StandardScaler to scale the data
scaler = StandardScaler()
amount = X['Amount'].values
X['Amount'] = scaler.fit_transform(amount.reshape(-1,1))


# In[27]:


#We have dropped Time column because it's an external factor.
print(X.columns)


# In[29]:


#splitting the dataset into training and testing dataset.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)


# In[32]:


model = svm.SVC(kernel='linear')
model.fit(X_train,y_train)


# In[33]:


#predicting our dataset
y_pred = model.predict(X_test)


# In[35]:


print(classification_report(y_test,y_pred))


# In[37]:


#confusion matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[44]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels = model.classes_)
disp.plot()
plt.show()


# In[47]:


accuracy = accuracy_score(y_test,y_pred)
print("The accuracy score of our model is {}".format(accuracy))


# In[ ]:




