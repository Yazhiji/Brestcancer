#!/usr/bin/env python
# coding: utf-8

# IMPORTING DATA AND VISUALIZING
#  

# In[3]:


pip install seaborn


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer=load_breast_cancer()


# In[4]:


cancer


# In[5]:


cancer.keys()


# In[7]:


print(cancer['DESCR'])


# In[9]:


print(cancer['target'])


# In[10]:


print(cancer['target_names'])


# In[11]:


print(cancer['feature_names'])


# In[12]:


cancer['data'].shape


# In[17]:


#inorder to deal with data easier we use dataframe - makes  10times easier when comes to data
df_cancer=pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))
#take cancer data and target data - including 30 columns and addition clm includes targert data


# In[18]:


df_cancer.head() #few rows and columns


# #VISUALIZING THE DATA

# In[20]:


df_cancer.tail()


# In[21]:


df_cancer.tail()


# In[23]:


sns.pairplot(df_cancer, vars = ['mean radius','mean texture','mean area','mean perimeter','mean smoothness']) # vars= what variables we need


# In[24]:


#prolem is the graph doesnt show targer class so we add a column hue which select the column target
sns.pairplot(df_cancer,hue='target', vars = ['mean radius','mean texture','mean area','mean perimeter','mean smoothness']) 


# In[30]:


sns.countplot(df_cancer['target']) # to count the samples


# In[32]:


sns.scatterplot(x='mean area',y='mean smoothness', hue='target',data=df_cancer)
#plot specific 2 variables


# In[35]:


plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(),annot=True)
# 30 features in x and y axis 
#correlation is plotted


# #MODEL TRAINING 

# In[36]:


#DEFINE X AND Y VALUES IP AND OP RESP
X=df_cancer.drop(['target'],axis=1) #need entire df except target frame
X


# In[37]:


Y=df_cancer['target'] #only target columns


# In[39]:


#split data into train and testing
#shit tab to vie what train_test_split works
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=5)


# In[40]:


from sklearn.svm import SVC


# In[41]:


from sklearn.metrics import classification_report, confusion_matrix


# In[42]:


SVC_model=SVC()


# In[43]:


#perform train we use fit method to train the model using svm(support vector machine)
SVC_model.fit(X_train,Y_train) # input and op data


# #EVALUATING THE MODEL

# In[44]:


#after training the model and has to tested using another datasets called testing data
#the data the model hasnt seen before during training
#model generalization - we wanted the ml stratiges do not train for this data only we need it to be general
#overfittet model - the model has learnt only the training data
#generaliszed model - model generalization() - works btr



# In[45]:


#confusion matrix - colum true class , row prediction
# ct=true,ct=flase(good),c(t)t(f)and c(f)t(t)=not good however not worst
#type1error- prediction tells us the patient haas the disease but really he doesnt
#type2error - big pgm avoid 


# In[46]:


y_predict=SVC_model.predict(X_test)


# In[47]:


y_predict


# In[48]:


#call cm
cm=confusion_matrix(Y_test,y_predict)


# In[50]:


sns.heatmap(cm,annot=True)


# In[73]:


#data nomalization : x'=x-xmin/xmax-xmin
#unity based normalization (btwn 0 and 1)- feature scaling
#SVM PARA - C parameter - smooth decision boundary- like penality
#small c - generalize ml , larger c - overfitted model
# SVM PARAA - gamma parameter - specifies the control on how fat the onf of train set reaches
#larger gamma close reach , small gamma far reach
print(classification_report(Y_test,y_predict))


# #IMPROVEMENT 1

# In[52]:


#perform normalization
#get min value
#feature scaling = unity normalization range is btwn 0to1
min_train=X_train.min()
range_train=(X_train-min_train).max()
X_train_scaled=(X_train-min_train)/range_train


# In[53]:


sns.scatterplot(x=X_train['mean area'],y=X_train['mean smoothness'],hue=Y_train)


# In[54]:


sns.scatterplot(x=X_train_scaled['mean area'],y=X_train_scaled['mean smoothness'],hue=Y_train)


# In[55]:


min_test=X_test.min()
range_test=(X_test-min_test).max()
X_test_scaled=(X_test-min_test)/range_test


# In[56]:


SVC_model.fit(X_train_scaled,Y_train)


# In[58]:


y_predict=SVC_model.predict(X_test_scaled)
cm=confusion_matrix(Y_test,y_predict)


# In[59]:


sns.heatmap(cm,annot=True)


# In[60]:


print(classification_report(Y_test,y_predict))


# #IMPROVEMENT 2

# In[65]:


param_grid={'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV


# In[66]:


grid= GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[67]:


grid.fit(X_train_scaled,Y_train)


# In[68]:


#get best values
grid.best_params_


# In[69]:


#plot cm with the best values'optimise values
grid_predict=grid.predict(X_test_scaled)


# In[70]:


cm=confusion_matrix(Y_test,grid_predict)


# In[71]:


sns.heatmap(cm,annot=True)


# In[72]:


print(classification_report(Y_test,grid_predict))


# In[ ]:




