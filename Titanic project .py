#!/usr/bin/env python
# coding: utf-8
VARIABLE       DESCRIPTION            KEY

survival 	   Survival 	          0 = No, 1 = Yes
pclass 	       Ticket class 	      1 = 1st, 2 = 2nd, 3 = 3rd
sex 	       Sex 	
Age 	       Age in years 	
sibsp 	       # of siblings / spouses aboard the Titanic 	
parch 	       # of parents / children aboard the Titanic 	
ticket 	       Ticket number 	
fare 	       Passenger fare 	
cabin 	       Cabin number 	
embarked 	   Port of Embarkation 	  C = Cherbourg, Q = Queenstown, S = Southampton



Variable Notes

pclass: A proxy for socio-economic status (SES)
1st    = Upper
2nd    = Middle
3rd    = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[4]:


data = pd.read_csv('titanic_data.csv')


# In[5]:


data.head(10)


# In[6]:


data.info()


# In[8]:


data.isnull().sum()


# In[13]:


plt.figure(figsize = (12, 10))
heatmap = sns.heatmap(data[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot = True)//correlation graph which shows that how parametera are correlated with each other


# In[ ]:





# # Sibsp

# In[14]:


data['SibSp'].unique()


# In[15]:


data['SibSp'].nunique()


# In[16]:


sns.factorplot(x = "SibSp", y = "Survived", data = data, kind = "bar", size = 8)


# # Age

# In[19]:


age_visual = sns.FacetGrid(data, col = "Survived", size = 7)
age_visual = age_visual.map(sns.distplot, "Age")
age_visual = age_visual.set_ylabels("survived_probability")


# # Sex

# In[22]:


plt.figure(figsize= (12, 10))

sex_plot = sns.barplot(x = "Sex", y = "Survived", data = data)
sex_plot = sex_plot.set_ylabel("survived_probability")


# In[27]:


data[["Sex", "Survived"]].groupby("Sex").mean()


# In[28]:


sns.factorplot(x = "Pclass", y = "Survived", data = data, kind = "bar", size = 8)


# In[29]:


sns.factorplot(x = "Pclass", y = "Survived", hue = "Sex", data = data, kind = "bar", size = 8)


# In[ ]:





# In[30]:


data["Embarked"].isnull().sum()


# In[31]:


data["Embarked"].value_counts()


# In[32]:


data["Embarked"] = data["Embarked"].fillna("S")


# In[33]:


g = sns.factorplot(x="Embarked", y="Survived", data=data, size=7, kind="bar")


# there is a reason for this: pclas, sex, age

# # Preparing data

# In[34]:


data = pd.read_csv('titanic_data.csv')


# In[35]:


data.head()


# In[36]:


data.info()


# In[37]:


data.describe()


# In[40]:


mean = data['Age'].mean()
std = data['Age'].std()

is_null = data['Age'].isnull().sum()

rand_age = np.random.randint(mean - std, mean+std, size = is_null)

age_slice = data["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
data["Age"] = age_slice   


# In[41]:


data["Age"].isnull().sum()


# In[42]:


data["Embarked"].isnull().sum()


# In[43]:


data["Embarked"] = data["Embarked"].fillna("S")


# In[44]:


col_to_drop = ['PassengerId','Cabin', 'Ticket','Name']
data.drop(col_to_drop, axis=1, inplace = True)


# In[45]:


data.head()


# In[46]:


genders = {"male": 0, "female": 1}
data['Sex'] = data['Sex'].map(genders)


# In[47]:


data.head()


# In[48]:


ports = {"S": 0, "C": 1, "Q": 2}

data['Embarked'] = data['Embarked'].map(ports)


# In[49]:


data.head()


# In[50]:


x = data.drop(data.columns[[0]], axis = 1)
y = data['Survived']


# In[51]:


x.head()


# In[61]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.30, random_state =0)


# In[ ]:





# In[63]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

xtrain = sc_x.fit_transform(xtrain) 
xtest = sc_x.transform(xtest)


# In[64]:


logreg = LogisticRegression()
svc_classifier = SVC()
dt_classifier = DecisionTreeClassifier()
knn_classifier = KNeighborsClassifier(5)
rf_classifier = RandomForestClassifier(n_estimators=1000)


# In[65]:


logreg.fit(xtrain, ytrain)
svc_classifier.fit(xtrain, ytrain)
dt_classifier.fit(xtrain, ytrain)
knn_classifier.fit(xtrain, ytrain)
rf_classifier.fit(xtrain, ytrain)


# In[66]:


logreg_ypred = logreg.predict(xtest)
svc_classifier_ypred = svc_classifier.predict(xtest)
dt_classifier_ypred = dt_classifier.predict(xtest)
knn_classifier_ypred = knn_classifier.predict(xtest)
rf_classifier_ypred = rf_classifier.predict(xtest)


# In[67]:


logreg_ypred


# In[69]:


xtest


# In[71]:


from sklearn.metrics import accuracy_score


# In[72]:


logreg_acc = accuracy_score(ytest, logreg_ypred)
svc_classifier_acc = accuracy_score(ytest, svc_classifier_ypred)
dt_classifier_acc = accuracy_score(ytest, dt_classifier_ypred)
knn_classifier_acc = accuracy_score(ytest, knn_classifier_ypred)
rf_classifier_acc = accuracy_score(ytest, rf_classifier_ypred)


# In[73]:


print ("Logistic Regression : ", round(logreg_acc*100, 2))
print ("Support Vector      : ", round(svc_classifier_acc*100, 2))
print ("Decision Tree       : ", round(dt_classifier_acc*100, 2))
print ("K-NN Classifier     : ", round(knn_classifier_acc*100, 2))
print ("Random Forest       : ", round(rf_classifier_acc*100, 2))


# In[ ]:




