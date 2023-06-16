#!/usr/bin/env python
# coding: utf-8

# In[73]:


#importing initial libraries
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns


# In[58]:


#calling our dataset and naming it 'data'
data = pd.read_excel('null.xlsx')


# In[60]:


data


# In[61]:


data.head(6)


# In[62]:


data.tail(6)


# In[63]:


# descriptive analysis
# for independent variables

data.describe()


# In[64]:


#visuals
#box plots to check for outliers per variable

plt.boxplot(data['Temperature°C'])
plt.title('Temperature°C')
plt.grid('True')
plt.show()

plt.boxplot(data['Dew Point°C'])
plt.title('Dew Point°C')
plt.grid('True')
plt.show()


# In[65]:


plt.boxplot(data['Humidity%'])
plt.title('Humidity%')
plt.grid('True')
plt.show()

plt.boxplot(data['Wind speed Kph'])
plt.title('Wind speed Kph')
plt.grid('True')
plt.show()


# In[66]:


plt.boxplot(data['Pressure Hg'])
plt.title('Pressure Hg')
plt.grid('True')
plt.show()

plt.boxplot(data['Precipitation mm'])
plt.title('Precipitation mm')
plt.grid('True')
plt.show()


# In[67]:


#calculating the SPI//STANDARD PRECIPITATION INDEX

# Calculating the long-term average precipitation
long_term_average = sum(data['Precipitation mm']) / len(data['Precipitation mm'])

# Calculating the standard deviation of the long-term average precipitation
standard_deviation = np.std(data['Precipitation mm'])

# Calculating the SPI for each month
spi = [0 for i in range(len(data['Precipitation mm']))]
for i in range(len(data['Precipitation mm'])):
    spi[i] = (data['Precipitation mm'][i] - long_term_average) / standard_deviation

# Print the SPI values
print(spi)


# In[68]:


# creating a new column called 'spi'
data['spi'] = spi


# In[69]:


data['spi']


# In[71]:


# Creating a new column called "Rain_Draught (dependent variable)"//If the spi
# is less than zero we term it as draught and 
# if it greater than zero we term it as rain

data['Rain_Draught'] = data['spi'].apply(lambda x: 'Rain' if x > 0 else 'Draught')

# Print the DataFrame
print(data)


# In[97]:


#changing variable Rain_Draught to binary format 
data['Rain_Draught'] = data['Rain_Draught'].apply(lambda d: 1 if d == 'Rain' else 0)


# In[98]:


data


# In[101]:


#Cont. of Univariate Analysis

plt.style.use('ggplot')
figure, ax1 = plt.subplots(1,3)
data['Rain_Draught'].value_counts(normalize = True).plot(ax = ax1[0],figsize=(22, 7),kind = 
'bar',title = 'Rain_Draught', color = "r", rot = 0)
data['Temperature°C'].value_counts(normalize = True).plot(ax = ax1[1],kind = 
'bar',title = 'Temperature°C', color = "b", rot = 0)
data['Wind speed Kph'].value_counts(normalize = True).plot(ax = ax1[2], kind = 'bar', title = 
'Wind speed Kph', color = "g", rot = 0)
figure.tight_layout()


# In[102]:


#Bivariate Analysis
sns.countplot(y = 'Temperature°C', hue = 'Rain_Draught', data = data)


# In[105]:



sns.countplot(y = 'Dew Point°C', hue = 'Rain_Draught', data = data)


# In[112]:


sns.countplot(y = 'Pressure Hg', hue = 'Rain_Draught', data = data)


# In[113]:


sns.countplot(y = 'Wind speed Kph', hue = 'Rain_Draught', data = data)


# In[114]:


#checking data information after analysis
data.info()


# In[115]:


#Since machine learning algorithms take
#two variables for prediction
#the dependent variable (Rain_Draught) is declared
#as Y, whilst Temp, Dew Point,
#Humidity, Wind speed, and Pressure 
#are merged into one variable X

#Firstly, we have to change Humidity to a float, from %, thus
#we divide it by 100

data['Humidity%'] = data['Humidity%']/100


# In[116]:


data['Humidity%']


# In[117]:


data.info()


# In[118]:


#variable merging

X = data.iloc[:, np.r_[0,1,2,3,4]].values
Y = data.iloc[:,7].values


# In[119]:


Y


# In[120]:


X


# In[121]:


#data splitting for training and testing
#data will be split into 80%:20% respectively

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)


# In[122]:


#Normalizing the dataset by standardizing it 
#to have a mean of 0 and a standard deviation of 1

from sklearn.preprocessing import StandardScaler
da = StandardScaler()
X_train = da.fit_transform(X_train)
X_test = da.transform(X_test)


# In[146]:


#Building the Model KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
KNN = KNeighborsClassifier(n_neighbors = 2)
KNN.fit(X_train, Y_train)
Y_pred = KNN.predict(X_test)
Y_pred
kn_prob = KNN.predict_proba(X_test)
Confusion_Matrix = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix\n')
print(Confusion_Matrix)


print(classification_report(Y_test.reshape(-1, 1), Y_pred))

#sensitivity

sensitivity = Confusion_Matrix[0,0]/(Confusion_Matrix[0,1] + Confusion_Matrix[0,0])
print('Sensitivity is', sensitivity)


# In[167]:


cm = [[664, 11],
     [54 ,81]]
classes = ['0', '1']

df_cm = pd.DataFrame(cm, index = classes, columns = classes)
plt.figure(figsize = (9, 7))
cm_plot = sns.heatmap(df_cm,fmt = 'd',cmap = 'Blues',annot = True)
cm_plot.figure.savefig('cm_png')


# In[147]:


#Calculating the AUC-ROC of KNN
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

kn_prob = kn_prob[:,1]


# In[148]:


kn_auc = roc_auc_score(Y_test, kn_prob)


# In[149]:


print('K-Nearest Neighbor:AUROC = % 0.3f' %(kn_auc))


# In[150]:


#Predicting the test set results
kn_fpr,kn_tpr , _ = roc_curve(Y_test, kn_prob)


# In[151]:


#plotting the AUCROC
plt.plot(kn_fpr, kn_tpr, marker = '.', label = "K Nearest Neghbor (AUROC = % 0.3f)" %kn_auc)
plt.xlabel("False Positive Rate")
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

