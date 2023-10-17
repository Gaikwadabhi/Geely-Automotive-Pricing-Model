#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[3]:


car_df = pd.read_csv("/kaggle/input/car-price-prediction/CarPrice_Assignment.csv", engine='


# In[6]:


car_df = pd.read_csv('CarPrice_Assignment (1).csv')
car_df.head()


# In[7]:


car_df.isnull().sum()


# In[8]:


# Dataset dimensions

car_df.shape


# In[9]:


# Dataset information

car_df.info()


# In[10]:


car_df.describe()


# In[11]:


import statsmodels.api as sm


# In[12]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def get_VIF(X_train):
    # A dataframe that will contain the names of all the feature variables and their respective VIFs
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns
    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    print(vif)


# In[13]:


car_df.loc[:,'company'] = car_df.CarName.str.split(' ').str[0]


# In[14]:


car_df.company = car_df.company.apply(lambda x: str(x).lower())


# In[15]:


car_df.company.unique()


# In[16]:


car_df['company'].replace('maxda','mazda',inplace=True)
car_df['company'].replace('porcshce','porsche',inplace=True)
car_df['company'].replace('toyouta','toyota',inplace=True)
car_df['company'].replace(['vokswagen','vw'],'volkswagen',inplace=True)


# In[17]:


car_df.drop(columns = 'CarName', inplace=True)


# In[18]:


car_df.fuelsystem.unique()


# In[19]:


car_df['fuelsystem'].replace('mfi','mpfi',inplace=True)


# In[20]:


car_df.enginetype.unique()


# In[21]:


car_df['enginetype'].replace('dohcv','dohc',inplace = True)
car_df['enginetype'].replace('ohcv','ohc',inplace = True)


# In[22]:


car_df.drivewheel.unique()


# In[23]:


car_df['drivewheel'].replace('4wd', 'fwd', inplace = True)


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[25]:


sns.pairplot(car_df, diag_kind="kde")
plt.show()


# In[26]:


plt.figure(figsize=(20,12))
sns.heatmap(car_df.corr(), linewidths=.5, annot=True, cmap="YlGnBu")


# In[27]:


car_df.loc[:,'curbweight/enginesize'] = car_df.curbweight/car_df.enginesize


# In[28]:


car_df.loc[:,'enginesize/horsepower'] = car_df.enginesize/car_df.horsepower


# In[29]:


car_df.loc[:,'carwidth/carlength'] = car_df.carwidth/car_df.carlength


# In[30]:


car_df.loc[:,'highway/city'] = car_df.highwaympg/car_df.citympg


# In[31]:


car_df.drop(columns = ['enginesize','carwidth', 'carlength', 'highwaympg', 'citympg'],


# In[32]:


car_df.drop(columns = ['enginesize','carwidth', 'carlength', 'highwaympg', 'citympg'], inplace = True)


# In[33]:


car_df.head()


# In[34]:


car_df.drop(columns = 'car_ID', inplace=True)


# In[35]:


car_df.symboling = car_df.symboling.map({-3: 'safe', -2: 'safe',-1: 'safe',0: 'moderate',1: 'moderate',2: 'risky',3:'risky'})


# In[36]:


plt.figure(figsize=(20, 16))
plt.subplot(3,3,1)
sns.boxplot(x = 'symboling', y = 'price', data = car_df)
plt.subplot(3,3,2)
sns.boxplot(x = 'fueltype', y = 'price', data = car_df)
plt.subplot(3,3,3)
sns.boxplot(x = 'aspiration', y = 'price', data = car_df)
plt.subplot(3,3,4)
sns.boxplot(x = 'doornumber', y = 'price', data = car_df)
plt.subplot(3,3,5)
sns.boxplot(x = 'carbody', y = 'price', data = car_df)
plt.subplot(3,3,6)
sns.boxplot(x = 'drivewheel', y = 'price', data = car_df)
plt.subplot(3,3,7)
sns.boxplot(x = 'enginelocation', y = 'price', data = car_df)
plt.subplot(3,3,8)
sns.boxplot(x = 'cylindernumber', y = 'price', data = car_df)
plt.subplot(3,3,9)
sns.boxplot(x = 'fuelsystem', y = 'price', data = car_df)
plt.show()


# In[37]:


plt.figure(figsize=(20, 16))
sns.boxplot(x = 'company', y = 'price', data = car_df, palette="Reds")


# In[38]:


median_dict = car_df.groupby(['company'])[['price']].median().to_dict()
median_dict = median_dict['price']
median_dict


# In[39]:


dict_keys = list(median_dict.keys())

# Median price of category below 10000 is low, between 10000 and 20000 is med and above 20000 is high
for i in dict_keys:
    if median_dict[i] < 10000:
        median_dict[i] = 'low'
    elif median_dict[i] >= 10000 and median_dict[i] <= 20000:
        median_dict[i] = 'med'
    else:
        median_dict[i] = 'high'

median_dict


# In[40]:


car_df.company = car_df.company.map(median_dict)
car_df.company.unique()


# In[41]:


car_df = pd.get_dummies(car_df, drop_first=True)


# In[42]:


car_df.head()


# In[43]:


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(car_df, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[44]:


print("Train data shape: ", df_train.shape)
print("Test data shape: ", df_test.shape)


# In[45]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[46]:


conti_vars = ['wheelbase', 'carheight', 'boreratio', 'stroke', 'compressionratio', 'peakrpm', 'horsepower', 'curbweight', 'price', 'curbweight/enginesize', 'carwidth/carlength', 'highway/city', 'enginesize/horsepower']
df_train[conti_vars] = scaler.fit_transform(df_train[conti_vars])

df_train.describe()


# In[47]:


y_train = df_train.pop('price')
X_train = df_train


# In[48]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[49]:


lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 10)             # running RFE to select 10 best features
rfe = rfe.fit(X_train, y_train)


# In[ ]:




