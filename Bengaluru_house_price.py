#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv(r"C:\Users\Vignesh Chowdary\OneDrive\Documents\Downloads\Bengaluru_House_Data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:



df.columns


# In[6]:


df.size


# In[7]:


df2=df.drop(["area_type","availability","society","balcony"],axis="columns")


# In[8]:


df2.head()


# In[9]:


df2.isnull().sum()


# In[10]:


df3=df2.dropna()


# In[11]:


df3.isnull().sum()


# In[12]:


df3.shape


# In[13]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[14]:


df3.head()


# In[15]:


df3["total_sqft"].unique()


# In[16]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[17]:


df3[~df3['total_sqft'].apply(is_float)].head(7)


# In[18]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[19]:


df4 = df3.copy()
df4.total_sqft=df4.total_sqft.apply(convert_sqft_to_num)
df4.head()


# In[20]:


df4["price_per_sqft"]=df4["price"]*100000/df4["total_sqft"]
df4.head()


# In[21]:


locations=df4["location"].value_counts()
locations


# In[22]:


len(locations)


# In[23]:


len(locations[locations<=10])


# In[24]:


len(locations[locations>10])


# In[25]:


locations_lessthan_10=locations[locations<=10]


# In[26]:


locations_lessthan_10


# In[27]:


df4.location.nunique()


# In[28]:


df4.location=df4.location.apply(lambda x: "other" if x in locations_lessthan_10 else x)
df4.head(15)


# In[29]:


df4[df4["total_sqft"]/df4["bhk"]<300].head(15)


# In[30]:


df4.shape


# In[31]:


df5= df4[~(df4.total_sqft/df4.bhk<300)]
df5.shape


# In[32]:


df5.shape


# In[33]:


df5["price_per_sqft"].describe()


# In[34]:


def remove_pricesqft_outlier(df):
    vignesh=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        vignesh=pd.concat([vignesh,reduced_df],ignore_index=True)
    return vignesh
df6=remove_pricesqft_outlier(df5)
df6.head(20)


# In[35]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (13,8)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Price (Lakh Indian Rupees)')
    plt.title(location)
    plt.legend()
plot_scatter_chart(df6,"Rajaji Nagar")


# In[36]:


df6.location.unique()


# In[37]:


def remove_bhk_outliers(df):
    vignesh=np.array([])
    for location,location_df in df.groupby("location"):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk]={
                           "mean":np.mean(bhk_df.price_per_sqft),
                           "std":np.std(bhk_df.price_per_sqft),
                           "count":bhk_df.shape[0]
           }
        for bhk,bhk_df in location_df.groupby("bhk"):
            stats=bhk_stats.get(bhk-1)
            if stats and stats["count"]>5:
                vignesh=np.append(vignesh,bhk_df[bhk_df.price_per_sqft<(stats["mean"])].index.values)
    return df.drop(vignesh,axis="index")
df7=remove_bhk_outliers(df6)
df7.shape


# In[38]:


plot_scatter_chart(df7,"Rajaji Nagar")


# In[39]:


plt.hist(df7.price_per_sqft,rwidth=0.5)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[40]:


plt.hist(df7.bath,rwidth=0.5)
plt.xlabel("number of bathrooms")
plt.ylabel("count")


# In[41]:


df7[df7.bath>10]


# In[42]:


df8 = df7[df7.bath<df7.bhk+2]
df8.shape


# In[43]:


df8.head(20)


# In[44]:


df9 = df8.drop(['size','price_per_sqft'],axis='columns')
df9.head(3)


# In[45]:


dummies = pd.get_dummies(df9.location)
dummies.head(3)


# In[46]:


df10 = pd.concat([df9,dummies.drop('other',axis='columns')],axis='columns')
df10.head()


# In[47]:


df11 = df10.drop('location',axis='columns')
df11.head(2)


# In[48]:


ab= df11.drop(['price'],axis='columns')
ab.head(3)


# In[49]:


X=ab.copy()


# In[50]:


y = df11.price


# In[51]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=7)


# In[52]:


from sklearn.linear_model import LinearRegression
X.dropna(inplace=True)
y.dropna(inplace=True)
classifier = LinearRegression()
classifier.fit(X_train,y_train)
classifier.score(X_test,y_test)


# In[53]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return classifier.predict([x])[0]


# In[54]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[55]:


predict_price('Indira Nagar',1000, 2, 2)


# In[56]:


predict_price('Indira Nagar',1000, 3, 3)


# In[ ]:




