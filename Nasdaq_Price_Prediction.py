#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt


# In[2]:


import yfinance as yf


# In[3]:


nasdaq=yf.Ticker("^IXIC")


# In[4]:


nas=nasdaq.history(period="max")


# In[5]:


nas


# In[6]:


nas.index


# In[7]:


nas.plot.line(y='Close', use_index=True)


# In[8]:


nas.drop(['Dividends','Stock Splits'], axis=1)


# In[9]:


nas['Tomorrow']=nas['Close'].shift(-1)


# In[10]:


nas


# In[11]:


nas['Target']=(nas['Tomorrow']>nas['Close']).astype(int)


# In[12]:


nas


# In[13]:


nas=nas.loc['1990-01-01':].copy()


# In[14]:


nas


# In[15]:


from sklearn.ensemble import RandomForestClassifier


# In[16]:


model=RandomForestClassifier(n_estimators=100, min_samples_split=200,random_state=1)


# In[17]:


train=nas.iloc[:-100]
test=nas.iloc[-100:]


# In[18]:


predictors=['Open','High','Low','Close','Volume']


# In[19]:


model.fit(train[predictors],train['Target'])


# In[20]:


from sklearn.metrics import precision_score


# In[21]:


preds=model.predict(test[predictors])


# In[22]:


preds


# In[23]:


preds=pd.Series(preds,index=test.index)


# In[24]:


preds


# In[25]:


precision_score(test['Target'],preds)


# In[26]:


combined=pd.concat([test['Target'],preds],axis=1)


# In[27]:


combined.plot()


# In[28]:


def predict(train, test, predictors,model):
    model.fit(train[predictors],train['Target'])
    preds=model.predict(test[predictors])
    preds=pd.Series(preds,index=test.index, name='Predictions')
    combined=pd.concat([test['Target'],preds],axis=1)
    return combined  


# In[33]:


def backtest(data, model,predictors, start=2500, step=250):
    all_predictions=[]
    
    for i in range (start, data.shape[0], step):
        train=data.iloc[0:i].copy()
        test=data.iloc[i:(i+step)].copy()
        predictions=predict(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# In[34]:


predictions=backtest(nas,model,predictors)


# In[36]:


predictions["Predictions"].value_counts()


# In[38]:


precision_score(predictions['Target'], predictions['Predictions'])


# In[39]:


predictions['Target'].value_counts()/predictions.shape[0]


# In[49]:


horizons=[2,5,60,250,1000]
new_predictors=[]

for horizon in horizons:
    rolling_averages=nas.rolling(horizon).mean()
    
    ratio_column=f"Close_Ratio_{horizon}"
    nas[ratio_column]=nas["Close"]/rolling_averages["Close"]
    
    trend_column=f"Trend_{horizon}"
    nas[trend_column]=nas.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors +=[ratio_column, trend_column]


# In[42]:


nas.head()


# In[43]:


nas=nas.dropna()


# In[44]:


nas


# In[45]:


model=RandomForestClassifier(n_estimators=200, min_samples_split=50,random_state=1)


# In[51]:


def predict(train, test, predictors,model):
    model.fit(train[predictors],train['Target'])
    preds=model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6]=1
    preds[preds <.6]=0
    preds=pd.Series(preds,index=test.index, name='Predictions')
    combined=pd.concat([test['Target'],preds],axis=1)
    return combined 


# In[52]:


predictions=backtest(nas,model, new_predictors)


# In[53]:


predictions["Predictions"].value_counts()


# In[54]:


precision_score(predictions['Target'], predictions['Predictions'])


# In[ ]:





# In[ ]:




