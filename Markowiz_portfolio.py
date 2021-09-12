#!/usr/bin/env python
# coding: utf-8

# **Taking One month data from 25th of March to 30th April. To evaluate markowitz portfolio**

# In[16]:


#import python libraries
import numpy as np # Numerical Calculations
import pandas as pd # Data Processing
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns # Data Visualisation
import yfinance as yf
get_ipython().run_line_magic('matplotlib', 'inline')

import array
import datetime as dt
import math


# In[26]:


stocks = ['ICICIBANK.NS', 'TATASTEEL.NS', 'SPARC.NS', 'TCS.NS', 'EVERESTIND.NS', 'COFORGE.NS', 'APOLLOTYRE.NS', 'SJVN.NS', 'VAKRANGEE.NS', 'AUROPHARMA.NS', 'JUBLFOOD.NS', 'HDFCBANK.NS', 'RELINFRA.NS', 
         'SBIN.NS', 'WOCKPHARMA.NS', 'INDUSINDBK.NS', 'SUNTV.NS', 'M&MFIN.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'SRTRANSFIN.NS', 'MAGMA.NS', 'NTPC.NS']


# In[13]:


start = dt.datetime.today() - dt.timedelta(360)
end = dt.datetime.today()


# In[27]:


print(start, end)


# In[28]:


Company = pd.DataFrame()


# In[29]:


for ticker in stocks:
    Company[ticker] = yf.download(ticker, start, end)["Adj Close"]


# In[30]:


Company.head()


# In[2]:


#Last One year daily data
#........................Please Read Comment.................
#No need of CSV file already done with yfinace.............
#DONT RUN THIS CELL
#ICICIBANK   = pd.read_csv(r'ICICIBANK.NS.csv')
#TATASTEEL   = pd.read_csv(r'TATASTEEL.NS.csv')
#SPARC       = pd.read_csv(r'SPARC.NS.csv')
#TCS         = pd.read_csv(r'TCS.NS.csv')
#EVERESTIND  = pd.read_csv(r'EVERESTIND.NS.csv')
#NIITTECH    = pd.read_csv(r'NIITTECH.NS.csv')
#APOLLOTYRE  = pd.read_csv(r'APOLLOTYRE.NS.csv')
#SJVN        = pd.read_csv(r'SJVN.NS.csv')
#VAKRANGEE   = pd.read_csv(r'VAKRANGEE.NS.csv')
#AUROPHARMA  = pd.read_csv(r'AUROPHARMA.NS.csv')
#JUBLFOOD    = pd.read_csv(r'JUBLFOOD.NS.csv')
#HDFCBANK    = pd.read_csv(r'HDFCBANK.NS.csv')
#RELINFRA    = pd.read_csv(r'RELINFRA.NS.csv')
#SBIN        = pd.read_csv(r'SBIN.NS.csv')
#WOCKPHARMA  = pd.read_csv(r'WOCKPHARMA.NS.csv')
#INDUSINDBK  = pd.read_csv(r'INDUSINDBK.NS.csv')
#SUNTV       = pd.read_csv(r'SUNTV.NS.csv')
#MMFIN       = pd.read_csv(r'M&MFIN.NS.csv')
#BAJFINANCE  = pd.read_csv(r'BAJFINANCE.NS.csv')
#ASIANPAINT  = pd.read_csv(r'ASIANPAINT.NS.csv')
#SRTRANSFIN  = pd.read_csv(r'SRTRANSFIN.NS.csv')
#MAGMA       = pd.read_csv(r'MAGMA.NS.csv')
#NTPC        = pd.read_csv(r'NTPC.NS.csv')

#NIITTECH.head()


# In[20]:



## NO NEED TO RUN THIS CELL
#Company = pd.DataFrame()
#Company['ICICIBANK']     = ICICIBANK['Close']
#Company['TATASTEEL']     = TATASTEEL['Close']
#Company['SPARC']         = SPARC['Close']
##Company['TCS']           = TCS['Close']
#Company['EVERESTIND']    = EVERESTIND['Close']
#Company['NIITTECH']      = NIITTECH['Close']
#Company['APOLLOTYRE ']   = APOLLOTYRE ['Close']
#Company['SJVN']          = SJVN ['Close']
#Company['VAKRANGEE']     = VAKRANGEE['Close']
#Company['AUROPHARMA']    = AUROPHARMA['Close']
#Company['JUBLFOOD ']     = JUBLFOOD ['Close']
#Company['HDFCBANK  ']    = HDFCBANK['Close']
#Company['RELINFRA ']     = RELINFRA ['Close']
#Company['SBIN ']         = SBIN ['Close']
#Company['WOCKPHARMA']    = WOCKPHARMA ['Close']
#Company['INDUSINDBK']    = INDUSINDBK ['Close']
#Company['SUNTV']         = SUNTV ['Close']
#Company['MMFIN']         = MMFIN['Close']
#Company['BAJFINANCE']    = BAJFINANCE ['Close']
#Company['ASIANPAINT']    = ASIANPAINT ['Close']
#Company['SRTRANSFIN ']   = SRTRANSFIN  ['Close']
#Company['MAGMA']         = MAGMA['Close']
#Company['NTPC']          = NTPC ['Close']

Company.head()


# In[31]:


Company.describe()


# In[32]:


Company.info()


# In[33]:


#shows company daily return

Comp_return = Company.pct_change()
Comp_return


# In[34]:


Comp_return.info()


# In[35]:


describe = Comp_return.describe()
describe


# In[36]:


avg_daily_ret = describe.loc['mean',:]
avg_daily_ret


# In[37]:


Annual_Covar_Mat = Comp_return.loc[:,:].cov()*21
Annual_Covar_Mat


# In[38]:


#Initializing equal weight to each company

Weight = np.array([1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23,1/23])


# In[39]:


Weight.T


# In[40]:


#Calculating Annual Portfoilio variance

port_var=np.dot(Weight.T,np.dot(Annual_Covar_Mat,Weight))
port_var


# In[41]:


# Calculating Portfolio Standard deviation (risk)

port_standard_deviation=np.sqrt(port_var)
port_standard_deviation


# In[42]:


portfolio_return = np.sum(avg_daily_ret*Weight)*21
portfolio_return


# In[43]:


#Calculate Expected return , Standard deviation (risk) and Variance

percent_var=str(round(port_var,2)*100)+'%'
percent_std=str(round(port_standard_deviation,2)*100)+'%'
percent_ret=str(round(portfolio_return ,2)*100)+'%'
print("Expected return: "+percent_ret)
print("Standard Deviation : "+percent_std)
print("Variance: "+percent_var)


# In[44]:
#TO INSTALL 

pip install PyPortfolioOpt


# In[45]:


from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


# In[46]:


# Portfolio optimization !!

# Calculate the expected returns and the annualized sample covariance matix of asset returns

m_r = expected_returns.mean_historical_return(Company)
S   = risk_models.sample_cov(Company)


# Maximize the sharpe ratio 


ef= EfficientFrontier(m_r,S)
weight= ef.max_sharpe()
cleaned_weights=ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)


# In[49]:


clean = pd.DataFrame.from_dict(cleaned_weights, orient='index')


# In[50]:


print(clean)


# In[65]:


fig, (axs1, axs2) = plt.subplots(2,3)
x = Company.index
axs1[0].plot(x,Company["TATASTEEL.NS"])
axs1[0].set_title('TATASTEEL')
axs2[0].plot(x ,Company["COFORGE.NS"] )
axs2[0].set_title("COFORGE")
axs1[1].plot(x, Company["JUBLFOOD.NS"])
axs1[1].set_title("JUBLFOOD")
axs2[1].plot(x, Company["RELINFRA.NS"])
axs2[1].set_title("RELINFRA")
axs1[2].plot(x, Company["ASIANPAINT.NS"])
axs1[2].set_title("ASIANPAINT")
axs2[2].plot(x, Company["MAGMA.NS"])
axs2[2].set_title("MAGMA")

