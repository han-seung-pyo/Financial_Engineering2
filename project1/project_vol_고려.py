# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 07:35:45 2019

@author: 한승표
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import datetime as dt_
from tqdm import tqdm
import scipy as sp
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay


class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]
#%%
def business_dates(start, end):
    us_cal = USTradingCalendar()
    kw = dict(start=start, end=end)
    return pd.DatetimeIndex(freq='B', **kw).drop(us_cal.holidays(**kw))

bd_list_2019_ = []
bd_other = []
bd_list_2019_real = []
bd_other_real = []
for j in range(6):
    if j==0:
        for i in range(11):
            if i+2 == 2:
               bd = len(business_dates(start=date(2019+j,1+i,28), end=date(2019+j,2+i,26)))
               bd_ = business_dates(start=date(2019+j,1+i,28), end=date(2019+j,2+i,26))
            else:
                 if  (i+2)%2 == 1:
                     bd = len(business_dates(start=date(2019+j,1+i,27), end=date(2019+j,2+i,26)))
                     bd_ = business_dates(start=date(2019+j,1+i,27), end=date(2019+j,2+i,26))
                 else:
                     bd = len(business_dates(start=date(2019+j,1+i,27), end=date(2019+j,2+i,26))) 
                     bd_ = business_dates(start=date(2019+j,1+i,27), end=date(2019+j,2+i,26))
            
            bd_list_2019_.append(bd)
            bd_list_2019_real.append(bd_)
    else:
        for i in range(12):
            if i+1 == 1 :
               bdc_ = len(business_dates(start=date(2019+j-1,12,27), end=date(2019+j,1+i,26)))
               bdc = business_dates(start=date(2019+j-1,12,27), end=date(2019+j,1+i,26))
            else:
                 if  (i+1)%2 == 1:
                     bdc_ = len(business_dates(start=date(2019+j,i,27), end=date(2019+j,1+i,26)))
                     bdc = business_dates(start=date(2019+j,i,27), end=date(2019+j,1+i,26))
                 else:
                     bdc_ = len(business_dates(start=date(2019+j,i,27), end=date(2019+j,1+i,26))) 
                     bdc = business_dates(start=date(2019+j,i,27), end=date(2019+j,1+i,26))
                  
            bd_other.append(bdc_)
            bd_other_real.append(bdc)
#%%
bd_node = bd_list_2019_ + bd_other
bd_node = bd_node[:59]
cpn_node = list(np.cumsum(bd_node))
cpn_node_ = cpn_node
bd_node_real = bd_list_2019_real + bd_other_real


#%%
def interpolation(start,date1,rate1,date2,rate2,target_date):
    
    x1 = (date1 - start).days
    x2 = (date2 - start).days
    x_target = (target_date - start).days
    target = (x_target - x1)*(rate2 - rate1)/(x2 - x1) + rate1
    
    return target


st = dt_.date(2019,1,28)
d1 = dt_.date(2023,10,3)
d2 = dt_.date(2024,10,3)
td = dt_.date(2024,1,26)

r_est = interpolation(st,d1,0.03056,d2,0.03065,td)
#parameter
[s,r,sigma,T,N,B,cpn,FV,q] = [2643.85,r_est,0.224,5,1259,0.8*2643.85,0.0615/12,1000,0.0199]
cpn_node = [0]+list(np.cumsum(bd_node))
#%%
df = pd.read_csv('C:/Users/한승표/Desktop/미국관련/미국수업/FIN514(Financial Engineering2)/project/project1/vol.csv', 
                 index_col=['Maturity'], parse_dates=['Maturity'],engine='python')

Moneyness80 = df['80']
vol = pd.DataFrame(Moneyness80)

#%%
real_date = ([(dt_.date(2019,1,28) + dt_.timedelta(days=x)) for x in range(0, 1825)])
implied_vol = pd.DataFrame(np.zeros((len(real_date),1)),index=real_date)
implied_vol.index = pd.to_datetime(implied_vol.index,format ='%Y-%m-%d')

#%%
for date_ in vol.index :
    
    if date_ in implied_vol.index:
       implied_vol.loc[date_][0] = vol.loc[date_]['80']
       
implied_vol.loc['2024-01-26']=0.224

#%%
for i in range(1,len(vol)):
       
    for date_ in implied_vol.index:
            
       target = date_
       
       if target > vol.index[i-1] and target < vol.index[i]:
           
          start = dt_.datetime(2019,1,28)
          date1 = vol.index[i-1]
          date2 = vol.index[i]
          rate1 = vol.iloc[i-1]['80']
          rate2 = vol.iloc[i]['80']
          implied_vol.loc[target][0] = interpolation(start,date1,rate1,date2,rate2,target) 
#%%
bd_real_date = pd.DataFrame()
for i in range(60):
    bd_real_date = pd.concat([bd_real_date,pd.Series(bd_node_real[i])],axis=0,ignore_index=True)

bd_real_date = bd_real_date.set_index(bd_real_date[0])
bd_implied_vol = implied_vol.loc[implied_vol.index.isin(bd_real_date.index)]  
bd_implied_vol.index = np.arange(0,len(bd_implied_vol))  
bd_implied_vol_ = list(bd_implied_vol[0]) 
#%%
def showTree(tree):
    t = np.linspace(T/N, T, N+1)
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    for i in range(len(t)):
        for j in range(i+1):
            ax.plot(t[i], tree[i][j], '.b')
            if i<len(t)-1:
                ax.plot([t[i],t[i+1]], [tree[i][j], tree[i+1][j]], '-b')
                ax.plot([t[i],t[i+1]], [tree[i][j], tree[i+1][j+1]], '-b')
    fig.show()
#%%
def CRR_tree(s,sigma_list,q,T,N):
    dt = T/N
    u  = np.exp(sigma_list*np.sqrt(dt))
    d = 1/u
    u.loc[0][0]=1
    d.loc[0][0]=1
    u = list(u[0])
    d = list(d[0])
    stock = np.zeros([N + 1, N + 1])
    for i in tqdm(range(N + 1)):
        for j in tqdm(range(i + 1)):
            stock[j, i] = s * (d[i-j]**(i-j))*(u[j]**j)
    return stock
#%%
sigma_list = bd_implied_vol
stock_path = CRR_tree(s,bd_implied_vol,q,T,N)  
#showTree(stock_path)
value_path = np.zeros([N + 1, N + 1])
u  = np.exp(sigma_list*np.sqrt(T/N))
d = 1/u
dt = T/N
p = (np.exp((r-q)*dt)-d)/ (u-d)
p = list(p[0])
#%%
#만기
for i in np.arange(N+1):
    if stock_path[i,-1] >=B:
        value_path[i,-1] = FV
    else:
        value_path[i,-1] = round(FV * (1-(B-stock_path[i,-1])/s),6)
#%%
#조기상환 있는 노드들
for i in np.arange(N-1,1239,-1):
    for j in range(i+1):
        value_path[j,i] =np.exp(-r*dt) * (p[j]* value_path[j+1,i+1]+(1-p[j])*value_path[j,i+1])
    

for i in np.arange(1239,252,-1):
    for j in range(i+1):
        if stock_path[j,i] >= B:
            if i in cpn_node:
                value_path[j,i] = min(FV*np.exp(-r*(cpn_node[-1]-i)*dt),np.exp(-r*dt) * (p[j]* value_path[j+1,i+1]+(1-p[j])*value_path[j,i+1]))+\
                (FV*cpn/(cpn_node[-1]-cpn_node[-2]))*np.exp(-r*(cpn_node[-1]-i)*dt)
            else:
                value_path[j,i] =np.exp(-r*dt) * (p[j]* value_path[j+1,i+1]+(1-p[j])*value_path[j,i+1]) + (FV*cpn/(cpn_node[-1]-cpn_node[-2]))*np.exp(-r*(cpn_node[-1]-i)*dt)
        else:
            if i in cpn_node:
                value_path[j,i] = min(FV*np.exp(-r*(cpn_node[-1]-i)*dt),np.exp(-r*dt) * (p[j]* value_path[j+1,i+1]+(1-p[j])*value_path[j,i+1]))
            else:
                value_path[j,i] =np.exp(-r*dt) * (p[j]* value_path[j+1,i+1]+(1-p[j])*value_path[j,i+1])
               
    if i<cpn_node[-2]:
        cpn_node = cpn_node[:-1]
#%%
#조기상환이 아예 없는 노드들
for i in np.arange(252,-1,-1):
    for j in range(i+1):
        if stock_path[j,i] >= B:
            value_path[j,i] =np.exp(-r*dt) * (p[j]* value_path[j+1,i+1]+(1-p[j])*value_path[j,i+1]) + (FV*cpn/(cpn_node[-1]-cpn_node[-2]))*np.exp(-r*(cpn_node[-1]-i)*dt)
        else:
            value_path[j,i] =np.exp(-r*dt) * (p[j]* value_path[j+1,i+1]+(1-p[j])*value_path[j,i+1])
    
    
    while i<cpn_node[-2]:
        if  np.size(cpn_node)==2:
            break
        else:
            cpn_node = cpn_node[:-1]

print("Binomial tree price %0.3f" %(value_path[0,0]))
#%%
def RAN_sim(s,r,sigma,B,T,cpn,cpn_node_,FV,q,n_trials,n_steps):
    z_matrix = np.random.standard_normal(size =(n_trials,n_steps))
    st_matrix = np.zeros((n_trials,n_steps))
    value_matrix = np.zeros((n_trials,n_steps))
    st_matrix[:,0] = s
    dt = T/n_steps
    for i in range(n_steps-1):
        st_matrix[:,i+1] = st_matrix[:,i]*np.exp((r-q-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z_matrix[:,i])
    
    for i in range(n_trials):
        if st_matrix[i,-1] >B:
            value_matrix[i,-1] = FV
        else:
            value_matrix[i,-1] = FV*(1-(B-st_matrix[i,-1])/s)
    
    #쿠폰 고려 없을 떄
    for i in np.arange(n_steps-2,1239,-1):
        for j in range(n_trials):
            value_matrix[j,i] = np.exp(-r*dt)*value_matrix[j,i+1]

    #조기 상환 고려할 때
    for i in np.arange(1239,251,-1):
        for j in range(n_trials):
            if st_matrix[j,i] >B:
                if i in cpn_node_:
                    value_matrix[j,i] = min(FV*np.exp(-r*(cpn_node_[-1]-i)*dt),np.exp(-r*dt)*value_matrix[j,i+1]) + (FV*cpn/(cpn_node_[-1]-cpn_node_[-2])) * np.exp(-r*(cpn_node_[-1]-i)*dt)
                else:
                    value_matrix[j,i] = np.exp(-r*dt)*value_matrix[j,i+1] + (FV*cpn/(cpn_node_[-1]-cpn_node_[-2])) * np.exp(-r*(cpn_node_[-1]-i)*dt)
            else:
                if i in cpn_node_:
                    value_matrix[j,i] = min(FV*np.exp(-r*(cpn_node_[-1]-i)*dt),np.exp(-r*dt)*value_matrix[j,i+1])
                else:
                     value_matrix[j,i] = np.exp(-r*dt)*value_matrix[j,i+1]
        if i<cpn_node_[-2]:
            cpn_node_ = cpn_node_[:-1]
    
    #조기상환 안되는 노드
    for i in np.arange(251,-1,-1):
        for j in range(n_trials):
            if st_matrix[j,i] >B:
                value_matrix[j,i] = np.exp(-r*dt)*value_matrix[j,i+1] + (FV*cpn/(cpn_node_[-1]-cpn_node_[-2])) * np.exp(-r*(cpn_node_[-1]-i)*dt)
            else:
                 value_matrix[j,i] = np.exp(-r*dt)*value_matrix[j,i+1]
    
        while i<cpn_node_[-2]:
            if  np.size(cpn_node_)==2:
                break
            else:
                cpn_node_ = cpn_node_[:-1]
    return value_matrix[:,0].mean()