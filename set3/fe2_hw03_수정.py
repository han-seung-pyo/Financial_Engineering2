# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:00:48 2019

@author: 한승표
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import time ## 시간 측정하는 library. time.time(측정할 것)
from tqdm import tqdm
#%%
#BLACK SHOLES PRICE
def d1(s,k,r,q,T,sigma):
    return (np.log(s/k) + (r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))

def d2(s,k,r,q,T,sigma):
    return (np.log(s/k) + (r-q-0.5*sigma**2)*T)/(sigma*np.sqrt(T))

def bs_price(s,k,r,q,T,sigma,option_type):
    if option_type == 'call':
        x = 1;
    if option_type == 'put':
        x = -1;
    d_1 = d1(s,k,r,q,T,sigma)
    d_2 = d2(s,k,r,q,T,sigma)
    #d_1 = (np.log(s/k) + (r-q+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    #d_2 = (np.log(s/k) + (r-q-0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    option_price = x * s * np.exp(-q*T) * norm.cdf(x*d_1) -x*k*np.exp(-r*T) *norm.cdf(x*d_2);
    return option_price.round(3);

#%%
def option_tree(s,k,r,T,sigma,N,option_type1,option_type2,tree_type,ann_div=None):
    dt = T/N
    v = np.zeros(N+1)
    vv =np.zeros(N+1)
    x =np.zeros(N+1)
    def Leisen_Reimer(s,k,sigma,r,ann_div,T,N):
        dt = T/N
        d1 = (np.log(s/k)+(r-ann_div+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = (np.log(s/k)+(r-ann_div-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        
        def h_func(d):
            
            if d < 0:
               
               h = 0.5 - np.sqrt(0.25-0.25*np.exp(-np.power((d/(N+(1/3))),2)*(N+(1/6))))
               
            else:
                
               h = 0.5 + np.sqrt(0.25-0.25*np.exp(-np.power((d/(N+(1/3))),2)*(N+(1/6)))) 
               
            return h
        
        q_ = h_func(d1)
        q = h_func(d2)
        u = q_*np.exp((r-ann_div)*dt)/q
        d = (np.exp((r-ann_div)*dt)-q*u)/(1-q)
        
        return q,u,d
    
    if tree_type == 'CRR':
        u  = np.exp(sigma*np.sqrt(dt))
        d = 1/u
    elif tree_type == 'Binomial':
        u  = np.exp(r*T/N+sigma*np.sqrt(dt))
        d = np.exp(r*T/N-sigma*np.sqrt(dt))
        
    elif tree_type == 'Rendleman':
        u = np.exp((r-ann_div-0.5*sigma**2)*dt+sigma*np.sqrt(dt))
        d = np.exp((r-ann_div-0.5*sigma**2)*dt-sigma*np.sqrt(dt))
        
    elif tree_type == 'LR':
        u = Leisen_Reimer(s,k,sigma,r,ann_div,T,N)[1]
        d = Leisen_Reimer(s,k,sigma,r,ann_div,T,N)[2]
        
        
    p = (np.exp(r*dt)-d)/(u-d)
    
    if option_type1=='call':    
        sign = 1
    else:
        sign = -1
    
    if option_type2 == 'European':
        for i in range(N+1):
            s1 = s* (u**i)*(d**(N-i))
            v[i] = max(sign*(s1-k),0)
        
        for j in range(1,N+1):
            for i in range(N+1-j):
                vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
            for i in range(N+1):
                v[i] = vv[i]
                
            
    elif option_type2=='American':
        for i in range(N+1):
            s1 = s* (u**i)*(d**(N-i))
            v[i] = max(sign*(s1-k),0)
        
        for j in range(1,N+1):
            for i in range(N+1-j):
                x[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
                s1 = s * (u**i) * (d**(N-j-i))
                vv[i] = max(x[i],sign*(s1-k))
            for i in range(N+1):
                v[i] = vv[i]
            
    return np.round(v[0],5)
#%%
#BD-Method
def BD_method(s,k,r,sigma,div,T,N):
    dt = T/N
    u  = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp((r-div)*dt)-d)/(u-d)
    v=np.zeros(N+1)
    vv = np.zeros(N+1)
    
    for i in range(N):
        s1 = s*(u**i)*(d**(N-1-i))
        d1_ = (np.log(s1/k)+((r-div+0.5*sigma**2)*dt))/(sigma * np.sqrt(dt))
        d2_ = (np.log(s1/k)+((r-div-0.5*sigma**2)*dt))/(sigma * np.sqrt(dt))
        v[i] = max(-s1 * np.exp(-div*dt)*norm.cdf(-d1_) + k*np.exp(-r*dt)*norm.cdf(-d2_),k-s1)
    
    for j in range(2,N+1):
        for i in range(N+1-j):
            vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
            s1 = s * (u**i) * (d**(N-j-i))
            vv[i] = max(vv[i],(k-s1))
        for i in range(N+1):
            v[i] = vv[i]
            
    return np.round(v[0],6)
#%%
#1-a
s= 100;
k = 105;
r = 0.1;
q = 0;
sigma = 0.3;
T = 0.2
N= 51
option_type = 'put'
option_type2 ='European'
a_1_European_put = bs_price(s,k,r,q,T,sigma,option_type)
#%%
#1-b
crr_result = []
RB_result = []
LR_result = []

for i in range(50,1001):
    print ("%s step start!" %(i))
    crr = option_tree(s,k,r,T,sigma,i,option_type,option_type2,'CRR',q)
    rb = option_tree(s,k,r,T,sigma,i,option_type,option_type2,'Rendleman',q)
    crr_result.append(crr)
    RB_result.append(rb)

for i in np.arange(51,1000,2):
    print ("%s step start!" %(i))
    lr = option_tree(s,k,r,T,sigma,i,option_type,option_type2,'LR',q)
    LR_result.append(lr)
    
#%%
#1-c
Error_CRR = crr_result - a_1_European_put
Error_RB = RB_result - a_1_European_put
Error_LR = LR_result - a_1_European_put

x_axis = np.arange(50,1001)
plt.figure(1)
plt.scatter(x_axis,Error_CRR)
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('Error of CRR - BS model')
plt.savefig('Error of CRR - BS model')
plt.show()

plt.figure(2)
plt.scatter(x_axis,Error_RB)
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('Error of RB - BS model')
plt.savefig('Error of RB - BS model')
plt.show()

x_axis_ = np.arange(51,1000,2)
plt.figure(3)
plt.scatter(x_axis_,Error_LR)
plt.ylim(0,0.0003)
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('Error of LR - BS model')


plt.savefig('Error of LR - BS model')

#%%
#2-a
s= 100;
k = 105;
r = 0.1;
q = 0;
sigma = 0.3;
T = 0.2
N= 51
option_type = 'put'
option_type2_ ='American'
exact_result = []

for i in range(10001,10010,2):
    print ("%s step start!" %(i))
    lr = option_tree(s,k,r,T,sigma,i,option_type,option_type2_,'LR',q)
    exact_result.append(lr)

exact_price = np.mean(exact_result)

#%%
#2-b
crr_result_american = []
bd_result_american = []
LR_result_amreican = []
for i in tqdm(range(50,1000)):
    print ("%s step start!" %(i))
    crr_american = option_tree(s,k,r,T,sigma,i,option_type,option_type2_,'CRR',q)
    bd = BD_method(s,k,r,sigma,q,T,i)
    crr_result_american.append(crr_american)
    bd_result_american.append(bd)
#%%
for i in tqdm(range(51,1000,2)):
    print ("%s step start!" %(i))
    lr = option_tree(s,k,r,T,sigma,i,option_type,option_type2_,'LR',q)
    LR_result_amreican.append(lr)
    
#BD = pd.DataFrame(bd_result_american)
#crr = pd.DataFrame(crr_result_american)
#%%
#2-b
Error_CRR_American = crr_result_american - exact_price
Error_BD_American = bd_result_american - exact_price
Error_LR_American = LR_result_amreican - exact_price

x_axis = np.arange(50,1000)
plt.figure(4)
plt.scatter(x_axis,Error_CRR_American)
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('Error of CRR_A - BS model')
plt.show()

plt.savefig('Error of CRR_A - BS model')
plt.figure(5)
plt.scatter(x_axis,Error_BD_American)
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('Error of BD_A - BS model')
plt.show()
plt.savefig('Error of BD_A - BS model')
x_axis_ = np.arange(51,1000,2)
plt.figure(6)
plt.scatter(x_axis_,Error_LR_American)
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('Error of LR_A - BS model')
plt.savefig('Error of LR_A - BS model')
#%%
#2-c
#Excersise boundary
s= 100;
k = 105;
r = 0.1;
div = 0;
sigma = 0.3;
T = 0.2
N= 100
option_type = 'put'
option_type2_ ='American'

dt = T/N
u  = np.exp(sigma*np.sqrt(dt))
d = 1/u
p = (np.exp((r-div)*dt)-d)/(u-d)
v=np.zeros(N+1)
vv = np.zeros(N+1)
exercise_s = pd.DataFrame(np.zeros((N+1,N+1)))
for i in range(N):
    s1 = s*(u**i)*(d**(N-1-i))
    d1_ = (np.log(s1/k)+((r-div+0.5*sigma**2)*dt))/(sigma * np.sqrt(dt))
    d2_ = (np.log(s1/k)+((r-div-0.5*sigma**2)*dt))/(sigma * np.sqrt(dt))
    v[i] = max(-s1 * np.exp(-div*dt)*norm.cdf(-d1_) + k*np.exp(-r*dt)*norm.cdf(-d2_),k-s1)
    
for j in range(1,N+1):
    for i in range(N+1-j):
        vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
        s1 = s * (u**i) * (d**(N-j-i))
        if k-s1 > vv[i]:
            exercise_s.loc[i,j] = s1
        vv[i] = max(vv[i],(k-s1))
    for i in range(N+1):
        v[i] = vv[i]

exercise_s = exercise_s.replace(0,np.nan)
boundary =exercise_s.max(axis=0)
boundary_new = boundary.sort_index(ascending=False)
boundary_new = pd.DataFrame(boundary_new.values)
time_index = np.arange(0,1.01,0.01)
plt.figure(7)
plt.plot(time_index,boundary_new,label='Boundary')
plt.xlabel('Time Step')
plt.ylabel('Stock price')
plt.legend(loc='best')
plt.title('Put option early excersize boundary')
plt.savefig('Put option early excersize boundary')
