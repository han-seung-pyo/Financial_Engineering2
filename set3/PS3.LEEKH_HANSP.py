# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:00:48 2019

@author: 한승표
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
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

for i in tqdm(range(50,1001)):
    crr_result.append(option_tree(s,k,r,T,sigma,i,option_type,option_type2,'CRR',q))
    RB_result.append(option_tree(s,k,r,T,sigma,i,option_type,option_type2,'Rendleman',q))

for i in tqdm(np.arange(51,1000,2)):
    LR_result.append(option_tree(s,k,r,T,sigma,i,option_type,option_type2,'LR',q))
#%%
#홀짝 나눠보기
crr_result_ = pd.DataFrame(crr_result)
RB_result_ = pd.DataFrame(RB_result)
LR_result_ = pd.DataFrame(LR_result)

crr_result_even = []
RB_result_even = []
crr_result_odd = []
RB_result_odd= []
for i in crr_result_.index:
    if i%2==0:
        crr_result_even.append(crr_result_.ix[i].values)
        RB_result_even.append(RB_result_.ix[i].values)
    else:
        crr_result_odd.append(crr_result_.ix[i])
        RB_result_odd.append(RB_result_.ix[i])


crr_result_even = pd.DataFrame(crr_result_even)
RB_result_even = pd.DataFrame(RB_result_even)
crr_result_odd = pd.DataFrame(crr_result_odd)
RB_result_odd = pd.DataFrame(RB_result_odd)
#%%
err_crr_even =crr_result_even - a_1_European_put
err_crr_odd =crr_result_odd - a_1_European_put
err_rb_even =RB_result_even - a_1_European_put
err_rb_odd =RB_result_odd - a_1_European_put

x_axis = np.arange(50,1001,2)
x_axis_ =  np.arange(51,1000,2)

plt.figure(1)
plt.scatter(x_axis,err_crr_even)
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('err_crr_even')
plt.show()

plt.figure(2)
plt.scatter(x_axis_,err_crr_odd)
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('err_crr_odd')
plt.show()


plt.figure(3)
plt.scatter(x_axis,err_rb_even)
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('err_rb_even')
plt.show()

plt.figure(4)
plt.scatter(x_axis_,err_rb_odd)
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('err_rb_odd')
plt.show()



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

for i in tqdm(range(10001,10010,2)):
    lr = option_tree(s,k,r,T,sigma,i,option_type,option_type2_,'LR',q)
    exact_result.append(lr)

exact_price = np.mean(exact_result)

#%%
#2-b
crr_result_american = []
bd_result_american = []
LR_result_amreican = []
for i in tqdm(range(50,1000)):
    crr_american = option_tree(s,k,r,T,sigma,i,option_type,option_type2_,'CRR',q)
    bd = BD_method(s,k,r,sigma,q,T,i)
    crr_result_american.append(crr_american)
    bd_result_american.append(bd)

for i in tqdm(range(51,1000,2)):
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
#%%
def DAO_bs_price(s,k,B,r,q,t,sigma,option_type):
    if option_type == 'call':
        x = 1;
    if option_type == 'put':
        x = -1;
    d_1 = (np.log(s/k) + (r-q+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d_2 = (np.log(s/k) + (r-q-0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    h_1 = (np.log((B**2)/(k*s)) + (r-q+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    h_2 = (np.log((B**2)/(k*s)) + (r-q-0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    term1 = x * s * np.exp(-q*T) * norm.cdf(x*d_1) -x*k*np.exp(-r*T) *norm.cdf(x*d_2);
    term2 = ((B/s)**(1+2*r/(sigma**2))*s*norm.cdf(h_1))
    term3 = (B/s)**(-1+2*r/(sigma**2))*k*np.exp(-r*t)*norm.cdf(h_2)
    option_price = term1 - term2 +term3
    return option_price.round(3)


#%%
def DAO_Binomial(s,k,B,r,q,sigma,T,N,option_type,tree_type,barrier_type=None,Barrier_number=None,Barrier_time=None):
    dt = T/N
    v = np.zeros(N+1)
    vv =np.zeros(N+1)
    cond=0
    
    def Leisen_Reimer(s,k,sigma,r,q,T,N):
        dt = T/N
        d1 = (np.log(s/k)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = (np.log(s/k)+(r-q-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        
        def h_func(d):
            
            if d < 0:
               
               h = 0.5 - np.sqrt(0.25-0.25*np.exp(-np.power((d/(N+(1/3))),2)*(N+(1/6))))
               
            else:
                
               h = 0.5 + np.sqrt(0.25-0.25*np.exp(-np.power((d/(N+(1/3))),2)*(N+(1/6)))) 
               
            return h
        
        q_ = h_func(d1)
        q = h_func(d2)
        u = q_*np.exp((r-q)*dt)/q
        d = (np.exp((r-q)*dt)-q*u)/(1-q)
        
        return q,u,d
    
    
    if tree_type == 'CRR':
        u  = np.exp(sigma*np.sqrt(dt))
        d = 1/u
    elif tree_type == 'Binomial':
        u  = np.exp(r*T/N+sigma*np.sqrt(dt))
        d = np.exp(r*T/N-sigma*np.sqrt(dt))
        
    elif tree_type == 'Rendleman':
        u = np.exp((r-q-0.5*sigma**2)*dt+sigma*np.sqrt(dt))
        d = np.exp((r-q-0.5*sigma**2)*dt-sigma*np.sqrt(dt))
        
    elif tree_type == 'LR':
        u = Leisen_Reimer(s,k,sigma,r,q,T,N)[1]
        d = Leisen_Reimer(s,k,sigma,r,q,T,N)[2]
        
        
    p = (np.exp(r*dt)-d)/(u-d)
    

    if option_type=='call':    
        sign = 1
    else:
        sign = -1
        
    for i in range(N+1):
        s1 = s* (u**i)*(d**(N-i))
        v[i] = max(sign*(s1-k),0)
        
        if s1 >B and cond == 0:
            sk = s1
            sk_under = s *u**(i-1)*d**(N-(i-1))
            lambda_cal = (sk-B)/(sk-sk_under)
            cond=1
            
    for j in range(1,N+1):
        for i in range(N+1-j):
            vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
            s1 = s * (u**i) * (d**(N-j-i))
            if s1<B:
                vv[i]=0
                
        for i in range(N+1):
            v[i] = vv[i]
            
    if barrier_type=='Discrete':
        dn= N/Barrier_number
        barrier_node = dn * Barrier_time #정수 일수도, 아닐 수도 있다!
        barrier_node_ad = barrier_node.astype(int)
        barrier_index = -1
        
        for i in range(N+1):
            s1 = s* (u**i)*(d**(N-i))
            v[i] = max(sign*(s1-k),0)
        
            if s1 >B and cond == 0:
                sk = s1
                sk_under = s *u**(i-1)*d**(N-(i-1))
                lambda_cal = (sk-B)/(sk-sk_under)
                cond=1
        
        for j in range(1,N+1):
            for i in range(N+1-j):
                vv[i] = np.exp(-r*dt)*(p*v[i+1]+(1-p)*v[i])
                s1 = s * (u**i) * (d**(N-j-i))
                
                if j in barrier_node_ad:
                    if s1*np.exp(r*(barrier_node[barrier_index]-j)*dt)<B:
                        vv[i] = 0
            
            if j in barrier_node_ad:
                barrier_index -=1
                
            for i in range(N+1):
                v[i] = vv[i]
        
    return v[0], lambda_cal

#%%
#3-a,b
s = 100
k=100
B= 95
r = 0.1
q = 0
sigma = 0.3
T = 0.2
N = 51
exact_d = DAO_bs_price(s,k,B,r,q,T,sigma,'call')
DAO_result = []
for i in tqdm(range(50,1000)):
    DAO_result.append(DAO_Binomial(s,k,B,r,q,sigma,T,i,'call','CRR'))
DAO_result_ = pd.DataFrame(DAO_result,columns=['CRR','lambda'])

#%%
#3-b
Err_DOA_CRR = DAO_result_['CRR'] - exact_d
x_axis = np.arange(50,1000)
plt.figure(8)
plt.plot(x_axis,DAO_result_['CRR'])
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('Err_DOA_CRR - BS model')
plt.savefig('Err_DOA_CRR- BS mode')
plt.show()

plt.figure(9)
plt.scatter(x_axis,DAO_result_['lambda'])
plt.xlabel('Number of Steps')
plt.ylabel('lambda')
plt.title('lambda')
plt.savefig('lambda')
plt.show()
#%%
#4-a
s = 100
k=100
B= 95
r = 0.1
q = 0
sigma = 0.3
T = 0.2
Barrier_time = np.array([1,2,3,4])
eaxct_dc = 5.6711051343
Barrier_number=5

dc_DAO_result=[]
for i in tqdm(np.arange(50,100,10)):
    dc_DAO_result.append(DAO_Binomial(s,k,B,r,q,sigma,T,i,'call','CRR','Discrete',Barrier_number,Barrier_time))
    
dc_DAO_result = pd.DataFrame(dc_DAO_result ,columns=['dc_crr','lambda'])
#%%
#4-a,b
dc_err = dc_DAO_result['dc_crr'] - eaxct_dc
x_axis = np.arange(50,1000,10)
plt.figure(10)
plt.plot(x_axis,dc_err)
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.title('Err_DOA_CRR_DC - BS model')
plt.show()
plt.savefig('Err_DOA_CRR_DC - BS model')

plt.figure(11)
plt.scatter(x_axis,dc_DAO_result['lambda'])
plt.xlabel('Number of Steps')
plt.ylabel('lambda')
plt.title('lambda_discrete')
plt.savefig('lambda_discrete')
plt.show()

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
line1 = ax1.plot(x_axis,dc_err)
line2 = ax2.scatter(x_axis,dc_DAO_result['lambda'])

ax1.set_xlabel(title[0])
ax1.set_ylabel('Error')
ax2.set_ylabel('Lambda')

ax1.set_ylim(-0.2,0.2)
ax2.set_ylim(0,1)

lines = line1+ line2
labels = [l.get_labels() for l in lines]
plt.legend(lines,labels,loc=2)
fig.tight_layout()
plt.show()
