# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 07:54:22 2019

@author: 한승표
"""

#일단 상수로 놓고, payoff 코딩만 해보기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tqdm

[sigma,r,div,s0,Barrier_level,cpn_rate,FV,upper_ratio,S_node,T_node,T]=\
[0.25,0.025,0.007,195.09,0.78,0.020375,1000,2,100,780,1.25]
issue_date = dt.date(2019,3,26)
cpn_date = [dt.date(2019,6,26),dt.date(2019,9,26),dt.date(2019,12,27),dt.date(2020,3,26),dt.date(2020,6,25)]

#%%
def FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,S_node,T_node,T,issue_date,cpn_date,method,print_result=False):
    
    Barrier = s0*Barrier_level
    lower = 0
    upper = s0*upper_ratio
    ds = (upper-lower)/S_node
    d_t = T/T_node
    s_range = np.linspace(lower,upper,S_node+1)
    barrier_node = int(Barrier/ds)
    s0_node = int(S_node/2)
    
    def TDMAsolver(a, b, c, d):
 
        nf = len(d)     # number of edivuations
        ac, bc, cc, dc = map(np.array, (a, b, c, d))     # copy the array
        for it in range(1, nf):
            mc = ac[it-1]/bc[it-1]
            bc[it] = bc[it] - mc*cc[it-1] 
            dc[it] = dc[it] - mc*dc[it-1]
    
        xc = bc
        xc[-1] = dc[-1]/bc[-1]
    
        for il in range(nf-2, -1, -1):
            xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
    
#        del bc, cc, dc  # delete variables from memory
    
        return xc  
    
    # Change coupon payment date to node 
    cnp_node = []
    for i in range(len(cpn_date)):
        cnp_node.append(int(((cpn_date[i]-issue_date).days/365)/d_t))
        
    ## Boundary condition    
    FDM_value = np.zeros((S_node,T_node+1))
    
    # Terminal coundary condition
    for i in range(S_node): 
        if s_range[i] >= s0 or s_range[i]>=Barrier:
            FDM_value[i,-1] = 1000*(1+cpn_rate)
            
        elif s_range[i] < s0 and s_range[i]>Barrier:
            FDM_value[i,-1] = 1000*(cpn_rate+(s_range[i]-s0)/s0)
            
        else:
            FDM_value[i,-1] = 1000*(1+(s_range[i]-s0)/s0)
            
    # Lower boundary condition
    FDM_value[0,:]= 0
    # Upper boundary condition
    indexing = 0
    for i in range(T_node):
        FDM_value[-1,i] = 1000*(1+cpn_rate) * np.exp(-(cnp_node[indexing] - i)*d_t)
        if i>= cnp_node[indexing]:
            indexing = indexing +1
    
    FDM_LU_value = FDM_value.copy()         
      
    # Calculate payoff for each node 
    for i in tqdm(np.arange(T_node-1,-1,-1)):
        if method == 'IFDM' or method =='CN':
           a = np.zeros(S_node); b = np.zeros(S_node); c = np.zeros(S_node); d = np.zeros(S_node)
           a[-1]=0 ; b[0] = 1 ; b[-1]=1 ; c[0]=0 ; d[0] = 0 ;
        else: pass   
                   
        for j in np.arange(1,S_node-1):
            
            if method == 'EFDM':
                a = (0.5*(sigma**2)*(j**2)+0.5*(r-div)*j)*d_t
                b = 1-r*d_t -(sigma**2)*(j**2)*d_t
                c = (0.5*(sigma**2)*(j**2)-0.5*(r-div)*j)*d_t
                FDM_value[j,i] = a * FDM_value[j+1,i+1] + b*FDM_value[j,i+1]+ c * FDM_value[j-1,i+1]
            
            elif method == 'IFDM':
                 a[j] = 0.5*((r-div)*j-(sigma**2)*(j**2))*d_t
                 b[j] = 1+((sigma**2)*(j**2))*d_t+r*d_t
                 c[j] = 0.5*(-(r-div)*j-(sigma**2)*(j**2))*d_t
                 d[j] = FDM_value[j,i+1]
                   
            else:
                 a[j] = 0.25*((sigma**2)*(j**2)-(r-div)*j)
                 b[j] = -0.5*((sigma**2)*(j**2)+r+2/d_t)
                 c[j] = 0.25*((sigma**2)*(j**2)+(r-div)*j)
                 d[j] = -0.25*((sigma**2)*(j**2)-(r-div)*j)*FDM_value[j-1,i+1] \
                        +0.5*((sigma**2)*(j**2)+r-(2/d_t))*FDM_value[j,i+1] \
                        -0.25*((sigma**2)*(j**2)+(r-div)*j)*FDM_value[j+1,i+1]
                              
        if method != 'EFDM':        
            d[-1] = FDM_value[-1,i] 
                   
            FDM_value[:,i] = TDMAsolver(a[1:],b,c[:-1],d)      
                
            x = np.matrix(np.diag(b)+np.diag(a[1:],k=-1)+np.diag(c[:-1],k=1))
            FDM_LU_value[:,i] = np.linalg.solve(x,d)
        else: pass    
           
        if i in cnp_node[1:-1]:
            
            if method == 'EFDM':
               FDM_value[barrier_node:,i] += FV*cpn_rate
               FDM_value[s0_node:,i] = FV*(1+cpn_rate)
                 
            else:
               FDM_LU_value[barrier_node:,i] += FV*cpn_rate
               FDM_LU_value[s0_node:,i] = FV*(1+cpn_rate)
               FDM_value[barrier_node:,i] += FV*cpn_rate
               FDM_value[s0_node:,i] = FV*(1+cpn_rate)
               
    if print_result == True: 
       print('{0} price is'.format(method),round(FDM_value[int(S_node/2),0],2))
       print('LU price is %4.3f',round(FDM_LU_value[int(S_node/2),0],2))
       
    return round(FDM_value[int(S_node/2),0],2) ,round(FDM_LU_value[int(S_node/2),0],2)

#%% Sensitivity Analysis
    
real_price = 958.90
T_range = np.arange(100,1001,100)
CN_error = []
EFDM_error = []
IFDM_error =  []
EFDM_est_price_list =[]
CN_est_price_list = []
IFDM_est_price_list = []
for time_node in tqdm(T_range):  

    CN_est_price = FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,S_node,time_node,T,issue_date,cpn_date,'CN')[0]
    EFDM_est_price = FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,S_node,time_node,T,issue_date,cpn_date,'EFDM')[0]
    IFDM_est_price = FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,S_node,time_node,T,issue_date,cpn_date,'IFDM')[0]
    CN_diff = CN_est_price - real_price
    EFDM_diff =EFDM_est_price - real_price
    IFDM_diff = IFDM_est_price - real_price
    CN_error.append(CN_diff)
    EFDM_error.append(EFDM_diff)
    IFDM_error.append(IFDM_diff)
    CN_est_price_list.append(CN_est_price)
    EFDM_est_price_list.append(EFDM_est_price)
    IFDM_est_price_list.append(IFDM_est_price)
#%%
plt.figure(0)
#plt.scatter(T_range,CN_error, color = 'b', label = ' CN ERROR')
plt.scatter(T_range,EFDM_error, color = 'r', label = ' EFDM ERROR')
#plt.scatter(T_range,IFDM_error, color = 'g', label = ' IFDM ERROR')
plt.xlabel('Number of Time node')
plt.ylabel('Error')
plt.title('Error depending on the number of time node')
plt.legend()
#%%
S_range = np.arange(100,1001,100)
CN_error2 = []
EFDM_error2 = []
IFDM_error2 =  []
EFDM_est_price_list2 =[]
CN_est_price_list2 = []
IFDM_est_price_list2 = []
for stcck_node in tqdm(S_range):  
    CN_est_price = FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,stcck_node,T_node,T,issue_date,cpn_date,'CN')[0]
    EFDM_est_price = FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,stcck_node,T_node,T,issue_date,cpn_date,'EFDM')[0]
    IFDM_est_price = FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,stcck_node,T_node,T,issue_date,cpn_date,'IFDM')[0]
    CN_diff = CN_est_price - real_price
    EFDM_diff =EFDM_est_price - real_price
    IFDM_diff = IFDM_est_price - real_price
    CN_error2.append(CN_diff)
    EFDM_error2.append(EFDM_diff)
    IFDM_error2.append(IFDM_diff)
    CN_est_price_list2.append(CN_est_price)
    EFDM_est_price_list2.append(EFDM_est_price)
    IFDM_est_price_list2.append(IFDM_est_price)
#%%
plt.figure(1)
plt.scatter(S_range,CN_error2, color = 'b', label = ' CN ERROR')
plt.scatter(S_range,EFDM_error2, color = 'r', label = ' EFDM ERROR')
plt.scatter(S_range,IFDM_error2, color = 'g', label = ' IFDM ERROR')
plt.xlabel('Number of Time node')
plt.ylabel('Error')
plt.title('Error depending on the number of stock node')
plt.legend()
#%%
sigma_range = np.arange(0.1,0.51,0.05)
CN_error3 = []
EFDM_error3 = []
IFDM_error3 =  []
EFDM_est_price_list3 =[]
CN_est_price_list3 = []
IFDM_est_price_list3 = []
for i in tqdm(range(len(sigma_range))):  
    CN_est_price = FDM_pricing(s0,r,div,sigma_range[i],Barrier_level,cpn_rate,FV,upper_ratio,S_node,T_node,T,issue_date,cpn_date,'CN')[0]
    EFDM_est_price = FDM_pricing(s0,r,div,sigma_range[i],Barrier_level,cpn_rate,FV,upper_ratio,S_node,T_node,T,issue_date,cpn_date,'EFDM')[0]
    IFDM_est_price = FDM_pricing(s0,r,div,sigma_range[i],Barrier_level,cpn_rate,FV,upper_ratio,S_node,T_node,T,issue_date,cpn_date,'IFDM')[0]
    CN_diff = CN_est_price - real_price
    EFDM_diff =EFDM_est_price - real_price
    IFDM_diff = IFDM_est_price - real_price
    CN_error3.append(CN_diff)
    EFDM_error3.append(EFDM_diff)
    IFDM_error3.append(IFDM_diff)
    CN_est_price_list3.append(CN_est_price)
    EFDM_est_price_list3.append(EFDM_est_price)
    IFDM_est_price_list3.append(IFDM_est_price)
#%%  
plt.figure(2)
#plt.scatter(sigma_range,CN_error3, color = 'b', label = ' CN ERROR')
plt.scatter(sigma_range,EFDM_error3, color = 'r', label = ' EFDM ERROR')
#plt.scatter(sigma_range,IFDM_error3, color = 'g', label = ' IFDM ERROR')
plt.xlabel('Standard Deviation')
plt.ylabel('Error')
plt.title('Error depending on volatility')
plt.legend()
#%%
s_node_range = [50,100,150,200]
t_node_range = [196,780,1758,3125]
CN_error4 = []
EFDM_error4 = []
IFDM_error4 =  []
EFDM_est_price_list4 =[]
CN_est_price_list4 = []
IFDM_est_price_list4 = []
for i in tqdm(range(len(s_node_range))):  
    CN_est_price = FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,s_node_range[i],t_node_range[i],T,issue_date,cpn_date,'CN')[0]
    EFDM_est_price = FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,s_node_range[i],t_node_range[i],T,issue_date,cpn_date,'EFDM')[0]
    IFDM_est_price = FDM_pricing(s0,r,div,sigma,Barrier_level,cpn_rate,FV,upper_ratio,s_node_range[i],t_node_range[i],T,issue_date,cpn_date,'IFDM')[0]
    CN_diff = CN_est_price - real_price
    EFDM_diff =EFDM_est_price - real_price
    IFDM_diff = IFDM_est_price - real_price
    CN_error4.append(CN_diff)
    EFDM_error4.append(EFDM_diff)
    IFDM_error4.append(IFDM_diff)
    CN_est_price_list4.append(CN_est_price)
    EFDM_est_price_list4.append(EFDM_est_price)
    IFDM_est_price_list4.append(IFDM_est_price)
#%%
plt.figure(3)
plt.scatter(s_node_range,CN_error4, color = 'b', label = ' CN ERROR')
plt.scatter(s_node_range,EFDM_error4, color = 'r', label = ' EFDM ERROR')
plt.scatter(s_node_range,IFDM_error4, color = 'g', label = ' IFDM ERROR')
plt.xlabel('the number of stock node')
plt.ylabel('Error')
plt.title('Error depending on the number of nodes')
plt.legend()
