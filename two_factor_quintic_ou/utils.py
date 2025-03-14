#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import scipy


from py_vollib.black_scholes.implied_volatility import implied_volatility

def vec_find_iv_rat(opt_price, S, K, t, r, flag):
    return np.vectorize(implied_volatility)(opt_price, S, K, t, r, flag)

def gen_bm_path(n_steps,N_sims):
    w1 = np.random.normal(0, 1, (n_steps, N_sims))
    #Antithetic variates
    w1 = np.concatenate((w1, -w1), axis = 1)
    return w1

def horner_vector(poly, n, x):
    #Initialize result
    result = poly[0].reshape(-1,1)
    for i in range(1,n):
        result = result*x + poly[i].reshape(-1,1)
    return result

def sim_ou_process(lamda, ou_init, t, w1):
    exp1 = np.exp(lamda * t)
    exp2 = np.exp(2 * lamda * t)
    diff_exp2 = np.concatenate((np.array([0.0]), np.diff(exp2)))
    std_vec = np.sqrt(diff_exp2 / (2 * lamda))[
        :, np.newaxis
    ]  # to be broadcasted columnwise
    exp1 = exp1[:, np.newaxis]
    ou = (1 / exp1) * np.cumsum(
        std_vec * np.concatenate((np.zeros(w1.shape[1])[np.newaxis, :], w1)), axis=0
    )
    ou = ou_init * np.exp(-lamda * t).reshape(-1, 1) + ou
    return ou

def bs_price_call(s,sigma,T,K):
    d1 = (np.log(s/K)+0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    price = s*scipy.special.ndtr(d1)-K*scipy.special.ndtr(d2)
    return price


# In[ ]:




