#!/usr/bin/env python
# coding: utf-8

# In[88]:

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import scipy
from scipy import interpolate
from .utils import vec_find_iv_rat, gen_bm_path, horner_vector, sim_ou_process, bs_price_call

class TwoFactorQuinticOUModel:
    
    @staticmethod
    def compute_quadrature(n_t=100):
        """
        The number of quadrature points used to compute the integral in time
        """
        return np.polynomial.legendre.leggauss(n_t + 1)
        
    @staticmethod
    def doublefactorial(n):
        """
        Return the double factorial of a number.
        """
        if n <= 0:
            return 1
        else:
            return n * TwoFactorQuinticOUModel.doublefactorial(n-2)
    
    def gaussian_moments(self,sigma, deg):
        """
        Compute Gaussian moments for given degree and sigma.
        """
        if deg % 2 == 0:
            return TwoFactorQuinticOUModel.doublefactorial(deg-1) * sigma**deg
        else:
            return sigma*0.0  # For odd degrees, the moment is 0
    
    def __init__(self, params, fv_curves):
        """
        Initialize the model with the necessary parameters.
        
        """
        """
        Initialize the model with the necessary parameters from a dictionary.
        """
        # Unpack parameters from the dictionary
        self.theta = params.get('theta')
        self.rho = params.get('rho')
        self.lamda_x = params.get('lamda_x')
        self.lamda_y = params.get('lamda_y')
        self.p_k = params.get('p_k')
        self.T_nodes = fv_curves.get('T_nodes')
        self.int_var_nodes = fv_curves.get('int_var_nodes')
        
        # Load quantization points and weights
        self.q_points, self.q_weights = self.load_quantization_points()
        
    def show_parameters(self):
        """
        Show the main model parameters.
        
        """
        print(f"Model Parameters:")
        print(f"  theta: {self.theta}")
        print(f"  rho: {self.rho}")
        print(f"  lamda_x: {self.lamda_x}")
        print(f"  lamda_y: {self.lamda_y}")
        print(f"  p_k: {self.p_k}")

    def show_fwd_var(self):
        """
        Show forward var
        """
        print(f"  T_nodes: {self.T_nodes}\n")
        print(f"  int_var_nodes: {self.int_var_nodes}")
        
    def plot_fwd_int_var(self):
        r"""
        Plot the forward integrated variance $\int_0^T \xi_0(t)dt$
        """
        t_grids = np.linspace(0,np.max(self.T_nodes),20000)      
        inter_var_spline = interpolate.splrep(self.T_nodes, self.int_var_nodes,k=3)
        int_var = interpolate.splev(t_grids, inter_var_spline, der=0) #### int_var has to be positive
        plt.figure(1,figsize=(10,1.2))
        plt.title(r'$\int_0^T \xi_0(t)dt$')
        plt.plot(t_grids,int_var)
        plt.xlabel('T')
        plt.show()


    def co_var_func(self,theta, lamda_x, lamda_y,t, show_covar=False):

        var_1 = (1-np.exp(-2*lamda_x*(t)))/(2*lamda_x)
        var_2 = (1-np.exp(-2*lamda_y*(t)))/(2*lamda_y)
        covar = (1-np.exp(-(lamda_x+lamda_y)*(t)))/\
                    (lamda_x+lamda_y)
        var = theta**2*var_1+(1-theta)**2*var_2+ 2*theta*(1-theta)*covar

        if show_covar==False:
            return var
        else:
            return covar
    
    def load_quantization_points(self):
        """
        Method to load the quantization points and weights from the file.
        """
        ### TO BE UPDATED LATER TO INCLUDE OTHER POINTS
        
        path_quantization = os.path.dirname(os.path.realpath(__file__)) # pointint to the directory while the package is at
        q_all_points = np.array(pd.read_csv(path_quantization+'\\gauss_2d_quantization_1450',header=None,delim_whitespace=True))
        q_points = q_all_points[:-1,1:3]
        q_weights = q_all_points[:-1,0].reshape(-1,1)
        return q_points, q_weights
    
    def vix_pricing_quantization(self, T, strike_perc_vix, compute_iv=True):
        """
        Method to calculate VIX pricing with 2-factor quantization.
        """
        delt = 30 / 360
        T_delta = T + delt
        
        # Compute quadrature points and weights only once if not computed already
        if not hasattr(self, 'tt_gs'):
            self.tt_gs, self.w_t_gs = self.compute_quadrature()
        
        w_tt, tt = self.w_t_gs / 2 * (T_delta - T), 0.5 * (self.tt_gs + 1) * (T_delta - T) + T

        # Compute g_0 squared
        inter_var_spline = interpolate.splrep(self.T_nodes, self.int_var_nodes,k=3)
        FV_curve_all_vix = interpolate.splev(tt, inter_var_spline, der=1)   #### xi_0(t) for t smaller than the grid this the first value of fvc
        FV_curve_all_vix = np.sqrt(FV_curve_all_vix**2)   #### xi_0(t) needs to be positive
        
        std_Z_t = np.sqrt(self.co_var_func(self.theta, self.lamda_x, self.lamda_y, tt))
        cauchy_product = np.convolve(self.p_k, self.p_k)
        n = len(self.p_k)
        normal_cst = np.sum(cauchy_product[np.arange(0, 2*n, 2)].reshape(-1,1) * std_Z_t**(np.arange(0, 2*n, 2).reshape(-1,1)) * np.maximum(scipy.special.factorial2(np.arange(0, 2*n, 2).reshape(-1,1)-1), 1), axis=0)
        cauchy_product = np.convolve(self.p_k, self.p_k)
        g_0_s_squared = FV_curve_all_vix / normal_cst
        
        # Compute moments of G_s^t
        std_G_s_t = np.sqrt(self.co_var_func(self.theta, self.lamda_x, self.lamda_y, tt - T))
        combinatory_matrix = scipy.special.comb(np.arange(2*n-1), np.arange(2*n-1).reshape(-1,1))
        mmt_G_s_t = np.array([self.gaussian_moments(std_G_s_t, deg) for deg in np.arange(2*n-1)])            

        # Convert 2d standard Gaussian (i.i.d.) quantizer to correlated 2d Gaussian with moment matching
        var_X_t = self.co_var_func(1, self.lamda_x, self.lamda_y, T)
        var_Y_t = self.co_var_func(0.0, self.lamda_x, self.lamda_y, T)
        covar_XY_t = self.co_var_func(0.5, self.lamda_x, self.lamda_y, T, True)
        corr_XY_t = covar_XY_t / np.sqrt(var_X_t * var_Y_t)
        X = self.q_points[:,:1] * np.sqrt(var_X_t)
        Y = (corr_XY_t * self.q_points[:,:1] + np.sqrt(1 - corr_XY_t**2) * self.q_points[:,1:]) * np.sqrt(var_Y_t)
        mvt_match_adj_X = (3 * var_X_t**2 / np.sum(X**4 * self.q_weights))**(1/4)
        mvt_match_adj_Y = (3 * var_Y_t**2 / np.sum(Y**4 * self.q_weights))**(1/4)
        X = X * mvt_match_adj_X
        Y = Y * mvt_match_adj_Y

        # Compute VIX squared
        X_power = X ** np.arange(0, 2*n-1)[np.newaxis]
        Y_power = Y ** np.arange(0, 2*n-1)[np.newaxis]
        l_array = np.arange(0, 2*n-1)
        beta_m_l = np.zeros((n*2-1, n*2-1))
        vix_T_squared = 0.0
        for m in range(0, 2*n-1):
            for l in range(m, 2*n-1):
                integrand_2 = np.exp(-(m * self.lamda_x + (l - m) * self.lamda_y) * (tt - T)) 
                k_array = np.arange(l, 2*n-1)
                integral = np.sum((mmt_G_s_t[k_array - l] * integrand_2 * g_0_s_squared) * w_tt, axis=1)
                beta_m_l[m, l] = np.sum(cauchy_product[k_array] * integral * combinatory_matrix[l, l:]) * combinatory_matrix[m, l] * (1 - self.theta)**(l - m) * self.theta**m        
            vix_T_squared += np.sum(X_power[:, m:m+1] * Y_power[:, l_array[m:] - m] * beta_m_l[m, l_array[m:]], axis=1)
        vix_T_squared = vix_T_squared / delt
        
        # Compute VIX future and VIX call options
        vix_T = np.sqrt(vix_T_squared).reshape(-1, 1)
        Ft = np.sum(vix_T * self.q_weights)
        vix_strike = strike_perc_vix * Ft
        vix_opt = np.sum(np.maximum(vix_T - vix_strike, 0.0) * self.q_weights, axis=0)

        if compute_iv:
            flag = 'c'
            vix_iv = vec_find_iv_rat(vix_opt, Ft, Ft * strike_perc_vix, T, 0.0, flag)
            return Ft * 100, vix_opt * 100, vix_iv
        else:
            return Ft * 100, vix_opt * 100
            
    def vix_pricing_mc(self, T,strike_perc_vix,Z_mc,compute_iv = True):  

        delt = 30/360
        T_delta = T+delt
      
        # Compute quadrature points and weights only once if not computed already
        if not hasattr(self, 'tt_gs'):
            self.tt_gs, self.w_t_gs = self.compute_quadrature()

        w_tt, tt = self.w_t_gs / 2 * (T_delta - T), 0.5 * (self.tt_gs + 1) * (T_delta - T) + T

        # Compute g_0 squared
        inter_var_spline = interpolate.splrep(self.T_nodes, self.int_var_nodes,k=3)
        FV_curve_all_vix = interpolate.splev(tt, inter_var_spline, der=1)   #### xi_0(t) for t smaller than the grid this the first value of fvc
        FV_curve_all_vix = np.sqrt(FV_curve_all_vix**2)   #### xi_0(t) needs to be positive
        
        std_Z_t = np.sqrt(self.co_var_func(self.theta, self.lamda_x, self.lamda_y,tt))
        cauchy_product = np.convolve(self.p_k,self.p_k)
        n = len(self.p_k)
        normal_cst = np.sum(cauchy_product[np.arange(0,2*n,2)].reshape(-1,1)*std_Z_t**(np.arange(0,2*n,2).reshape(-1,1))*\
        np.maximum(scipy.special.factorial2(np.arange(0,2*n,2).reshape(-1,1)-1),1),axis=0)
        
        std_Z_t = np.sqrt(self.co_var_func(self.theta, self.lamda_x, self.lamda_y, tt))
        cauchy_product = np.convolve(self.p_k, self.p_k)
        n = len(self.p_k)
        normal_cst = np.sum(cauchy_product[np.arange(0, 2*n, 2)].reshape(-1,1) * std_Z_t**(np.arange(0, 2*n, 2).reshape(-1,1)) * np.maximum(scipy.special.factorial2(np.arange(0, 2*n, 2).reshape(-1,1)-1), 1), axis=0)

        cauchy_product = np.convolve(self.p_k, self.p_k)
        
        g_0_s_squared = FV_curve_all_vix / normal_cst
        std_G_s_t = np.sqrt(self.co_var_func(self.theta, self.lamda_x, self.lamda_y, tt - T))
        beta_m_l = np.zeros((n*2-1, n*2-1))
        combinatory_matrix = scipy.special.comb(np.arange(2*n-1), np.arange(2*n-1).reshape(-1,1))
        mmt_G_s_t = np.array([self.gaussian_moments(std_G_s_t, deg) for deg in np.arange(2*n-1)])            


        var_X_t = self.co_var_func(1, self.lamda_x, self.lamda_y, T)
        var_Y_t = self.co_var_func(0.0, self.lamda_x, self.lamda_y, T)
        covar_XY_t = self.co_var_func(0.5, self.lamda_x, self.lamda_y, T, True)
        corr_XY_t = covar_XY_t / np.sqrt(var_X_t * var_Y_t)
        
        X = Z_mc[0:1].T*np.sqrt(var_X_t)
        Y = (corr_XY_t*Z_mc[0:1].T+np.sqrt(1-corr_XY_t**2)*Z_mc[1:2].T)*np.sqrt(var_Y_t)
        X_power = X**np.arange(0,2*n-1)[np.newaxis]
        Y_power = Y**np.arange(0,2*n-1)[np.newaxis]
        l_array = np.arange(0,2*n-1)

        vix_T_squared = 0.0
        for m in range(0,2*n-1):
            for l in range(m,2*n-1):
                integrand_2 = np.exp(-(m*self.lamda_x+(l-m)*self.lamda_y)*(tt-T))
                k_array = np.arange(l,2*n-1)
                integral = np.sum((mmt_G_s_t[k_array-l]*integrand_2*g_0_s_squared)*w_tt,axis=1)
                beta_m_l[m,l] = np.sum(cauchy_product[k_array]*integral*combinatory_matrix[l,l:])*\
                    combinatory_matrix[m,l]*(1-self.theta)**(l-m)*self.theta**m        
            vix_T_squared+=np.sum(X_power[:,m:m+1]*Y_power[:,l_array[m:]-m]*beta_m_l[m,l_array[m:]],axis = 1)
        vix_T_squared = vix_T_squared/delt
        vix_T = np.sqrt(vix_T_squared)
        Ft = np.average(vix_T)
        vix_strike = strike_perc_vix*Ft
        vix_call_payoff = np.maximum(vix_T-vix_strike.reshape(-1,1),0)
        vix_opt = np.average(vix_call_payoff,axis = 1)
        vix_opt_std = np.std(vix_call_payoff,axis = 1)

        if compute_iv:
            flag = 'c'
            vix_opt_u = vix_opt + vix_opt_std/np.sqrt(Z_mc.shape[1])*1.96
            vix_opt_l = vix_opt - vix_opt_std/np.sqrt(Z_mc.shape[1])*1.96
            vix_iv= vec_find_iv_rat(vix_opt, Ft, Ft*strike_perc_vix,T, 0.0, 'c')
            vix_iv_u = vec_find_iv_rat(vix_opt_u, Ft, Ft*strike_perc_vix,T, 0.0, 'c')
            vix_iv_l = vec_find_iv_rat(vix_opt_l, Ft, Ft*strike_perc_vix,T, 0.0, 'c')
            return Ft*100, vix_opt*100, vix_opt_std*100, vix_iv,vix_iv_u,vix_iv_l

        else:
            return Ft*100, vix_opt*100, vix_opt_std*100
        
    def spx_pricing_mc(self,x1,x2,S_0,T,strikes,n_steps,N_sims,w1,compute_iv = True):

        dt = T / n_steps
        tt = np.linspace(0.0, T, n_steps + 1)
        t_vec = np.linspace(dt, T, n_steps)
        ti_1 = np.tile(t_vec, n_steps).reshape(n_steps, n_steps).T
        tj_1 = ti_1.T

        X1_t = sim_ou_process(self.lamda_x, x1, tt, w1)
        X2_t = sim_ou_process(self.lamda_y, x2, tt, w1)
        Xt = self.theta * X1_t + (1 - self.theta) * X2_t

        n = len(self.p_k)
        cauchy_product = np.convolve(self.p_k, self.p_k)
        var_X = (
            self.theta**2 / (2 * self.lamda_x) * (1 - np.exp(-2 * self.lamda_x * tt))
            + (1-self.theta)**2 / (2 * self.lamda_y) * (1 - np.exp(-2 * self.lamda_y * tt))
            + 2
            * self.theta*(1-self.theta)
            / (self.lamda_x + self.lamda_y)
            * (1 - np.exp(-(self.lamda_x + self.lamda_y) * tt))
        )
        std_X_t = np.sqrt(var_X)

        n = len(self.p_k)
        cauchy_product = np.convolve(self.p_k, self.p_k)
        normal_var = np.sum(
            cauchy_product[np.arange(0, 2 * n, 2)].reshape(-1, 1)
            * std_X_t ** (np.arange(0, 2 * n, 2).reshape(-1, 1))
            * np.maximum(
                scipy.special.factorial2(np.arange(0, 2 * n, 2).reshape(-1, 1) - 1), 1
            ),
            axis=0,
        )

        f_func = horner_vector(self.p_k[::-1], len(self.p_k), Xt)
        volatility = f_func / np.sqrt(normal_var.reshape(-1, 1))
        del f_func
        
        inter_var_spline = interpolate.splrep(self.T_nodes, self.int_var_nodes,k=3)
        fv_curve = np.sqrt(interpolate.splev(tt, inter_var_spline, der=1)**2).reshape(-1, 1) #### xi_0 has to be positive       

        volatility = np.sqrt(fv_curve) * volatility

        logS1 = np.log(S_0)
        for i in range(w1.shape[0]):
            logS1 = logS1-0.5*dt*(volatility[i]*self.rho)**2+np.sqrt(dt)*self.rho*volatility[i]*w1[i]

        w1_shape = w1.shape
        del w1
        ST1 = np.exp(logS1)
        del logS1    

        int_var = np.sum(volatility[:-1,]**2*dt,axis=0)
        Q = np.max(int_var)+1e-9
        del volatility
        X = (bs_price_call(ST1,np.sqrt((1-self.rho**2)*int_var/T),T,strikes.reshape(-1,1))).T
        Y = (bs_price_call(ST1,np.sqrt(self.rho**2*(Q-int_var)/T),T,strikes.reshape(-1,1))).T
        del int_var
        del ST1
        eY = (bs_price_call(S_0,np.sqrt(self.rho**2*(Q)/T),T,strikes.reshape(-1,1))).T

        c = []
        for i in range(strikes.shape[0]):
            cova = np.cov(X[:,i]+10,Y[:,i]+10)[0,1]
            varg = np.cov(X[:,i]+10,Y[:,i]+10)[1,1]
            if (cova or varg)<1e-8:
                temp = 1e-40
            else:
                temp = np.nan_to_num(cova/varg,1e-40)
            temp = np.minimum(temp,2)
            c.append(temp)
        c = np.array(c)

        call_mc_cv1 = X-c*(Y-eY)
        del X
        del Y
        del eY
        p_mc_cv1 = np.average(call_mc_cv1,axis=0)
        std_mc_cv1 = np.std(call_mc_cv1,axis=0)

        if compute_iv:
            flag = 'c'
            imp_mc = vec_find_iv_rat(p_mc_cv1, S_0, strikes, T, 0.0, flag)
            imp_mc_upper = vec_find_iv_rat(p_mc_cv1 + 1.96*std_mc_cv1/(np.sqrt(w1_shape[1])), S_0, strikes, T, 0.0, flag)
            imp_mc_lower = vec_find_iv_rat(p_mc_cv1 - 1.96*std_mc_cv1/(np.sqrt(w1_shape[1])), S_0, strikes, T, 0.0, flag)

            return p_mc_cv1, std_mc_cv1, imp_mc, imp_mc_upper, imp_mc_lower

        else:
            return p_mc_cv1,std_mc_cv1
        
    def spx_ssr_mc(self,T,n_steps,N_sims,w1):
        
        # SSR calculations parameter
        h = 1e-4
        S_0 = 100.0
        lm = np.array([-h,0,h])
        strike_ssr = np.exp(lm)*S_0
        ssr_h = 0.0001
        rho_ssr = self.rho
        
        inter_var_spline = interpolate.splrep(self.T_nodes, self.int_var_nodes,k=3)
        v_0 = np.sqrt(interpolate.splev([0], inter_var_spline, der=1)[0]**2)   # v_0 needs to be positive
        incr_ssr = ssr_h * rho_ssr / np.sqrt(v_0)
        compute_iv = False
        
        opt_h,_ = self.spx_pricing_mc(0.0,0.0,S_0,T,strike_ssr,n_steps,N_sims,w1,compute_iv = False)
        opt_atm,_ = self.spx_pricing_mc(incr_ssr,incr_ssr,S_0,T,np.array([S_0]),n_steps,N_sims,w1,compute_iv = False)
        
        iv_atm = vec_find_iv_rat(opt_atm, S_0, S_0, T, 0.0, "c")
        iv_atm_h = vec_find_iv_rat(opt_h, S_0, strike_ssr, T, 0.0, "c")
        iv_atm_skew = (iv_atm_h[2] - iv_atm_h[0]) / (2 * h)
        iv_atm_h = iv_atm_h[1]
        ssr = ((iv_atm - iv_atm_h) / ssr_h / iv_atm_skew)[0]
        
        return ssr


# In[ ]:




