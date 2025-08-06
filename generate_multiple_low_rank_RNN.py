import numpy as np
import pandas as pd
import coupled_LDS
import scipy.stats as stats
import scipy.linalg
from utils import *
import matplotlib.pyplot as plt
from plotting_utils import *
import autograd.numpy as anp 
import os
from sippy_unipi import *
from sklearn.linear_model import LinearRegression
import scipy.linalg as sl
from coupled_LDS import *

z = 0
for K1,K2 in [(2,1),(3,1),(5,2)]:

    S = 1000
    T = 100
    K = K1 + K2
    D = 50
    M = 2

    LDS = coupled_LDS(D, K1, K2, M)    
    # generate known inputs 
    u = LDS.generate_inputs(S,T)

    Mw, Nw, Um, Um_n, Un = generate_low_rank(D,K1,K2)

    true_A = np.zeros((K1+K2,K1+K2))
    true_A[:K1,:K1] = Un.T @ Mw @ Nw.T @ Un # A11
    true_A[K1:,:K1] = Um_n.T @ Mw @ Nw.T @ Un # A21

    true_B, true_Q, true_mu0, true_Q0, true_C, true_d, true_R = LDS.generate_other_parameters()
    true_x, true_y = LDS.generate_latents_and_observations(S, T, u, true_A, true_B, true_Q, true_mu0, true_Q0, true_C, true_d, true_R)

    np.savez(f'models/K1={K1}_K2={K2}_true_parameters_and_data_low_rank', u=u, true_x=true_x, true_y=true_y, true_A=true_A, true_B=true_B, true_Q=true_Q, true_mu0=true_mu0, true_Q0=true_Q0, true_C=true_C, true_d=true_d, true_R=true_R)



