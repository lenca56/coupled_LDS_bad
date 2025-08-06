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

df = pd.DataFrame(columns=['K1','K2','simulation']) # in total z=0,269 
z = 0
for K1 in [1,2,3]:
    for K2 in [1,2,3]:
        for simulation in range(30):
            df.loc[z, 'K1'] = K1
            df.loc[z, 'K2'] = K2
            df.loc[z, 'simulation'] = simulation
            z += 1 

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
K1 = df.loc[idx, 'K1']
K2 = df.loc[idx, 'K2']
simulation = df.loc[idx, 'simulation']
np.random.seed(simulation)

S = 1000
max_S = 200
T = 100
K = K1 + K2
D = 50
M = 2
LDS = coupled_LDS(D, K1, K2, M)
max_iter = 300

param = np.load(f'models/K1={K1}_K2={K2}_true_parameters_and_data_random.npz')
u=param['u']
true_x=param['true_x']
true_y=param['true_y']
true_A=param['true_A']
true_B=param['true_B']
true_Q=param['true_Q']
true_mu0=param['true_mu0']
true_Q0=param['true_Q0']
true_C=param['true_C']
true_d=param['true_d']
true_R=param['true_R']

if simulation == 0: # true
    ecll_new, ecll_old, elbo, ll, A, B, Q , mu0, Q0, C, d, R  = LDS.fit_EM(u, true_y, true_A, true_B, true_Q, true_mu0, true_Q0, true_C, true_d, true_R, max_iter=max_iter, verbosity=0)

elif simulation == 1: # C PCA + A REG + SCHUR
    y_flatten = true_y.reshape(true_y.shape[0] * true_y.shape[1], true_y.shape[2]) 
    y_mean   = y_flatten.mean(axis=0, keepdims=True)
    y_pca    = y_flatten - y_mean
    Y_cov = (y_pca.T @ y_pca) / y_pca.shape[0]              # Gram / inner‑product matrix
    eigvals, eigvecs = np.linalg.eigh(Y_cov)     # ascending order
    C_PCA = eigvecs[:, -K:][:, ::-1]             # take largest K, flip to descending D x K
    x_PCA = np.zeros((S,T,K))
    for s in range(S):
        x_PCA[s] = true_y[s] @ C_PCA
    x_PCA_before = x_PCA[:,:-1,:]
    x_PCA_before = x_PCA_before.reshape(x_PCA_before.shape[0] * x_PCA_before.shape[1], x_PCA_before.shape[2])
    x_PCA_after = x_PCA[:,1:,:]
    x_PCA_after = x_PCA_after.reshape(x_PCA_after.shape[0] * x_PCA_after.shape[1], x_PCA_after.shape[2])
    reg = LinearRegression().fit(x_PCA_before, x_PCA_after)
    A_REG = reg.coef_
    A_Schur, V = sl.schur(A_REG.T, output='real')
    A_Schur = A_Schur.T

    init_B, init_Q, init_mu0, init_Q0, init_C, init_d, init_R = LDS.generate_other_parameters()
    ecll_new, ecll_old, elbo, ll, A, B, Q , mu0, Q0, C, d, R  = LDS.fit_EM(u, true_y, A_Schur, init_B, init_Q, init_mu0, init_Q0, C_PCA, init_d, init_R, max_iter=max_iter, verbosity=0)

elif simulation == 2:   # Ho-Kalman SSID: C Ortho + A Schur
    method = "MOESP"
    U = u[:max_S].reshape(-1, M).T
    true_y_zerom = true_y[:max_S]-np.mean(true_y[:max_S], axis=(0,1), keepdims=True)
    Y = true_y_zerom.reshape(-1, D).T
    sys_id = system_identification(Y, U, method, SS_fixed_order=K, SS_p=2*K,             # # past block‐rows
        SS_f=2*K)
    Q, R = np.linalg.qr(sys_id.C, mode='reduced')   
    P = np.linalg.inv(R)                     
    C_orth = sys_id.C @ P # to get C orthogonal
    A_hat = R @ sys_id.A @ P
    A_Schur, V = sl.schur(A_hat.T, output='real')
    A_Schur = A_Schur.T

    init_B, init_Q, init_mu0, init_Q0, _, init_d, init_R = LDS.generate_other_parameters()
    ecll_new, ecll_old, elbo, ll, A, B, Q , mu0, Q0, C, d, R  = LDS.fit_EM(u, true_y, A_Schur, init_B, init_Q, init_mu0, init_Q0, C_orth, init_d, init_R, max_iter=max_iter, verbosity=0)

elif simulation >=3:
    eigvals1_init = generate_eigenvalues(K1, R=1) # in disc of radius R = 1
    eigvals2_init = generate_eigenvalues(K2, R=1) 
    init_A = LDS.generate_dynamics_matrix(eigvals1_init, eigvals2_init, disconnected=False)

    init_B, init_Q, init_mu0, init_Q0, init_C, init_d, init_R = LDS.generate_other_parameters()
    ecll_new, ecll_old, elbo, ll, A, B, Q , mu0, Q0, C, d, R  = LDS.fit_EM(u, true_y, init_A, init_B, init_Q, init_mu0, init_Q0, init_C, init_d, init_R, max_iter=max_iter, verbosity=0)
        
np.savez(f'models/K1={K1}_K2={K2}_fitted_param_simulation={simulation}', ecll_new=ecll_new, ecll_old=ecll_old, elbo=elbo, ll=ll, A=A, B=B, Q=Q , mu0=mu0, Q0=Q0, C=C, d=d, R=R)

