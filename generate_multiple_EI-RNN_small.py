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
from EI_subspace_RNN import * 

z = 0
for K1 in [2]:#[1,2,3]:
    for K2 in [2]:#[1,2,3]:

        S = 1000
        T = 100
        K = K1 + K2
        D = 50
        M = 2
        N_e = D * 5 # number of units in the RNN
        N_i = D * 5
        N = N_e + N_i
        sparsity = 0.25     
        N_weights = int (N * N * sparsity)

        LDS = coupled_LDS(D, K1, K2, M)    
        # generate known inputs 
        u = LDS.generate_inputs(S,T)

        J1 = np.random.normal(0, 1/np.sqrt(N), (N,N))
        J, _ = np.linalg.qr(J1)  # QR decomposition, Q is the orthogonal matrix
        J = J[:K1,:]
        J_inv = np.linalg.pinv(J) # pseudo-inverse (J * J_inv = identity, but J_inv * J is not)

        LDS1 = coupled_LDS(D, K1, 0, M)  
        eigvals1 = generate_eigenvalues(K1, R=1) # in disc of radius R = 1
        A11 = LDS1.generate_dynamics_matrix(eigvals1, np.array([]), disconnected=False)
        print(A11)
        
        RNN = EI_subspace_RNN(N_e, N_i, sparsity, J, seed=1)
        zeta_alpha_beta_gamma_list = [(10**i,1,1,10**(i-2)) for i in list(np.arange(-2,0.5,0.5))]

        initW0, initW, loss_W, w_all = RNN.generate_or_initialize_weights_from_dynamics_LDS(A_target=A11, R=0.85, zeta_alpha_beta_gamma_list = zeta_alpha_beta_gamma_list)
        # init_w = RNN.get_nonzero_weight_vector(initW)
        # initA = RNN.build_dynamics_matrix_A(initW, J)

        # true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R = RNN.generate_parameters(D, K)
        # true_x, true_y = RNN.generate_latents_and_observations(U, T, trueA, true_b, true_s, true_mu0, true_Q0, true_C_, true_d, true_R)
        
        # eigvals1 = generate_eigenvalues(K1, R=1) # in disc of radius R = 1
        # eigvals2 = generate_eigenvalues(K2, R=1) 

        # LDS = coupled_LDS(D, K1, K2, M)
        # true_A = LDS.generate_dynamics_matrix(eigvals1, eigvals2, disconnected=False)


        # true_B, true_Q, true_mu0, true_Q0, true_C, true_d, true_R = LDS.generate_other_parameters()
        # true_x, true_y = LDS.generate_latents_and_observations(S, T, u, true_A, true_B, true_Q, true_mu0, true_Q0, true_C, true_d, true_R)

        # np.savez(f'models/K1={K1}_K2={K2}_true_parameters_and_data_low_rank', u=u, true_x=true_x, true_y=true_y, true_A=true_A, true_B=true_B, true_Q=true_Q, true_mu0=true_mu0, true_Q0=true_Q0, true_C=true_C, true_d=true_d, true_R=true_R)



