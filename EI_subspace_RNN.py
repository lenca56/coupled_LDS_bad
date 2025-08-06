import numpy as np
import scipy.stats as stats
from numpy.random import Generator, PCG64
import utils
from scipy.optimize import minimize, Bounds
import sys

class EI_subspace_RNN():
    """
    Class for fitting Excitatory-Inhibitory Recurrent Neural Network with K-dim (low) self-contained dynamics
    like in Lea's paper

    Notation: 
        N_e: number of excitatory units
        N_i: number of inhibitory units
        W_indices: weights indices in (N_e + N_i) x (N_e + N_i) that are non-zero (fixed)
        J: K x (N_e + N_i) projection subspace matrix 
    """

    def __init__(self, N_e, N_i, sparsity, J, seed):
        ''' 
        
        '''
        self.N_e, self.N_i, self.sparsity, self.J = N_e, N_i, sparsity, J
        self.K = J.shape[0]
        self.N = self.N_e + self.N_i

        # reproducible random fixed indices for non-zero values of weights
        self.N_weights = int(sparsity * self.N ** 2)
        self.w_ind = []
        for i in range(0,self.N):
            self.w_ind.append(self.N * i + i)

        rng1 = Generator(PCG64(seed))
        rand = rng1.integers(0, self.N * self.N, size = 2 * self.N_weights + self.N)

        count = 0
        i = 0
        while count < self.N_weights - self.N:
            if rand[i] not in self.w_ind:
                self.w_ind.append(rand[i])
                count += 1
            i += 1
        
        self.w_ind_unravel = np.array(np.unravel_index(self.w_ind, (self.N, self.N))).T

        self.w_ind_pos = []
        self.w_ind_neg = []
        for ind in range(self.N_weights):
            if self.w_ind_unravel[ind,1] <= self.N_e-1: # excitatory cell
                self.w_ind_pos.append(self.w_ind[ind])
            elif self.w_ind_unravel[ind,1] >= self.N_e: # inhibitory cell
                self.w_ind_neg.append(self.w_ind[ind])

    def build_full_weight_matrix(self, w, Dale_law=1):
        W = np.zeros((self.N,self.N))
        if w.shape[0] != self.N_weights:
            raise Exception('Values of weights do not match in length with non-zero indices of matrix weights')
        for ind in range(self.N_weights):
            if Dale_law == 1: # Dale's law needs to be respected
                if self.w_ind[ind] in self.w_ind_pos: # excitatory cell
                    W[self.w_ind_unravel[ind,0], self.w_ind_unravel[ind,1]] = w[ind]
                elif self.w_ind[ind] in self.w_ind_neg: # inhibitory cell
                    W[self.w_ind_unravel[ind,0], self.w_ind_unravel[ind,1]] = - w[ind]
                else:
                    raise Exception('Indices of non-zero values go beyond possible shape')
            elif Dale_law == 0:
                W[self.w_ind_unravel[ind,0], self.w_ind_unravel[ind,1]] = w[ind] # no E-I constraints
        return W
    
    def generate_stable_weights(self, R=0.85, Dale_law=1):
        ''' 
        like in Hannequin et al 2012

        R:
            spectral radius 
        '''
        prod = self.sparsity * (1-self.sparsity)
        w0 = R / np.sqrt(prod)
        w = np.zeros((self.N_weights, 1))
        if Dale_law == 1: # w is all positive and EI enforced when building W
            w[:] = w0 / np.sqrt(self.N)
        elif Dale_law == 0:
            w[:] = w0 / np.sqrt(self.N)
            w = np.random.choice([-1, 1], size=w.shape) * w
        return w
    
    def generate_or_initialize_weights_from_dynamics_LDS(self, A_target, R=0.85, zeta_alpha_beta_gamma_list = [(1,1,1,0)], Dale_law=1):

        # ADD TOLERANCE TO STOP WHEN ITERATING

        # step 1 like in Hannequin et al 2012
        w0 = self.generate_stable_weights(R=R, Dale_law=Dale_law) 
        W0 = self.build_full_weight_matrix(w0, Dale_law=Dale_law)

        JpJ = np.linalg.pinv(self.J) @ self.J
        for iter in range(50):
            if iter % 10 == 0:
                print(iter)
            if zeta_alpha_beta_gamma_list[0][1] != 0: # alpha != 0
                # to satisfy low-dim dynamics constraint
                W0 = JpJ @ W0 @ JpJ + (np.eye(self.N) - JpJ) @ W0 @ (np.eye(self.N) - JpJ)

            if zeta_alpha_beta_gamma_list[0][2] != 0: # beta != 0
                # to satisfy E-I balance 
                W0 = W0 - W0 @ np.ones((self.N,1)) @ np.ones((1, self.N)) / self.N

            # to keep only active weights & Dale's law
            if Dale_law == 1:
                w0 = np.abs(self.get_nonzero_weight_vector(W0, Dale_law=Dale_law))
            else:
                w0 = self.get_nonzero_weight_vector(W0, Dale_law=Dale_law)
            W0 = self.build_full_weight_matrix(w0, Dale_law=Dale_law)

            # print(np.linalg.norm(J @ W0 @ (np.eye(N)-JpJ))) # checked
            # print(np.linalg.norm(W0 @ np.ones((N,1))))
        
        w_all = np.zeros((len(zeta_alpha_beta_gamma_list), self.N_weights))
        loss_W = np.zeros((len(zeta_alpha_beta_gamma_list)+1, 3))
        w_old =  np.copy(w0)
        # step 2 (target LDS dynamics matrix - either true one or fit)
        for ind in range(len(zeta_alpha_beta_gamma_list)):
            print(zeta_alpha_beta_gamma_list[ind])
            zeta = zeta_alpha_beta_gamma_list[ind][0]
            alpha = zeta_alpha_beta_gamma_list[ind][1]
            beta = zeta_alpha_beta_gamma_list[ind][2]
            gamma = zeta_alpha_beta_gamma_list[ind][3]
            loss_W[ind, :] = self.check_loss_weights_LDS(w_old.flatten(), A_target, Dale_law=Dale_law)
            opt_fun = lambda w_flattened: self.loss_weights_target_LDS(w_flattened, A_target, zeta, alpha, beta, gamma, Dale_law=Dale_law)
            opt_grad = lambda w_flattened: self.gradient_weights_target_LDS(w_flattened, A_target, zeta, alpha, beta, gamma, Dale_law=Dale_law)
            if Dale_law == 1:
                bounds = [(0, None)] * w0.shape[0]
            elif Dale_law == 0:
                bounds = [(None, None)] * w0.shape[0]
            w = minimize(opt_fun, w_old.flatten(), jac=opt_grad, method='L-BFGS-B', bounds=bounds).x  
            w_old = np.copy(w)
            w_all[ind, :] = np.copy(w)
        
        loss_W[-1, :] = self.check_loss_weights_LDS(w.flatten(), A_target, Dale_law=Dale_law)
        W = self.build_full_weight_matrix(w, Dale_law=Dale_law)

        return W0, W, loss_W, w_all 
    
    def get_nonzero_weight_vector(self, W, Dale_law=1):
        w = np.zeros((self.N_weights, 1))
        for ind in range(self.N_weights):
            if Dale_law == 1:
                if self.w_ind[ind] in self.w_ind_pos: # excitatory cell
                    w[ind] = W[self.w_ind_unravel[ind,0], self.w_ind_unravel[ind,1]] 
                elif self.w_ind[ind] in self.w_ind_neg: # inhibitory cell
                    w[ind] = - W[self.w_ind_unravel[ind,0], self.w_ind_unravel[ind,1]]
            else:
                w[ind] = W[self.w_ind_unravel[ind,0], self.w_ind_unravel[ind,1]] 
        return w
    
    def build_network_covariance(self, s):
        return np.diag(s * np.ones((self.N)))
    
    def build_dynamics_covariance(self, s):
        return self.J @ np.diag(s * np.ones((self.N))) @ self.J.T
    
    def generate_parameters(self, D, K):
        ''' 
        Parameters
        ----------
        N_weights: int
            number of non-zero element in connectivity matrix of RNN
        D: int
            dimension of data y_t

        Returns
        -------
        w: N_weights x 1 numpy vector
            non-zero weight values
        b: dict of length 2
            b[0] = K x 1 numpy vector corresponding to input during preparatory period
            b[1] = K x 1 numpy vector corresponding to input during preparatory period
        s: int
            S = np.diag(s) is N x N covariance matrix of Gaussian RNN noise
        mu0: K x 1 numpy vector
            mean of Gaussian distr. of first latent
        Q0: K x K numpy array
            covariance of Gaussiant distr. of first latent
        C_: D x K numpy array
            output mapping from latents x_t to data y_t
        d: D x 1 numpy vector
            offset term for mapping of observations
        R: D x D numpy array
            covariance matrix of Gaussian observation noise
        '''
        s = np.random.uniform(0.5, 1.5, 1)

        if K == 1:
            b1 = np.random.uniform(0.75, 1.25, 1).reshape((1,1))
            b2 = np.random.uniform(0.75, 1.25, 1).reshape((1,1))
        else:
            # normalize for input to have norm 1 for K>=2
            b1 = np.random.normal(0, 1, K)
            b1 = b1.reshape((K,1))
            b1 = b1/np.linalg.norm(b1)
            b2 = np.random.normal(0, 1, K)
            b2 = b2.reshape((K,1))
            b2 = b2/np.linalg.norm(b2)
        b = {0: b1, 1:b2}

        C_ = np.random.normal(1, 2, (D,K))
        d = np.random.normal(2, 1, (D,1))

        mu0 = np.random.normal(0, 0.1, (K,1))
        Q0 = np.random.normal(0.3, 0.1, (K, K))
        Q0 = np.dot(Q0, Q0.T) # to make P.S.D
        Q0 = 0.5 * (Q0 + Q0.T) # to make symmetric

        R = np.random.normal(0.25, 0.5, (D, D))
        R = np.dot(R, R.T)
        R = 0.5 * (R + R.T)
        
        return b, s, mu0, Q0, C_, d, R
    
    def generate_latents_and_observations(self, U, T, A, b, s, mu0, Q0, C_, d, R):
        ''' 
        Parameters
        ----------
        S: number of trials
        T: number of time points in trial
        '''
        D = C_.shape[0]
        S = self.build_network_covariance(s)
        Q = self.build_dynamics_covariance(s)
        t_s = int(T/2)

        x = np.zeros((U, T, self.K, 1))
        y = np.zeros((U, T, D, 1))

        for u in range(U):
            x[u, 0] = np.random.multivariate_normal(mu0.flatten(), Q0).reshape((self.K,1))
            y[u, 0] = C_ @ x[u, 0] + d
            for i in range(1, T):
                x[u, i] = np.random.multivariate_normal( (A @ x[u, i-1] + b[i-1 >= t_s]).reshape((self.K)), Q).reshape((self.K,1))
                y[u, i] = np.random.multivariate_normal((C_ @ x[u, i] + d).reshape(D), R).reshape((D,1))
                
        return x, y
    
    def generate_network_activity(self, U, T, W, b, s, mu0, Q0):

        t_s = int(T/2)
        S = self.build_network_covariance(s)
        v = np.zeros((U, T, self.N, 1))
        J_pinv = np.linalg.pinv(self.J)
        for u in range(U):
            v[u, 0] = np.random.multivariate_normal((J_pinv @ mu0).flatten(), J_pinv @ Q0 @ J_pinv.T).reshape((self.N,1)) 
            for i in range(1,T):
                v[u, i] = np.random.multivariate_normal((W @ v[u, i-1] + J_pinv @ b[i-1 >= t_s]).reshape((self.N)), S).reshape((self.N,1))
                
        return v

    def Kalman_filter_E_step(self, y, W, b, s, mu0, Q0, C_, d, R):
        ''' 
        for each trial individually
        '''

        A = utils.build_dynamics_matrix_A(W, self.J)
        Q = self.build_dynamics_covariance(s)
        T = y.shape[0]
        t_s = int(T/2) # assume switch from preparatory to movement happens midway
        
        mu = np.zeros((T, self.K, 1))
        mu_prior = np.zeros((T, self.K, 1))
        V = np.zeros((T, self.K, self.K))
        V_prior = np.zeros((T, self.K, self.K))
        norm_fact = np.zeros((T))
        
        # first step
        mu_prior[0] = mu0
        V_prior[0] = Q0
        V[0] = np.linalg.inv(C_.T @ np.linalg.inv(R) @ C_  + np.linalg.inv(V_prior[0]))
        mu[0] = V[0] @ (C_.T @ np.linalg.inv(R) @ (y[0] - d) + np.linalg.inv(V_prior[0]) @ mu_prior[0])
        # - 0.5 * (2 * np.pi) ** self.K
        norm_fact[0] =  - 0.5 * np.log(np.linalg.det(V_prior[0])) - 0.5 * (y[0] - C_ @ mu_prior[0] - d).T @ np.linalg.inv(C_ @ V_prior[0] @ C_.T + R) @ (y[0] - C_ @ mu_prior[0] - d)

        for t in range (1,T):
            # prior update
            mu_prior[t] = A @ mu[t-1] + b[t-1 >= t_s]
            V_prior[t] = A @ V[t-1] @ A.T + Q

            # normalizing factor = log p(y_t|y_{1:t-1}) 
            # ignoring - 0.5 * K * np.log(2 * np.pi)
            norm_fact[t] = - 0.5 * np.log(np.linalg.det(C_ @ V_prior[t] @ C_.T + R)) - 0.5 * (y[t] - C_ @ mu_prior[t] - d).T @ np.linalg.inv(C_ @ V_prior[t] @ C_.T + R) @ (y[t] - C_ @ mu_prior[t] - d)
        
            # filter update
            V[t] = np.linalg.inv(C_.T @ np.linalg.inv(R) @ C_  + np.linalg.inv(V_prior[t]))
            mu[t] = V[t] @ (C_.T @ np.linalg.inv(R) @ (y[t] - d) + np.linalg.inv(V_prior[t]) @ mu_prior[t])

        # marginal log likelihood p(y_{1:T})
        ll = norm_fact.sum()

        return mu, mu_prior, V, V_prior, ll

    def Kalman_smoother_E_step(self, A, mu, mu_prior, V, V_prior):
        ''' 
        for each trial individually
        '''
        T = mu.shape[0]
    
        m = np.zeros((T, self.K, 1))
        cov = np.zeros((T, self.K, self.K))
        cov_next = np.zeros((T-1, self.K, self.K))

        # last step (equal to last Kalman filter output)
        m[-1] = mu[-1]
        cov[-1] = V[-1]

        for t in range (T-2,-1,-1):
            # auxilary matrix
            L = V[t] @ A.T @ np.linalg.inv(V_prior[t+1])

            # smoothing updates
            m[t] = mu[t] + L @ (m[t+1] - mu_prior[t+1])
            cov[t] = V[t] + L @ (cov[t+1] - V_prior[t+1]) @ L.T
            cov_next[t] = L @ cov[t+1]

        return m, cov, cov_next

    def closed_form_M_step(self, y, d, W, m, cov, cov_next):
        ''' 
        closed-form updates for all parameters except the weights
        '''
        A = utils.build_dynamics_matrix_A(W, self.J)
        U = y.shape[0]
        T = y.shape[1]
        t_s = int(T/2)

        M1 = np.sum(m, axis=tuple([0,1]))
        M1_T = np.sum(cov, axis=tuple([0,1]))
        M_next = np.sum(cov_next, axis=tuple([0,1]))
        Y1 = np.sum(y, axis=tuple([0,1]))
        Y2 = np.zeros((y.shape[2], y.shape[2]))
        Y_tilda = np.zeros((y.shape[2], self.K))
        M_first = np.zeros((self.K, self.K))
        M_last = np.zeros((self.K, self.K))

        for u in range(U):
            M_first = M_first + m[u,0] @ m[u,0].T
            M_last = M_last + m[u,-1] @ m[u,-1].T
            for t in range(0,T):
                M1_T = M1_T + m[u,t] @ m[u,t].T
                Y_tilda = Y_tilda + y[u,t] @ m[u,t].T
                Y2 = Y2 + y[u,t] @ y[u,t].T
                if t != T-1:
                    M_next = M_next + m[u,t] @ m[u,t+1].T
                    
        # updates first latent (average over different trials)
        mu0 = np.mean(m, axis=0)[0]
        Q0 = np.mean(cov, axis=0)[0] + 1/U * M_first - mu0 @ mu0.T

        # updates observation parameters
        # if np.linalg.cond(M1_T) > 10 ** 10:
        #     print('bad C')
        # C_ = (Y1 @ M1.T - T * U * Y_tilda) @ np.linalg.inv(M1 @ M1.T - T * U * M1_T)
        C_ = (Y_tilda - d @ M1.T) @ np.linalg.inv(M1_T)
        d = 1/(T*U) * (Y1 - C_ @ M1)

        # R = np.zeros((y.shape[2], y.shape[2]))
        # for u in range(U):
        #     for t in range(T):
        #         residual = y[u, t] - d  # Fix: Center the observations
        #         E_zt = m[u, t]  # Smoothed expectation of latent variable
        #         E_zt_ztT = cov[u, t] + np.outer(m[u, t], m[u, t])  # E[z_n z_n^T]
        #         R += (np.outer(residual, residual)  
        #             - C_ @ np.outer(E_zt, residual)  
        #             - np.outer(residual, E_zt) @ C_.T  
        #             + C_ @ E_zt_ztT @ C_.T)  
        # R /= (T * U)  # Normalize

        R = 1/(T*U) * (Y2 + T * U * d @ d.T - d @ Y1.T - Y1 @ d.T - Y_tilda @ C_.T - C_ @ Y_tilda.T + d @ M1.T @ C_.T + C_ @ M1 @ d.T + C_ @ M1_T @ C_.T)
        eigenvalues = np.linalg.eigvalsh(R)  # Compute eigenvalues (optimized for symmetric matrices)
        # if np.all(eigenvalues >= 0) == False:
        #     print('bad R')

        # updates dynamics parameters
        b = {0:'', 1:''}
        b[0] = np.mean(m[:,1:t_s+1], axis= tuple([0,1])) - A @ np.mean(m[:,0:t_s], axis= tuple([0,1]))
        b[1] = np.mean(m[:,t_s+1:T], axis=tuple([0,1])) - A @ np.mean(m[:,t_s:T-1], axis=tuple([0,1]))
        J_aux = np.linalg.inv(self.J @ self.J.T)
        s = np.trace(J_aux @ (M1_T - np.sum(cov[:,0], axis=0) - M_first + A @ (M1_T - np.sum(cov[:,-1], axis=0) - M_last) @ A.T - 2 * A @ M_next))
        s_aux = np.zeros((self.K, self.K))
        for u in range(U):
            for t in range(0,T):
                if t != T-1:
                    s_aux = s_aux + b[t >= t_s] @ b[t >= t_s].T - 2 * b[t >= t_s] @ (m[u,t+1].T - m[u,t].T @ A.T)
        s = s + np.trace(J_aux @ s_aux)
        s = s / (self.K * (T-1) * U)

        return b, s, mu0, Q0, C_, d, R
    
    def loss_weights_target_LDS(self, w_flattened, A_target, zeta=1, alpha=1, beta=1, gamma=0, Dale_law=1):
        ''' 
        for weight initialization procedure

        '''
    
        W = self.build_full_weight_matrix(w_flattened.reshape((self.N_weights, 1)), Dale_law=Dale_law)
        A = utils.build_dynamics_matrix_A(W, self.J)
        res = A - A_target
        Jpinv_aux = self. J @ W @ (np.identity((self.N)) - np.linalg.pinv(self.J) @ self.J)

        loss_W = 0.5 * zeta * np.trace(res @ res.T) # target of W is 
        loss_W = loss_W + 0.5 * alpha * np.trace(Jpinv_aux @ Jpinv_aux.T)
        loss_W = loss_W + 0.5 * beta * np.ones((self.N,1)).T @ W.T @ W @ np.ones((self.N,1)) 
        loss_W = loss_W + 0.5 * gamma * np.trace(W @ W.T)

        return loss_W[0,0] # to get scalar from 1 x 1 array
    
    def gradient_weights_target_LDS(self, w_flattened, A_target, zeta=1, alpha=1, beta=1, gamma=0, Dale_law=1):
        ''' 
        for weight initialization procedure
        
        '''
        
        W = self.build_full_weight_matrix(w_flattened.reshape((self.N_weights, 1)), Dale_law=Dale_law)
        A = utils.build_dynamics_matrix_A(W, self.J)
        res = A - A_target
        Jpinv_aux = self.J @ W @ (np.identity((self.N)) - np.linalg.pinv(self.J) @ self.J)

        grad_W = zeta * self.J.T @ res @ np.linalg.pinv(self.J).T # derivative of Tr((JWJ_pinv - A_target) @ (JWJ_pinv - A_target).T)
        grad_W = grad_W + alpha * self.J.T @ Jpinv_aux
        grad_W = grad_W + beta * W @ np.ones((self.N,1)) @ np.ones((self.N,1)).T
        grad_W = grad_W + gamma * W
        
        return self.get_nonzero_weight_vector(grad_W, Dale_law=Dale_law).flatten() # a sign switching has to happen again

    def loss_weights_M_step(self, w_flattened, s, b, m, cov, cov_next, alpha=1, beta=1, Dale_law=1):
        '''
        gradient of loss function of weights to be minimized
        '''
        
        W = self.build_full_weight_matrix(w_flattened, Dale_law=Dale_law)
        A = utils.build_dynamics_matrix_A(W, self.J)

        U = m.shape[0]
        T = m.shape[1]
        t_s = int(T/2)
        aux = np.zeros((self.K, self.K))
        M1_T1 = np.sum(cov[:,:-1], axis=tuple([0,1]))
        M_next = np.sum(cov_next, axis=tuple([0,1]))
        for u in range(U):
            for t in range(0,T-1):
                M1_T1 = M1_T1 + m[u,t] @ m[u,t].T
                M_next = M_next + m[u,t] @ m[u,t+1].T
                aux = aux + b[t >= t_s] @ m[u,t].T

        J_aux = np.linalg.inv(self.J @ self.J.T)
        Jpinv_aux = self. J @ W @ (np.identity((self.N)) - np.linalg.pinv(self.J) @ self.J)

        # negative complete data log likelihood term
        loss_W = - 1/s * - np.trace(A.T @ J_aux @ aux) 
        loss_W = loss_W - 1/s * - 0.5 * np.trace(A.T @ J_aux @ A @ M1_T1)
        loss_W = loss_W - 1/s * np.trace(J_aux @ A @ M_next)

        # regularization
        loss_W = loss_W + 0.5 * alpha * np.trace(Jpinv_aux @ Jpinv_aux.T)
        loss_W = loss_W + 0.5 * beta * np.ones((self.N,1)).T @ W.T @ W @ np.ones((self.N,1)) 
        
        return loss_W[0,0]
    
    def gradient_weights_M_step(self, w_flattened, s, b, m, cov, cov_next, alpha=1, beta=1, Dale_law=1):
        '''
        gradient of loss function of weights to be minimized
        '''

        W = self.build_full_weight_matrix(w_flattened, Dale_law=Dale_law)
        A = utils.build_dynamics_matrix_A(W, self.J)

        U = m.shape[0]
        T = m.shape[1]
        t_s = int(T/2)
        aux = np.zeros((self.K, self.K))
        M1_T1 = np.sum(cov[:,:-1], axis=tuple([0,1]))
        M_next = np.sum(cov_next, axis=tuple([0,1]))
        for u in range(U):
            for t in range(0,T-1):
                M1_T1 = M1_T1 + m[u,t] @ m[u,t].T
                M_next = M_next + m[u,t] @ m[u,t+1].T
                aux = aux + b[t >= t_s] @ m[u,t].T

        J_aux = np.linalg.inv(self.J @ self.J.T)
        Jpinv_aux = self. J @ W @ (np.identity((self.N)) - np.linalg.pinv(self.J) @ self.J)

        grad_W = -1/s * self.J.T @ J_aux @ (- aux - A @ M1_T1 + M_next.T) @ np.linalg.pinv(self.J).T
        grad_W = grad_W + alpha * self.J.T @ Jpinv_aux
        grad_W = grad_W + beta * W @ np.ones((self.N,1)) @ np.ones((self.N,1)).T

        return self.get_nonzero_weight_vector(grad_W, Dale_law=Dale_law).flatten()

    def check_loss_weights_LDS(self, w_flattened, A_target, Dale_law=1):

        W = self.build_full_weight_matrix(w_flattened.reshape((self.N_weights, 1)), Dale_law=Dale_law)
        A = utils.build_dynamics_matrix_A(W, self.J)
        res = A - A_target
        Jpinv_aux = self. J @ W @ (np.identity((self.N)) - np.linalg.pinv(self.J) @ self.J)

        loss1_W = 0.5 * np.trace(res @ res.T) # target of W is 
        loss2_W =  0.5  * np.trace(Jpinv_aux @ Jpinv_aux.T)
        loss3_W = 0.5 * np.ones((self.N,1)).T @ W.T @ W @ np.ones((self.N,1)) 

        return loss1_W, loss2_W, loss3_W[0,0]

    def check_loss_weights(self, w_flattened, b, s, m, cov, cov_next, Dale_law=1):

        W = self.build_full_weight_matrix(w_flattened.reshape((self.N_weights, 1)), Dale_law=Dale_law)
        A = utils.build_dynamics_matrix_A(W, self.J)

        U = m.shape[0]
        T = m.shape[1]
        t_s = int(T/2)
        aux = np.zeros((self.K, self.K))
        M1_T1 = np.sum(cov[:,:-1], axis=tuple([0,1]))
        M_next = np.sum(cov_next, axis=tuple([0,1]))
        for u in range(U):
            for t in range(0,T-1):
                M1_T1 = M1_T1 + m[u,t] @ m[u,t].T
                M_next = M_next + m[u,t] @ m[u,t+1].T
                aux = aux + b[t >= t_s] @ m[u,t].T

        J_aux = np.linalg.inv(self.J @ self.J.T)
        Jpinv_aux = self. J @ W @ (np.identity((self.N)) - np.linalg.pinv(self.J) @ self.J)

        # negative complete data log likelihood term
        loss1_W = - 1/s * - np.trace(A.T @ J_aux @ aux) 
        loss1_W = loss1_W - 1/s * - 0.5 * np.trace(A.T @ J_aux @ A @ M1_T1)
        loss1_W = loss1_W - 1/s * np.trace(J_aux @ A @ M_next)

        # regularization terms
        loss2_W =  0.5  * np.trace(Jpinv_aux @ Jpinv_aux.T)
        loss3_W = 0.5 * np.ones((self.N,1)).T @ W.T @ W @ np.ones((self.N,1)) 

        return loss1_W, loss2_W, loss3_W[0,0]

    def compute_ELBO(self, y, W, b, s, mu0, Q0, C_, d, R, m, cov, cov_next):

        U = m.shape[0]
        T = m.shape[1]
        t_s = int(T/2)
        R_inv = np.linalg.inv(R)
        Q0_inv = np.linalg.inv(Q0)
        Q = self.build_dynamics_covariance(s)
        Q_inv = np.linalg.inv(Q)
        A = utils.build_dynamics_matrix_A(W, self.J)

        ecll = 0
        for u in range(U):
            ecll +=  -0.5 * np.log(np.linalg.det(Q0)) + m[u,0].T @ Q0_inv @ mu0 - 0.5 * mu0.T @ Q0_inv @ mu0 - 0.5 * np.trace(Q0_inv @ (cov[u,0] + m[u,0] @ m[u,0].T))
            for t in range(T):
                ecll += - 0.5 * np.log(np.linalg.det(R)) - 0.5 * (y[u,t] - d).T @ R_inv @ (y[u,t] - d) + m[u,t].T @ C_.T @ R_inv @ (y[u,t] - d) - 0.5 * np.trace(C_.T @ R_inv @ C_ @ (cov[u,t] + m[u,t] @ m[u,t].T))
                if t!= 0:
                    b_ = b[t-1 >= t_s]
                    ecll += - 0.5 * np.log(np.linalg.det(Q)) - 0.5 * b_.T @ Q_inv @ b_ - 0.5 * np.trace(Q_inv @ (cov[u,t] + m[u,t] @ m[u,t].T)) \
                    - 0.5 * np.trace(A.T @ Q_inv @ A @ (cov[u,t-1] + m[u,t-1] @ m[u,t-1].T)) + np.trace(Q_inv @ A @ (cov_next[u, t-1] + m[u,t-1] @ m[u,t].T)) \
                    + (m[u,t].T - m[u,t-1].T @ A.T) @ Q_inv @ b_
        elbo = ecll + 0
        return ecll, elbo

    def fit_EM(self, y, init_w, init_b, init_s, init_mu0, init_Q0, init_C_, init_d, init_R, alpha=1, beta=1, Dale_law=1, max_iter=300):
        
        U = y.shape[0]
        T = y.shape[1]
        w = np.copy(init_w)
        b = init_b.copy()
        s = np.copy(init_s)
        mu0 = np.copy(init_mu0)
        Q0 = np.copy(init_Q0)
        C_ = np.copy(init_C_)
        d = np.copy(init_d)
        R = np.copy(init_R)
        
        # marginal log likelihood 
        ecll = np.zeros((max_iter+1))
        ll = np.zeros((max_iter+1, U))
        lossW = np.zeros((max_iter+1, 3))

        for iter in range(max_iter):
            # if iter % 10 == 0:
            #     print(iter)

            W = self.build_full_weight_matrix(w, Dale_law=Dale_law)
            A = utils.build_dynamics_matrix_A(W, self.J)

            m = np.zeros((U, T, self.K, 1))
            cov = np.zeros((U, T, self.K, self.K))
            cov_next = np.zeros((U, T-1, self.K, self.K))

            for u in range(U): # iterate across all trials
                # E-step
                mu, mu_prior, V, V_prior, ll[iter, u] = self.Kalman_filter_E_step(y[u], W, b, s, mu0, Q0, C_, d, R)
                m[u], cov[u], cov_next[u] = self.Kalman_smoother_E_step(A, mu, mu_prior, V, V_prior)
        
            ecll[iter], _ = self.compute_ELBO(y, W, b, s, mu0, Q0, C_, d, R, m, cov, cov_next)
            lossW[iter,0], lossW[iter,1], lossW[iter,2] = self.check_loss_weights(w, b, s, m, cov, cov_next, Dale_law=Dale_law)

            # # # checking - M-step separate just for one
            # _, _, _, _, _, _, R = self.closed_form_M_step(y, d, w, m, cov, cov_next)

            # M-step
            b, s, mu0, Q0, C_, d, R = self.closed_form_M_step(y, d, W, m, cov, cov_next)
            opt_fun = lambda w_flattened: self.loss_weights_M_step(w_flattened, s, b, m, cov, cov_next, alpha=alpha, beta=beta, Dale_law=Dale_law)
            opt_grad = lambda w_flattened: self.gradient_weights_M_step(w_flattened, s, b, m, cov, cov_next, alpha=alpha, beta=beta, Dale_law=Dale_law)
            if Dale_law == 1:
                bounds = [(0, None)] * init_w.shape[0]
            else:
                bounds = [(None, None)] * init_w.shape[0]
            w = minimize(opt_fun, w.flatten(), jac=opt_grad, method='L-BFGS-B', bounds=bounds).x 

        # compute loss and ecll and ll after last iteration
        W = self.build_full_weight_matrix(w, Dale_law=Dale_law)
        A = utils.build_dynamics_matrix_A(W, self.J)
        m = np.zeros((U, T, self.K, 1))
        cov = np.zeros((U, T, self.K, self.K))
        cov_next = np.zeros((U, T-1, self.K, self.K))
        for u in range(U): # iterate across all trials
            # E-step
            mu, mu_prior, V, V_prior, ll[-1, u] = self.Kalman_filter_E_step(y[u], W, b, s, mu0, Q0, C_, d, R)
            m[u], cov[u], cov_next[u] = self.Kalman_smoother_E_step(A, mu, mu_prior, V, V_prior)
        lossW[-1,0], lossW[-1,1], lossW[-1,2] = self.check_loss_weights(w, b, s, m, cov, cov_next, Dale_law=Dale_law)
        ecll[-1], _ = self.compute_ELBO(y, W, b, s, mu0, Q0, C_, d, R, m, cov, cov_next)
            
        return ecll, ll, lossW, w, b, s, mu0, Q0, C_, d, R 






        
    
    