import numpy as np
import scipy.stats as stats
from numpy.random import Generator, PCG64
import utils
from scipy.optimize import minimize, Bounds
import sys

import autograd.numpy as anp
import pymanopt 
import autograd.numpy as anp 
import scipy.linalg as sl

anp.random.seed(42)

class coupled_LDS():
    """
    Class for two-coupled LDS
     
    """

    def __init__(self, D, K1, K2, M):
        ''' 
        D: int
            dimensionality of data y
        K1: int
            dimensionality of task-dynamics self-contained space (first LDS)
        K2: int
            dimensionality of the other space (second LDS)
        M: int
            dimensionality of inputs u
        '''
        self.K1 = K1
        self.K2 = K2
        self.K = K1 + K2 # latent dimensionality of both systems together
        self.D = D # dim of data
        self.M = M # dim of inputs

    def generate_dynamics_matrix(self, eigvals1, eigvals2, mean_N=0, var_N=1, disconnected = False):

        # check that number of eigenvalues mathces latent dimensions
        if eigvals1.shape[0] != self.K1 or eigvals2.shape[0] != self.K2:
            raise Exception ('Number of eigenvalues in a system does not match its given dimensionality')

        # generate normal dynamics matrices for NOW !!!
        A1 = utils.generate_dynamics_A(eigvals1, normal=True, distr='normal')
        A2 = utils.generate_dynamics_A(eigvals2, normal=True, distr='normal')

        A = np.zeros((self.K, self.K))
        A[:self.K1,:self.K1] = A1
        A[self.K1:,self.K1:] = A2

        if disconnected == True:
            return A
        elif disconnected == False:
            A[self.K1:,:self.K1] = np.random.normal(mean_N, var_N, size=(self.K2,self.K1))
            return A
    
    def generate_dynamics_matrix_from_low_rank(self, eigvals1, eigvals2, mean_N=0, var_N=1, disconnected = False):

        # check that number of eigenvalues mathces latent dimensions
        if eigvals1.shape[0] != self.K1 or eigvals2.shape[0] != self.K2:
            raise Exception ('Number of eigenvalues in a system does not match its given dimensionality')

        # generate normal dynamics matrices for NOW !!!
        A1 = utils.generate_dynamics_A(eigvals1, normal=True, distr='normal')
        A2 = utils.generate_dynamics_A(eigvals2, normal=True, distr='normal')

        A = np.zeros((self.K, self.K))
        A[:self.K1,:self.K1] = A1
        A[self.K1:,self.K1:] = A2

        if disconnected == True:
            return A
        elif disconnected == False:
            A[self.K1:,:self.K1] = np.random.normal(mean_N, var_N, size=(self.K2,self.K1))
            return A

    def generate_inputs(self, S, T, type='constant'):
        ''' 
        note that only system 1 directly receives inputs 
        '''
        # u = np.zeros((S,T,self.M))
        # cond1 = np.random.normal(0, 1, size=(self.M))
        # cond2 = np.random.normal(0, 1, size=(self.M))
        # u[:int(S/2),:] = cond1
        # u[int(S/2):,:] = cond2

        u = np.zeros((S,T,self.M))
        cond1 = np.random.normal(0, 1, size=(T,self.M))
        cond2 = np.random.normal(0, 1, size=(T,self.M))
        u[:int(S/2)] = cond1
        u[int(S/2):] = cond2
        
        # check the rank of matrix U1_T that needs to be inverted for update for B
        U1_T = np.zeros((self.M, self.M)) # t = 1 to t = T_i - 1

        for s in range(S):
            for t in range(0,T-1):
                U1_T += np.outer(u[s,t],u[s,t])
        
        if np.linalg.cond(U1_T) > 2 ** 15: # condition number too large and matrix not invertible
            raise Exception('U1_T is not invertible and will cause problems for updates of B')

        # if type == 'constant': # constant inputs 
        #     u[:,:] = np.random.normal(0, 1, size=(self.M))
        # else:
        #     raise Exception ('Need to include other options that constant inputs')
        return u
    
    def generate_other_parameters(self):
        ''' 

        generates parameters except dynamics matrix A and inputs u 
        Note that C is constrained to be orthonormal matrix 

        Parameters
        ----------
        D: int
            dimension of data y_t

        Returns
        -------
        w: N_weights x 1 numpy vector
            non-zero weight values

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
        # Q = np.random.normal(1, 0.2, (self.K, self.K))
        # Q = np.dot(Q, Q.T)
        # Q = 0.5 * (Q + Q.T)

        B = np.zeros((self.K,self.M))
        B[:self.K1,:] = np.random.normal(0,1, size=(self.K1,self.M)) # inputs only arrive in system 1

        # Q is diagonal for now
        Q = np.diag(np.random.uniform(0.1, self.K/2, self.K))
        Q += 1e-8 * np.eye(Q.shape[0])
        
        # generate an orthonormal matrix C to actually be a projection matrix
        C = np.random.normal(0, 1, (self.D,self.D))
        C, _ = np.linalg.qr(C)  # QR decomposition, Q is the orthogonal matrix
        C = C[:self.K,:].T
        
        d = np.random.normal(2, 1, self.D)
        R = np.random.normal(0.25, 0.5, (self.D, self.D))
        R = np.dot(R, R.T) # to make P.S.D
        R = 0.5 * (R + R.T) # to make symmetric
        R += 1e-8 * np.eye(R.shape[0])

        mu0 = np.random.normal(0, 0.1, (self.K))
        Q0 = np.random.normal(1, 0.2, (self.K, self.K))
        Q0 = np.dot(Q0, Q0.T) # to make P.S.D
        Q0 = 0.5 * (Q0 + Q0.T) # to make symmetric
        Q0 += 1e-8 * np.eye(Q0.shape[0])
        
        return  B, Q, mu0, Q0, C, d, R
    
    
    def generate_latents_and_observations(self, S, T, u, A, B, Q, mu0, Q0, C, d, R):
        ''' 
        Parameters
        ----------
        S: number of trials/sessions
        T: number of time points in trial/session
        '''

        x = np.zeros((S, T, self.K))
        y = np.zeros((S, T, self.D))

        for s in range(S):
            x[s, 0] = np.random.multivariate_normal(mu0.flatten(), Q0)
            y[s, 0] = np.random.multivariate_normal((C @ x[s, 0] + d).reshape(self.D), R)
            for i in range(1, T):
                x[s, i] = np.random.multivariate_normal((A @ x[s, i-1] + B @ u[s,i-1]).reshape((self.K)), Q)
                y[s, i] = np.random.multivariate_normal((C @ x[s, i] + d).reshape(self.D), R)
                
        return x, y

    def Kalman_filter_E_step(self, y, u, A, B, Q , mu0, Q0, C, d, R):
        ''' 
        for each trial/session individually

        note that inputs come in only at the second time step
        '''

        T = y.shape[0]
        
        mu = np.zeros((T, self.K))
        mu_prior = np.zeros((T, self.K))
        V = np.zeros((T, self.K, self.K))
        V_prior = np.zeros((T, self.K, self.K))
        norm_fact = np.zeros((T))
        
        # first step
        mu_prior[0] = mu0 
        V_prior[0] = Q0
        V[0] = np.linalg.inv(C.T @ np.linalg.inv(R) @ C  + np.linalg.inv(V_prior[0]))
        mu[0] = V[0] @ (C.T @ np.linalg.inv(R) @ (y[0] - d) + np.linalg.inv(V_prior[0]) @ mu_prior[0])
        norm_fact[0] =  - 0.5 * np.log(np.linalg.det(C @ V_prior[0] @ C.T + R)) - 0.5 * (y[0] - C @ mu_prior[0] - d).T @ np.linalg.inv(C @ V_prior[0] @ C.T + R) @ (y[0] - C @ mu_prior[0] - d)

        for t in range (1,T):
            # prior update
            mu_prior[t] = A @ mu[t-1] + B @ u[t-1]
            V_prior[t] = A @ V[t-1] @ A.T + Q

            # normalizing factor = log p(y_t|y_{1:t-1}) 
            # ignoring - 0.5 * K * np.log(2 * np.pi)
            norm_fact[t] = - 0.5 * np.log(np.linalg.det(C @ V_prior[t] @ C.T + R)) - 0.5 * (y[t] - C @ mu_prior[t] - d).T @ np.linalg.inv(C @ V_prior[t] @ C.T + R) @ (y[t] - C @ mu_prior[t] - d)
        
            # filter update
            V[t] = np.linalg.inv(C.T @ np.linalg.inv(R) @ C  + np.linalg.inv(V_prior[t]))
            mu[t] = V[t] @ (C.T @ np.linalg.inv(R) @ (y[t] - d) + np.linalg.inv(V_prior[t]) @ mu_prior[t])

        # marginal log likelihood p(y_{1:T})
        ll = norm_fact.sum()
        ll -= 0.5 * T * self.D * np.log(2 * np.pi)

        return mu, mu_prior, V, V_prior, ll

    def Kalman_smoother_E_step(self, A, mu, mu_prior, V, V_prior):
        ''' 
        for each trial/session individually
        '''
        T = mu.shape[0]
    
        m = np.zeros((T, self.K))
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
    
    def compute_auxillary_matrices_M_step(self, u, y, m, cov, cov_next):
        S = y.shape[0]
        T = y.shape[1]

        M1 = np.sum(m, axis=tuple([0,1]))
        M1_T = np.sum(cov[:,:-1], axis=tuple([0,1])) # t = 1 to t= T_i-1 
        M_next = np.sum(cov_next, axis=tuple([0,1]))
        Y1 = np.sum(y, axis=tuple([0,1]))
        Y2 = np.zeros((self.D, self.D))
        Y_tilde = np.zeros((self.K, self.D))
        M_first = np.sum(cov[:,0], axis=0)
        M_last = np.sum(cov[:,-1], axis=0)
        U1_T = np.zeros((self.M, self.M)) # t = 1 to t = T_i - 1
        U_tilde = np.zeros((self.K, self.M))
        U_delta = np.zeros((self.K, self.M))

        for s in range(S):
            M_first += np.outer(m[s,0],m[s,0])
            M_last += np.outer(m[s,-1],m[s,-1])
            for t in range(0,T):
                Y_tilde += np.outer(m[s,t],y[s,t])
                Y2 += np.outer(y[s,t],y[s,t])
                if t != T-1:
                    M1_T += np.outer(m[s,t],m[s,t])
                    M_next += + np.outer(m[s,t],m[s,t+1])
                    U1_T += np.outer(u[s,t],u[s,t])
                    U_tilde += np.outer(m[s,t],u[s,t])
                    U_delta += np.outer(m[s,t+1],u[s,t])
        
        return M1, M1_T, M_next, Y1, Y2, Y_tilde, M_first, M_last, U1_T, U_tilde, U_delta
    
    # def loss_C(self, C, d, R, Y_tilde, M1, M1_T, M_last):
    #     ''' 
    #     optimize over Stiefel manifold of orthogonal matrices
    #     '''
    #     aux = np.linalg.inv(R) @ (-C @ Y_tilde + 0.5 * C @ M1_T @ C.T +  0.5 * C @ M_last @ C.T + np.outer(C @ M1,d))
    #     loss_C = np.trace(aux)
    #     return loss_C

    def modified_M_step(self, u, y, A, B, Q , mu0, Q0, C, d, R, m, cov, cov_next, verbosity):
        ''' 
        closed-form updates for all parameters except the C
        '''

        S = y.shape[0]
        T = y.shape[1]
        M1, M1_T, M_next, Y1, Y2, Y_tilde, M_first, M_last, U1_T, U_tilde, U_delta = self.compute_auxillary_matrices_M_step(u, y, m, cov, cov_next)
        
        # updates first latent (average over different trials/sessions S)
        mu0 = np.mean(m[:,0], axis=0)
        Q0 = 1/S * (M_first - np.outer(np.sum(m[:,0], axis=0), mu0)- np.outer(mu0, np.sum(m[:,0], axis=0)) + S * np.outer(mu0,mu0.T))

        # optimize over C
        C_anp = anp.asarray(C)
        R_inv_anp   = anp.linalg.inv(R)
        Y_tilde_anp = anp.asarray(Y_tilde)
        M1_T_anp    = anp.asarray(M1_T)
        M_last_anp  = anp.asarray(M_last)
        M1_anp      = anp.asarray(M1)
        d_anp   = anp.asarray(d)

        # closed form update for C
        manifold = pymanopt.manifolds.Stiefel(n=self.D, p=self.K)
        @pymanopt.function.autograd(manifold)
        def loss_C(C_anp): # d, R, Y_tilde, M1, M1_T, M_last
                ''' 
                optimize over Stiefel manifold of orthogonal matrices
                '''
                # aux = - C_anp @ Yw + 0.5 * C_anp @ M1_T_anp  @ C_anp.T + 0.5 * R_inv_anp @ C @ M_last_anp @ C.T + anp.outer(C_anp @ M1_anp, dw)
                # aux =  R_inv_anp @ (-C_anp @ Y_tilde_anp + 0.5 * C_anp @ M1_T_anp @ C_anp.T + anp.outer(C_anp @ M1_anp, d_anp))
                loss_C =  anp.trace(-C_anp @ Y_tilde_anp @ R_inv_anp) + anp.trace (0.5 * (M1_T_anp + M_last_anp) @ C_anp.T @ R_inv_anp @ C_anp)  + anp.trace(anp.outer(C_anp @ M1_anp, d_anp @ R_inv_anp))
                return loss_C
        
        # LONG TERM COULD TRY MANUAL C OPTIMIZATION WITH CONSTRAINT
        problem = pymanopt.Problem(manifold, loss_C)
        optimizer = pymanopt.optimizers.TrustRegions(max_iterations=50, verbosity=verbosity)
        # optimizer = pymanopt.optimizers.ConjugateGradient(max_iterations=10000, verbosity=verbosity) # SteepestDescent
        result = optimizer.run(problem, initial_point=C_anp)
        C = np.array(result.point)

        # update for d
        d = 1/(T*S) * (Y1 - C @ M1)

        # update for R
        R = 1/(T*S) * (Y2 + T * S * np.outer(d,d) - np.outer(d,Y1) - np.outer(Y1,d) - Y_tilde.T @ C.T - C @ Y_tilde + np.outer(d,M1) @ C.T + C @ np.outer(M1,d) + C @ M1_T @ C.T + C @ M_last @ C.T)

        # # FOR NUMERICAL STABILITY, MIGHT HAVE TO USE scipy.linalg.solve INSTEAD OF np.linalg.inv
        # blockwise update for A
        Qinv = np.linalg.inv(Q)
        # update for A_11
        A[:self.K1,:self.K1] = 2 * np.linalg.inv(Qinv[:self.K1,:self.K1]+Qinv[:self.K1,:self.K1].T) @ (Qinv[:self.K1,:self.K1].T @ M_next[:self.K1,:self.K1].T + Qinv[self.K1:,:self.K1].T @ M_next[:self.K1,self.K1:].T - 
                                0.5 * Qinv[self.K1:,:self.K1].T @ A[self.K1:,:self.K1] @ M1_T[:self.K1,:self.K1] - 0.5 * Qinv[:self.K1,self.K1:] @ A[self.K1:,:self.K1] @ M1_T[:self.K1,:self.K1] - 
                                0.5 * Qinv[:self.K1,self.K1:] @ A[self.K1:,self.K1:] @ M1_T[self.K1:,:self.K1] - 0.5 * Qinv[self.K1:,:self.K1].T @ A[self.K1:,self.K1:] @ M1_T[self.K1:,:self.K1] 
                                - Qinv[:self.K1,:self.K1].T @ B[:self.K1] @ U_tilde[:self.K1].T) @ np.linalg.inv(M1_T[:self.K1,:self.K1])
        # update for A_21
        A[self.K1:,:self.K1] = 2 * np.linalg.inv(Qinv[self.K1:,self.K1:]+Qinv[self.K1:,self.K1:].T) @ (Qinv[:self.K1,self.K1:].T @ M_next[:self.K1,:self.K1].T + Qinv[self.K1:,self.K1:].T @ M_next[:self.K1,self.K1:].T - 
                                0.5 * Qinv[:self.K1,self.K1:].T @ A[:self.K1,:self.K1] @ M1_T[:self.K1,:self.K1] - 0.5 * Qinv[self.K1:,:self.K1] @ A[:self.K1,:self.K1] @ M1_T[:self.K1,:self.K1] - 
                                0.5 * Qinv[self.K1:,self.K1:] @ A[self.K1:,self.K1:] @ M1_T[self.K1:,:self.K1] - 0.5 * Qinv[self.K1:,self.K1:].T @ A[self.K1:,self.K1:] @ M1_T[self.K1:,:self.K1] 
                                - Qinv[:self.K1,self.K1:].T @ B[:self.K1] @ U_tilde[:self.K1].T) @ np.linalg.inv(M1_T[:self.K1,:self.K1])
        # update for A_22
        A[self.K1:,self.K1:] = 2 * np.linalg.inv(Qinv[self.K1:,self.K1:]+Qinv[self.K1:,self.K1:].T) @ (Qinv[:self.K1,self.K1:].T @ M_next[self.K1:,:self.K1].T + Qinv[self.K1:,self.K1:].T @ M_next[self.K1:,self.K1:].T - 
                                0.5 * Qinv[:self.K1,self.K1:].T @ A[:self.K1,:self.K1] @ M1_T[:self.K1,self.K1:] - 0.5 * Qinv[self.K1:,:self.K1] @ A[:self.K1,:self.K1] @ M1_T[:self.K1,self.K1:] - 
                                0.5 * Qinv[self.K1:,self.K1:] @ A[self.K1:,:self.K1] @ M1_T[:self.K1,self.K1:] - 0.5 * Qinv[self.K1:,self.K1:].T @ A[self.K1:,:self.K1] @ M1_T[:self.K1,self.K1:] 
                                - Qinv[:self.K1,self.K1:].T @ B[:self.K1] @ U_tilde[self.K1:].T) @ np.linalg.inv(M1_T[self.K1:,self.K1:])

        
        # blockwise update for B
        # U1_T += 1e-8 * np.eye(U1_T.shape[0]) # to avoid singular matrix
        # B = (U_delta - A @ U_tilde) @ np.linalg.inv(U1_T)
        B[:self.K1] =  np.linalg.inv(Qinv[:self.K1,:self.K1]) @ (Qinv[:self.K1,:self.K1] @ U_delta[:self.K1] + Qinv[:self.K1,self.K1:] @ U_delta[self.K1:]
                        - Qinv[:self.K1,:self.K1] @ A[:self.K1,:self.K1] @ U_tilde[:self.K1] - Qinv[:self.K1,self.K1:] @ A[self.K1:,:self.K1] @ U_tilde[:self.K1]
                        - Qinv[:self.K1,self.K1:] @ A[self.K1:,self.K1:] @ U_tilde[self.K1:]) @ np.linalg.inv(U1_T)

        # update for Q
        Q = 1/((T-1)*S) * (M1_T - M_first + M_last + A @ M1_T @ A.T - A @ M_next - M_next.T @ A.T + B @ U1_T @ B.T - U_delta @ B.T - B @ U_delta.T + A @ U_tilde @ B.T + B @ U_tilde.T @ A.T)
    
        return A, B, Q, mu0, Q0, C, d, R
    

    def compute_ECLL(self, u, y, A, B, Q, mu0, Q0, C, d, R, m, cov, cov_next):

        S = y.shape[0]
        T = y.shape[1]
        M1, M1_T, M_next, Y1, Y2, Y_tilde, M_first, M_last, U1_T, U_tilde, U_delta = self.compute_auxillary_matrices_M_step(u, y, m, cov, cov_next)
        Q0_inv = np.linalg.inv(Q0)
        Q_inv = np.linalg.inv(Q)
        R_inv = np.linalg.inv(R)

        # first latent terms 
        ecll = - 0.5 * np.trace(Q0_inv @ M_first)
        ecll += np.trace(Q0_inv @ np.outer(np.sum(m[:,0,:], axis=0), mu0))
        ecll += -0.5 *S * np.trace(Q0_inv @ np.outer(mu0,mu0))

        # Q terms 
        ecll += -0.5 * np.trace(Q_inv @ (M1_T - M_first + M_last)) \
            + np.trace(Q_inv @ A @ M_next) \
            - 0.5 * np.trace(A.T @ Q_inv @ A @ M1_T)

        # R terms 
        ecll += -0.5 * np.trace(R_inv @ Y2) \
            + np.trace(R_inv @ C @ Y_tilde) \
            - 0.5 * np.trace(C.T @ R_inv @ C @ (M1_T + M_last))

        # d terms
        ecll += -0.5 * S * T * d.T @ R_inv @ d \
            + np.trace(R_inv @ np.outer(Y1,d)) \
            - np.trace(R_inv @ C @ np.outer(M1,d))

        # B, U terms
        ecll += -0.5 * np.trace(B.T @ Q_inv @ B @ U1_T) \
            + np.trace(B.T @ Q_inv @ U_delta) \
            - np.trace(B.T @ Q_inv @ A @ U_tilde)

        # logdet terms 
        ecll += - 0.5 * (T-1) * S * np.log(np.linalg.det(Q)) - 0.5 * (T-1) * S * Q.shape[0] * np.log(2*np.pi) \
            - 0.5 * T * S * np.log(np.linalg.det(R)) - 0.5 * T * S * R.shape[0] * np.log(2*np.pi) \
        - 0.5 * S * np.log(np.linalg.det(Q0)) - 0.5 * S * Q0.shape[0] * np.log(2*np.pi)

        H = 0.5 * S * T * self.K * np.log(2 * np.pi * np.e)
        for s in range(S):
            sign, logdet = np.linalg.slogdet(cov[s,0])
            H += 0.5 * sign * logdet
            for i in range(1,T):
                aux = cov[s,i] - cov_next[s,i-1].T @ np.linalg.inv(cov[s,i-1]) @ cov_next[s,i-1]
                sign, logdet = np.linalg.slogdet(aux)
                H += 0.5 * sign * logdet
        elbo = ecll + H

        return ecll, elbo

    def fit_EM(self, u, y, init_A, init_B, init_Q, init_mu0, init_Q0, init_C, init_d, init_R, max_iter=300, verbosity=0):
        
        S = y.shape[0]
        T = y.shape[1]

        A = np.copy(init_A)
        B = np.copy(init_B)
        Q = np.copy(init_Q)
        mu0 = np.copy(init_mu0)
        Q0 = np.copy(init_Q0)
        C = np.copy(init_C)
        d = np.copy(init_d)
        R = np.copy(init_R)
        
        # marginal log likelihood 
        ecll_old = np.zeros((max_iter))
        ecll_new = np.zeros((max_iter))
        elbo = np.zeros((max_iter))
        ll = np.zeros((max_iter, S))

        for iter in range(max_iter):
            if iter % 10 == 0:
                print(iter)

            m = np.zeros((S, T, self.K))
            cov = np.zeros((S, T, self.K, self.K))
            cov_next = np.zeros((S, T-1, self.K, self.K))

            for s in range(S): # iterate across all trials
                # E-step
                mu, mu_prior, V, V_prior, ll[iter, s] = self.Kalman_filter_E_step(y[s], u[s], A, B, Q, mu0, Q0, C, d, R)
                m[s], cov[s], cov_next[s] = self.Kalman_smoother_E_step(A, mu, mu_prior, V, V_prior)
            
            ecll_old[iter], elbo[iter] = self.compute_ECLL(u, y, A, B, Q, mu0, Q0, C, d, R, m, cov, cov_next)
            
            # M-step 
            A, B, Q, mu0, Q0, C, d, R = self.modified_M_step(u, y, A, B, Q, mu0, Q0, C, d, R, m, cov, cov_next, verbosity=verbosity)

            ecll_new[iter], _ = self.compute_ECLL(u, y, A, B, Q, mu0, Q0, C, d, R, m, cov, cov_next)
            
        # # compute loss and ecll and ll after last iteration
        # m = np.zeros((S, T, self.K))
        # cov = np.zeros((S, T, self.K, self.K))
        # cov_next = np.zeros((S, T-1, self.K, self.K))
        # for s in range(S): # iterate across all trials
        #     # E-step
        #     mu, mu_prior, V, V_prior, ll[-1, s] = self.Kalman_filter_E_step(y[s], u[s], A, B, Q , mu0, Q0, C, d, R)
        #     m[s], cov[s], cov_next[s] = self.Kalman_smoother_E_step(A, mu, mu_prior, V, V_prior)
        # ecll_old[-1], elbo[-1] = self.compute_ECLL(u, y, A, B, Q, mu0, Q0, C, d, R, m, cov, cov_next)
            
        return ecll_new, ecll_old, elbo, ll, A, B, Q , mu0, Q0, C, d, R
    
        # ecll = np.zeros(max_iter + 1)
        # elbo = np.zeros(max_iter + 1)
        # ll   = np.zeros((max_iter + 1, S))      # per-trial ll (kept for inspection)
        # ll_total = np.zeros(max_iter + 1)       # per-iteration total ll

        # for it in range(max_iter):
        #     if it % 10 == 0:
        #         print(it)

        #     # E-step (current θ): run filter/smoother for every trial
        #     m = np.zeros((S, T, self.K))
        #     cov = np.zeros((S, T, self.K, self.K))
        #     cov_next = np.zeros((S, T-1, self.K, self.K))

        #     for s in range(S):
        #         mu, mu_prior, V, V_prior, ll[it, s] = self.Kalman_filter_E_step(
        #             y[s], u[s], A, B, Q, mu0, Q0, C, d, R
        #         )
        #         m[s], cov[s], cov_next[s] = self.Kalman_smoother_E_step(A, mu, mu_prior, V, V_prior)

        #     # totals for this θ (BEFORE M-step)
        #     ll_total[it] = ll[it].sum()

        #     # ECLL for this same θ
        #     ecll[it], _ = self.compute_ECLL(u, y, A, B, Q, mu0, Q0, C, d, R, m, cov, cov_next)

        #     # Exact E-step ⇒ ELBO must equal marginal LL. Use identity to avoid entropy bugs.
        #     elbo[it] = ll_total[it]

        #     # (Optional: assert the equality numerically to catch issues early)
        #     # diff = ll_total[it] - ecll[it]
        #     # if np.abs(diff) > 1e-3:
        #     #     print(f"[warn] |LL - ECLL| = {diff:.3e} (expected to be entropy)")

        #     # M-step: update θ
        #     A, B, Q, mu0, Q0, C, d, R = self.modified_M_step(
        #         u, y, A, B, Q, mu0, Q0, C, d, R, m, cov, cov_next, verbosity=verbosity
        #     )

        # # Final E-step after last update to populate the (+1)-th slot
        # m = np.zeros((S, T, self.K))
        # cov = np.zeros((S, T, self.K, self.K))
        # cov_next = np.zeros((S, T-1, self.K, self.K))
        # for s in range(S):
        #     mu, mu_prior, V, V_prior, ll[-1, s] = self.Kalman_filter_E_step(
        #         y[s], u[s], A, B, Q, mu0, Q0, C, d, R
        #     )
        #     m[s], cov[s], cov_next[s] = self.Kalman_smoother_E_step(A, mu, mu_prior, V, V_prior)

        # ll_total[-1] = ll[-1].sum()
        # ecll[-1], _ = self.compute_ECLL(u, y, A, B, Q, mu0, Q0, C, d, R, m, cov, cov_next)
        # elbo[-1] = ll_total[-1]

        # return ecll, elbo, ll_total, A, B, Q, mu0, Q0, C, d, R





        
    
    