# File containing the covariance estimators as well as the different regularazations

import numpy as np
import numpy.linalg as la

def SCM(X):
    """ A function that computes Sample Covariance Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension

        Outputs:
            * SCM estimator
    """
    (p,n) = X.shape
    return(np.dot(X,X.conj().T)/n)

def SCM_LR(X,r):
    """ A function that computes low-rank Sample Covariance Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * r : rank

        Outputs:
            * LR-SCM estimator
    """
    (p,n) = X.shape
    Sigma = SCM(X)
    u,s,vh = la.svd(Sigma)
    u_signal = u[:,:r]
    u_noise = u[:,r:]
    sigma = np.mean(s[r:])
    Sigma = u_signal @ np.diag(s[:r])@u_signal.conj().T + sigma * u_noise@u_noise.conj().T
    return (Sigma)

def corr_phase(X):
    """ A function that computes phase only Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension

        Outputs:
            * PO estimator 
    """
    (p,n) = X.shape
    X = X/np.sqrt(abs(X)**2)
    return(np.dot(X,X.conj().T)/n)

def corr_phase_LR(X,r):
    """ A function that computes low-rank phase only Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * r : rank

        Outputs:
            * LR-PO estimator 
    """
    (p,n) = X.shape
    Sigma = corr_phase(X)
    u,s,vh = la.svd(Sigma)
    u_signal = u[:,:r]
    u_noise = u[:,r:]
    sigma = np.mean(s[r:])
    Sigma = u_signal @ np.diag(s[:r])@u_signal.conj().T + sigma * u_noise@u_noise.conj().T
    return(Sigma)

def regul_linear(Sigma,alpha):
    """ A function that computes the Shrinkage to identity for covariance matrix estimation
        Inputs:
            * Sigma covariance estimator
            * alpha : shrinkage parameter (<1)

        Outputs:
            * Shrinkage estimator
    """
    p = Sigma.shape[0]
    Sigma = alpha*Sigma+(1-alpha)*np.eye(p)*(np.trace(Sigma)/p) #Note:in roxane version, this is wrong: there should be sum of diag...
    return(Sigma)

def bandw(Sigma,band):
    """ A function that computes covariance matrix tapering for covariance matrix estimation
        Inputs:
            * Sigma covariance estimator
            * band : bandwidth parameter

        Outputs:
            * covariance matrix tapering estimator
    """
    N = Sigma.shape[0]
    transform = np.eye(N,N)
    
    for i in range(N-1):
        transform[i,(i):(i+1+band)] = 1 
        transform[(i):(i+1+band),i] = 1 
    Sigma = np.multiply(Sigma,transform)
    return(Sigma)

def tyler_estimator_covariance(X, tol=0.001, iter_max=20, return_tau=False):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * Sigma = the estimate
            * delta = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = X.shape
    delta = np.inf # Distance between two iterations
    Sigma = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (delta>tol) and (iteration<iter_max):
        
        # Computing expression of Tyler estimator (with matrix multiplication)
        tau = np.diagonal(X.conj().T@la.inv(Sigma)@X)
        X_bis = X / np.sqrt(tau)
        Sigma_new = (p/N) * X_bis@X_bis.conj().T

        # Imposing trace constraint: Tr(Sigma) = p
        Sigma_new = p*Sigma_new/np.trace(Sigma_new)

        # Condition for stopping
        delta = la.norm(Sigma_new - Sigma, 'fro') / la.norm(Sigma, 'fro')
        iteration = iteration + 1

        # Updating Sigma
        Sigma = Sigma_new

    # if iteration == iter_max:
    #     warnings.warn('Recursive algorithm did not converge')
    if return_tau:
        return (Sigma, tau)
    else:
        return Sigma
    
    
def tyler_estimator_covariance_LR(X, tol=0.001, iter_max=20, return_tau=False,r=1):
    """ A function that computes the non optimal Low-Rank Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
            * r = rank
        Outputs:
            * Sigma = the estimate
            * delta = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = X.shape
    delta = np.inf # Distance between two iterations
    Sigma = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (delta>tol) and (iteration<iter_max):
    
        # Computing expression of Tyler estimator (with matrix multiplication)
        tau = np.diagonal(X.conj().T@la.inv(Sigma)@X)
        # if iteration==0:
        #     tau = np.diagonal(X.conj().T@Sigma@X)
        # else:
        #     tau = np.diagonal(X.conj().T@la.inv(Sigma)@X)
            
        X_bis = X / np.sqrt(tau)
        Sigma_new = (p/N) * X_bis@X_bis.conj().T

        # Imposing trace constraint: Tr(Sigma) = p
        Sigma_new = p*Sigma_new/np.trace(Sigma_new)

        # Condition for stopping
        delta = la.norm(Sigma_new - Sigma, 'fro') / la.norm(Sigma, 'fro')
        iteration = iteration + 1

        # Updating Sigma
        Sigma = Sigma_new

    # LR structure
    u,s,vh = la.svd(Sigma)
    u_signal = u[:,:r]
    u_noise = u[:,r:]
    sigma = np.mean(s[r:])
    Sigma = u_signal @ np.diag(s[:r])@u_signal.conj().T + sigma * u_noise@u_noise.conj().T
    
    # if iteration == iter_max:
    #     warnings.warn('Recursive algorithm did not converge')
    
    if return_tau:
        return (Sigma, tau)
    else:
        return Sigma
    