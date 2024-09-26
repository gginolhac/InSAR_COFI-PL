# File containing the cost functions and the euclidean gradients

import numpy as np
import numpy.linalg as la

import scipy.linalg as La

import jax.numpy as jnp
import jax.numpy.linalg as jla


def logm(Ci):
    """log matrix operator."""
    eigvals, eigvects = La.eigh(Ci)
    eigvals = np.diag(np.log(eigvals))
    Out = np.dot(np.dot(eigvects, eigvals), eigvects.conj().T)
    return Out

def logm_jax(Ci):
    """log matrix operator in jax."""
    eigvals, eigvects = jla.eigh(Ci)
    eigvals = jnp.diag(jnp.log(eigvals))
    Out = jnp.dot(jnp.dot(eigvects, eigvals), eigvects.conj().T)
    return Out

def costfunction_LS(Sigma_tilde, w_theta, Psi_tilde):
    """ least squares cost function  
    Inputs:
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * w_theta : complex vector of phasis
        * Psi_tilde : abs(Sigma_tilde)

    Output : cost : cost function value
    """
    cost = -w_theta.conj().T@((Psi_tilde*Sigma_tilde)@w_theta)
    return np.squeeze(np.real(cost))

def costfunction_LS_jax(Sigma_tilde, w_theta, Psi_tilde):
    """ least squares cost function in jax
    Inputs:
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * w_theta : complex vector of phasis
        * Psi_tilde : abs(Sigma_tilde)

    Output : cost : cost function value
    """
    cost = -w_theta.conj().T@((Psi_tilde*Sigma_tilde)@w_theta)
    return jnp.squeeze(jnp.real(cost))

def costfunction_KL(Sigma_tilde, w_theta, Psi_tilde):
    """ KL cost function  
    Inputs:
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * w_theta : complex vector of phasis
        * Psi_tilde : abs(Sigma_tilde)

    Output : cost : cost function value
    """
    Psi_tilde_inv = la.inv(Psi_tilde)
    cost = w_theta.conj().T@((Psi_tilde_inv*Sigma_tilde)@w_theta)
    return np.squeeze(np.real(cost))

def costfunction_KL_jax(Sigma_tilde, w_theta, Psi_tilde):
    """ KL cost function in jax 
    Inputs:
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * w_theta : complex vector of phasis
        * Psi_tilde : abs(Sigma_tilde)

    Output : cost : cost function value
    """
    Psi_tilde_inv = jla.inv(Psi_tilde)
    cost = w_theta.conj().T@((Psi_tilde_inv*Sigma_tilde)@w_theta)
    return jnp.squeeze(jnp.real(cost))

def costfunction_comet(Sigma_tilde, w_theta, Psi_tilde):
    """ weighted least squares cost function  
    Inputs:
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * w_theta : complex vector of phasis
        * Psi_tilde : abs(Sigma_tilde)

    Output : cost : cost function value
    """
    p = Sigma_tilde.shape[0]
    I = np.eye(p)
    d, q = la.eigh(Sigma_tilde)
    Sigma_tilde_inv_sqrt = q@np.diag(1/np.sqrt(d))@q.conj().T
    return np.real(la.norm(I - Sigma_tilde_inv_sqrt@(Psi_tilde*(w_theta@w_theta.conj().T))@Sigma_tilde_inv_sqrt)**2)

def costfunction_comet_jax(Sigma_tilde, w_theta, Psi_tilde):
    """ weighted least squares cost function in jax
    Inputs:
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * w_theta : complex vector of phasis
        * Psi_tilde : abs(Sigma_tilde)

    Output : cost : cost function value
    """
    p = Sigma_tilde.shape[0]
    I = jnp.eye(p)
    d, q = jla.eigh(Sigma_tilde)
    Sigma_tilde_inv_sqrt = q@jnp.diag(1/jnp.sqrt(d))@q.conj().T
    return jnp.real(jla.norm(I - Sigma_tilde_inv_sqrt@(Psi_tilde*(w_theta@w_theta.conj().T))@Sigma_tilde_inv_sqrt)**2)

def grad_costfunction_comet(Sigma_tilde, w_theta, Psi_tilde):
    """ euclidean gradient for the weighted least squares cost function  
    Inputs:
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * w_theta : complex vector of phasis
        * Psi_tilde : abs(Sigma_tilde)

    Output : grad : euclidean gradient
    """
    p = Sigma_tilde.shape[0]
    Sigma_inv = la.inv(Sigma_tilde)
    W = np.diag(w_theta.squeeze())
    M = Sigma_inv@W@Psi_tilde
    grad1 = 2*np.diag(M)
    grad2 = 4*np.diag(M@W.conj().T@M)
    grad = grad2-2*grad1
    return(grad.reshape((p,1)))


def grad_costfunction_LS(Sigma_tilde, w_theta):
    """ euclidean gradient for the least squares cost function  
    Inputs:
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * w_theta : complex vector of phasis

    Output : grad : euclidean gradient
    """
    M = -np.abs(Sigma_tilde)*Sigma_tilde
    return(2*M@w_theta)

def grad_costfunction_KL(Sigma_tilde, w_theta):
    """ euclidean gradient for the KL cost function  
    Inputs:
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * w_theta : complex vector of phasis

    Output : grad : euclidean gradient
    """
    M = la.inv(np.abs(Sigma_tilde))*Sigma_tilde
    return(2*M@w_theta)