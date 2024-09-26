# File containing the optimization functions

import numpy as np
import numpy.linalg as la

import jax.numpy as jnp
from jax import grad, jit
#from jax.config import config # linux
#from jax._src import config # windows
from jax import config
config.update("jax_enable_x64",True)

from .manifold import ComplexCircle
from .costfunctions import costfunction_KL,costfunction_KL_jax, grad_costfunction_KL,\
                          costfunction_LS_jax, costfunction_LS, grad_costfunction_LS,\
                          costfunction_comet, costfunction_comet_jax,grad_costfunction_comet

# Euclidean gradient provided by jax
grad_euc_KL_func = jit(grad(costfunction_KL_jax, argnums=[1]))
grad_euc_LS_func = jit(grad(costfunction_LS_jax, argnums=[1]))
grad_euc_comet_func = jit(grad(costfunction_comet_jax, argnums=[1]))


def RG_comet_IPL(Sigma_tilde,maxIter,auto,comput_cost,conjugate):
    """ Riemannian Gradient for weighted least squares cost function
    Input :
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * maxIter : maximum iteration of the gradient descent
        * auto : True : computation of the gradient by jax if true. Else analytical gradient
        * comput_cost : True : cost function is computed
        * conjugate : True to compute the conjugate gradient algorithm
    Output : 
        * w_theta : complex vector of phasis
        * cost : cost function if comput_cost = True
    """ 

    p = Sigma_tilde.shape[0]
    Psi_tilde = np.abs(Sigma_tilde)
    w_theta = np.ones((p,1),dtype=complex)
    w_theta_jnp = jnp.array(w_theta)
    w_theta_prev = w_theta
    grad_riemann_prev = np.zeros((p,1),dtype=complex)
    Sigma_tilde_jnp = jnp.array(Sigma_tilde)
    Psi_tilde_jnp = jnp.abs(Sigma_tilde_jnp)
    
    if comput_cost:
        cost = []
        cost_temp = costfunction_comet(Sigma_tilde, w_theta, Psi_tilde)
        cost.append(cost_temp)

    # Manifold
    CC = ComplexCircle(p)

    err = 1
    it = 1
    alpha_0 = 1e-3
    beta_0 = alpha_0
    while err > 1e-4 and it<maxIter:
        # Euclidean gradient
        if auto:
            grad_euc_jax = grad_euc_comet_func(Sigma_tilde_jnp, w_theta_jnp, Psi_tilde_jnp)[0].conj()
            grad_euc = np.array(grad_euc_jax)
        else:
            grad_euc = grad_costfunction_comet(Sigma_tilde, w_theta,Psi_tilde)

        # grad euclidien to grad riemannien
        grad_riemann = CC.projection(w_theta,grad_euc)

        # option : conjugate
        if conjugate:
            Xi = -alpha_0*grad_riemann - beta_0 * CC.transport(w_theta, w_theta_prev, grad_riemann_prev)
        else:
            Xi = -alpha_0*grad_riemann

        # retraction
        w_theta_new = CC.retraction(w_theta,Xi)

        # error
        err = la.norm(w_theta_new-w_theta)

        # update
        if conjugate:
            w_theta_prev = w_theta
            grad_riemann_prev = grad_riemann

        w_theta = w_theta_new
        w_theta_jnp = jnp.array(w_theta)
        it = it +1

        # cost function
        if comput_cost:
            cost_temp = costfunction_comet(Sigma_tilde, w_theta, Psi_tilde)
            cost.append(cost_temp)

    if comput_cost:
        cost=cost-min(cost)
        return(w_theta,cost)
    else:
        return(w_theta)

def RG_KL_IPL(Sigma_tilde,maxIter,auto,comput_cost,conjugate):
    """ Riemannian Gradient for KL cost function
    Input :
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * maxIter : maximum iteration of the gradient descent
        * auto : True : computation of the gradient by jax if true. Else analytical gradient
        * comput_cost : True : cost function is computed
        * conjugate : True to compute the conjugate gradient algorithm
    Output : 
        * w_theta : complex vector of phasis
        * cost : cost function if comput_cost = True
    """ 

    p = Sigma_tilde.shape[0]
    Psi_tilde = np.abs(Sigma_tilde)
    w_theta = np.ones((p,1),dtype=complex)
    w_theta_prev = w_theta
    grad_riemann_prev = np.zeros((p,1),dtype=complex)
    w_theta_jnp = jnp.array(w_theta)
    Sigma_tilde_jnp = jnp.array(Sigma_tilde)
    Psi_tilde_jnp = jnp.abs(Sigma_tilde_jnp)
    
    if comput_cost:
        cost = []
        cost_temp = costfunction_KL(Sigma_tilde, w_theta, Psi_tilde)
        cost.append(cost_temp)

    # Manifold
    CC = ComplexCircle(p)

    err = 1
    it = 1
    alpha_0 = 1e-2
    beta_0 = alpha_0/4
    while err > 1e-4 and it<maxIter:
        # Euclidean gradient
        if auto:
            grad_euc_jax = grad_euc_KL_func(Sigma_tilde_jnp, w_theta_jnp, Psi_tilde_jnp)[0].conj()
            grad_euc = np.array(grad_euc_jax)
        else:
            grad_euc = grad_costfunction_KL(Sigma_tilde, w_theta)

        # grad euclidien to grad riemannien
        grad_riemann = CC.projection(w_theta,grad_euc)

        # option : conjugate
        if conjugate:
            Xi = -alpha_0*grad_riemann - beta_0 * CC.transport(w_theta, w_theta_prev, grad_riemann_prev)
        else:
            Xi = -alpha_0*grad_riemann

        # retraction
        w_theta_new = CC.retraction(w_theta,Xi)

        # error
        err = la.norm(w_theta_new-w_theta)

        # update
        if conjugate:
            w_theta_prev = w_theta
            grad_riemann_prev = grad_riemann

        w_theta = w_theta_new
        w_theta_jnp = jnp.array(w_theta)
        it = it +1

        # cost function
        if comput_cost:
            cost_temp = costfunction_KL(Sigma_tilde, w_theta, Psi_tilde)
            cost.append(cost_temp)

    if comput_cost:
        cost=cost-min(cost)
        return(w_theta,cost)
    else:
        return(w_theta)

def RG_LS_IPL(Sigma_tilde,maxIter,auto,comput_cost,conjugate):
    """ Riemannian Gradient for Least Squares cost function
    Input :
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * maxIter : maximum iteration of the gradient descent
        * auto : True : computation of the gradient by jax if true. Else analytical gradient
        * comput_cost : True : cost function is computed
        * conjugate : True to compute the conjugate gradient algorithm
    Output : 
        * w_theta : complex vector of phasis
        * cost : cost function if comput_cost = True
    """ 

    p = Sigma_tilde.shape[0]
    Psi_tilde = np.abs(Sigma_tilde)
    w_theta = np.ones((p,1),dtype=complex)
    w_theta_jnp = jnp.array(w_theta)
    w_theta_prev = w_theta
    grad_riemann_prev = np.zeros((p,1),dtype=complex)
    Sigma_tilde_jnp = jnp.array(Sigma_tilde)
    Psi_tilde_jnp = jnp.abs(Sigma_tilde_jnp)
    
    if comput_cost:
        cost = []
        cost_temp = costfunction_LS(Sigma_tilde, w_theta, Psi_tilde)
        cost.append(cost_temp)

    # Manifold
    CC = ComplexCircle(p)

    err = 1
    it = 1
    alpha_0 = 1e-2
    beta_0 = alpha_0
    while err > 1e-4 and it<maxIter:
        # Euclidean gradient
        if auto:
            grad_euc_jax = grad_euc_LS_func(Sigma_tilde_jnp, w_theta_jnp, Psi_tilde_jnp)[0].conj()
            grad_euc = np.array(grad_euc_jax)
        else:
            grad_euc = grad_costfunction_LS(Sigma_tilde, w_theta)

        # grad euclidien to grad riemannien
        grad_riemann = CC.projection(w_theta,grad_euc)

        # option : conjugate
        if conjugate:
            Xi = -alpha_0*grad_riemann - beta_0 * CC.transport(w_theta, w_theta_prev, grad_riemann_prev)
        else:
            Xi = -alpha_0*grad_riemann

        # retraction
        w_theta_new = CC.retraction(w_theta,Xi)

        # error
        err = la.norm(w_theta_new-w_theta)

        # update
        if conjugate:
            w_theta_prev = w_theta
            grad_riemann_prev = grad_riemann

        w_theta = w_theta_new
        w_theta_jnp = jnp.array(w_theta)
        it = it +1

        # cost function
        if comput_cost:
            cost_temp = costfunction_LS(Sigma_tilde, w_theta, Psi_tilde)
            cost.append(cost_temp)

    if comput_cost:
        cost=cost-min(cost)
        return(w_theta,cost)
    else:
        return(w_theta)

def MM_KL_IPL(Sigma_tilde,maxIter,comput_cost):
    """ MM KL cost function
    Input :
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * maxIter : maximum iteration of the gradient descent
        * comput_cost : True : cost function is computed
    Output : 
        * w : complex vector of phasis
        * cost : cost function if comput_cost = True
    """ 
        
    p = Sigma_tilde.shape[0]
    Psi_tilde = np.abs(Sigma_tilde)

    M = np.multiply(la.inv(abs(Sigma_tilde)),Sigma_tilde)
    _, D, _ = la.svd(M)
    lambdamax = D[0]
    lambdaI_minus = lambdamax*(np.eye(p)) - M
    
    w = np.ones((p,1))
    if comput_cost:
        cost = []
        cost_temp = costfunction_KL(Sigma_tilde, w, Psi_tilde)
        cost.append(cost_temp)

    for i in range (maxIter):
        tilde_w = lambdaI_minus@w 
        w = np.exp(1j*np.angle(tilde_w))
        # cost function
        if comput_cost:
            cost_temp = costfunction_KL(Sigma_tilde, w, Psi_tilde)
            cost.append(cost_temp)

    if comput_cost:
        cost=cost-min(cost)
        return(w,cost)
    else:
        return(w)

def MM_LS_IPL(Sigma_tilde,maxIter,comput_cost):
    """ MM LS cost function
    Input :
        * Sigma_tilde : covariance estimation (possibily after a regularization step)
        * maxIter : maximum iteration of the gradient descent
        * comput_cost : True : cost function is computed
    Output : 
        * w : complex vector of phasis
        * cost : cost function if comput_cost = True
    """ 
    p = Sigma_tilde.shape[0]
    Psi_tilde = np.abs(Sigma_tilde)

    M = np.multiply(abs(Sigma_tilde),Sigma_tilde)
    
    w = np.ones((p,1))
    if comput_cost:
        cost = []
        cost_temp = costfunction_LS(Sigma_tilde, w, Psi_tilde)
        cost.append(cost_temp)
    
    for i in range (maxIter):
        tilde_w = M@w 
        w = np.exp(1j*np.angle(tilde_w))
        # cost function
        if comput_cost:
            cost_temp = costfunction_LS(Sigma_tilde, w, Psi_tilde)
            cost.append(cost_temp)

    if comput_cost:
        cost=cost-min(cost)
        return(w,cost)
    else:
        return(w)

def MLE(X,model,param_cov,rank,args):
    """ MLE for phase and coherence estimation
    Inputs
    ----------
        * X : Input dataset with dimension NxL (N: number of acquisition, L: number of px within a resolution cell)
        * model : distribution of dataset- "Gauss" or "ScaledGauss"
        * param_cov : decomposition of covariance matrix - "Mod" or "Cor"
        * rank : truncated rank for Sigma (interger number < N)
        * args : number of iterations for BCD and MM algorithm

    Outputs
    -------
        * phase, Sigma and cost function
    """
    (N,L) = X.shape
    (maxiterBCD, maxiterMM,phasecorrectionchoice) = args
    SCM = np.dot(X,X.conj().T)/L
    likelihood = []
    Cov = SCM
    
    # In[ ]:Initialize phase as 0
    # w = np.ones((N,1))
    # diag_w = np.diag(w.squeeze())
    
    # In[ ]:Initialize phase with original PL
    
    w = MM_KL_IPL(SCM, 10)
    diag_w = np.diag(w.squeeze())
    
    # In[ ]: Calculate tau and SCM
    for i in range(maxiterBCD):
        if model == 'Gauss':
            SCMtilde = SCM
            tau = np.ones((N,1))
        elif model == 'ScaledGauss':
            tau = np.diagonal(X.conj().T@np.linalg.inv(Cov)@X)/N
            Xnorm = X / np.sqrt(tau)
            SCMtilde = (1/L) * Xnorm@Xnorm.conj().T
        else:
            raise KeyboardInterrupt
            print('Define wrong input model!')
     # In[ ]: EVD low rank case
        if param_cov =='Mod-Arg': # Cao formulation
            # Sigma = abs(SCMtilde)
            Sigma = abs(((diag_w.conj().T).dot(SCMtilde)).dot(diag_w))

            if rank < N:
                u,s,vh = np.linalg.svd(Sigma)
                u_signal = u[:,:rank]
                u_noise = u[:,rank:]
                sigma = np.mean(s[rank:])
                Sigma = u_signal @ np.diag(s[:rank])@u_signal.conj().T + sigma * u_noise@u_noise.conj().T
                
        elif  param_cov =='Cor-Arg': # IGARSS22 formulation
            Sigma = (((diag_w.conj().T).dot(SCMtilde)).dot(diag_w)).real
            if rank < N:
                u,s,vh = np.linalg.svd(Sigma)
                u_signal = u[:,:rank]
                u_noise = u[:,rank:]
                sigma = np.mean(s[rank:])
                Sigma = u_signal @ np.diag(s[:rank])@u_signal.conj().T + sigma*u_noise@u_noise.conj().T
                
        # Cov =  (diag_w.dot(Sigma)).dot(diag_w.conj().T)
        # lik_ty =(np.log(np.linalg.det(Cov))+ np.diag(np.linalg.inv(Cov).dot(SCMtilde)).sum() + N*np.sum(np.log(tau))/L).real 
        # likelihood.append(lik_ty)

        A = np.multiply(la.inv(Sigma),SCMtilde)

        _, D, _ = la.svd(A)
        lambdamax = D[0]
        lambdaI_minus = lambdamax*(np.eye(N)) - A

        # evalcost_MM = []
        
        for j in range (maxiterMM):
            # evalcost_MM.append((np.einsum('ji,jk,ki->i', np.conj(w),A,w)).real)
            tilde_w = lambdaI_minus.dot(w) 
            w = np.exp(1j*np.angle(tilde_w))
            diag_w = np.diag(w.squeeze())
            Cov = (diag_w.dot(Sigma)).dot(diag_w.conj().T)
            # lik_ty =(np.log(np.linalg.det(Cov))+ np.diag(np.linalg.inv(Cov).dot(SCMtilde)).sum() + N*np.sum(np.log(tau)/L)).real 
            # likelihood.append(lik_ty)

    if  phasecorrectionchoice == 3:
        newphase = phasecorrection3(Cov)
        return w, newphase
    elif phasecorrectionchoice == 4:
        Sigma= (((diag_w.conj().T).dot(SCMtilde)).dot(diag_w)).real
        newphase = phasecorrection4(Cov,Sigma)
        return w, newphase
    
def phasecorrection3(covmatrix):
    phase = -np.angle(covmatrix[0,:])
    phase2 = phase-phase[0]
    return phase2

def phasecorrection4(covmatrix,sigmamatrix):
    subdiagphase = np.zeros((sigmamatrix.shape[0]-1))
    phase = np.zeros((sigmamatrix.shape[0]))
    for i in range (sigmamatrix.shape[0]-1):
        subdiagphase[i] = ((np.angle(covmatrix[i,i+1])) +np.pi)%(2*np.pi)-np.pi
        phase[i+1] = (phase[i]+subdiagphase[i]+np.pi)%(2*np.pi)-np.pi
    return -phase