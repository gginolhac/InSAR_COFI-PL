import numpy as np
import scipy.linalg as La
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import tikzplotlib

import sys
import os
# adding lib to the system path
sys.path.insert(0,
    os.path.join(os.path.dirname(__file__), '../'))

from src.data_generation import simulate_Covariance, phasegeneration, \
    simulate_gaussiandata, simulate_scaledgaussiandata
from src.covariance_estimators import SCM, regul_linear, bandw, SCM_LR, \
    tyler_estimator_covariance, tyler_estimator_covariance_LR, corr_phase, corr_phase_LR
from src.optimization import MM_KL_IPL, MM_LS_IPL, RG_comet_IPL, RG_KL_IPL, RG_LS_IPL

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

# Main program
if __name__ == "__main__":

    # Paremeters setting
    Latex = True

    p = 50 # data size
    rho = 0.7 # correlation coefficient
    nu = 0 # scale parameter of K-distributed distribution (0 if Gaussian)
    b = 9 # bandwidth parameter
    alpha = 0.8 # coefficient regularization
    rank = 1 # rank of the covariance matrix (p if full-rank)
    maxphase = 2 # in radian
    phasechoice='linear',maxphase 
    estimator = 'SCM' # estimator : 'SCM', 'Tyler' or 'PO' 
    regul = False # regularization: False, LR, SK, BW 

    n = 2*p # number of samples

    # true covariance
    delta_thetasim = phasegeneration(phasechoice,p) #generate phase with either random or linear. for linear, define last phase is needed
    SigmaTrue = La.toeplitz(rho**np.arange(p))
    trueCov = simulate_Covariance(SigmaTrue, delta_thetasim)

    ###############################
    # cost function evaluation
    ###############################
    t_beginning = time.time()

    if nu == 0: 
        X = simulate_gaussiandata(trueCov, p,n)
    else:
        X = simulate_scaledgaussiandata(trueCov, nu, p,n)

    if estimator == 'SCM':
        Sigma = SCM(X)
        if regul == 'LR':
            Sigma_tilde = SCM_LR(X,rank)
        if regul == 'SK': 
            Sigma_tilde = regul_linear(Sigma,alpha)
        if regul == 'BW':
            Sigma_tilde = bandw(Sigma,b)
        if regul == False:
            Sigma_tilde = Sigma

    if estimator == 'PO':
        Sigma = corr_phase(X)
        if regul == 'LR':
            Sigma_tilde = corr_phase_LR(X,rank)
        if regul == 'SK': 
            Sigma_tilde = regul_linear(Sigma,alpha)
        if regul == 'BW':
            Sigma_tilde = bandw(Sigma,b)
        if regul == False:
            Sigma_tilde = Sigma

    if estimator == 'Tyler':
        Sigma = tyler_estimator_covariance(X, tol=0.001, iter_max=20, return_tau=False)
        if regul == 'LR':
            Sigma_tilde = tyler_estimator_covariance_LR(X, tol=0.001, iter_max=20, return_tau=False,r=rank)
        if regul == 'SK': 
            Sigma_tilde = regul_linear(Sigma,alpha)
        if regul == 'BW':
            Sigma_tilde = bandw(Sigma,b)
        if regul == False:
            Sigma_tilde = Sigma

    # cost functions
    theta,cost_MM_KL = MM_KL_IPL(Sigma_tilde,500,True)
    theta,cost_RG_KL = RG_KL_IPL(Sigma_tilde,500,False,True,False)
    theta,cost_RCG_KL = RG_KL_IPL(Sigma_tilde,500,False,True,True)
    theta,cost_MM_LS = MM_LS_IPL(Sigma_tilde,500,True)
    theta,cost_RG_LS = RG_LS_IPL(Sigma_tilde,500,False,True,False)
    theta,cost_RCG_LS = RG_LS_IPL(Sigma_tilde,500,False,True,True)
    theta,cost_RG_WLS = RG_comet_IPL(Sigma_tilde,1000,False,True,False)
    theta,cost_RCG_WLS = RG_comet_IPL(Sigma_tilde,1000,False,True,True)

    # figures

    fig = plt.figure()
    plt.xlabel('iterations')
    plt.ylabel('cost functions')
    plt.semilogy(cost_MM_KL,'-b', label = 'MM')
    plt.semilogy(cost_RG_KL,'-k', label = 'RG')
    plt.semilogy(cost_RCG_KL,'-r', label = 'RCG')
    #plt.plot(MSE_Corr[:,n],'o-r', label = '2-p InSAR')
    plt.legend()
    plt.grid("True")
    if regul == False:
        plt.title('KL. rho='+str(rho)+' with p='+str(p)+' n='+str(n)+'. Estimator:'+estimator+' with no regularization')
    else:
        plt.title('KL. rho='+str(rho)+' with p='+str(p)+' n='+str(n)+'. Estimator:'+estimator+' with regularization:'+regul)
    if Latex:
        tikzplotlib_fix_ncols(fig)
        if regul == False:
            tikzplotlib.save(('CostFunctionKL_rho'+str(rho)+'_p'+str(p)+'_Est'+estimator+'_RegulNo'+'.tex'))
        else:
            tikzplotlib.save(('CostFunctionKL_rho'+str(rho)+'_p'+str(p)+'_Est'+estimator+'_Regul'+regul+'.tex'))


    fig = plt.figure()
    plt.xlabel('iterations')
    plt.ylabel('cost functions')
    plt.semilogy(cost_MM_LS,'-b', label = 'MM')
    plt.semilogy(cost_RG_LS,'-k', label = 'RG')
    plt.semilogy(cost_RCG_LS,'-r', label = 'RCG')
    plt.legend()
    plt.grid("True")
    if regul == False:
        plt.title('LS. rho='+str(rho)+' with p='+str(p)+' n='+str(n)+'. Estimator:'+estimator+' with no regularization')
    else:
        plt.title('LS. rho='+str(rho)+' with p='+str(p)+' n='+str(n)+'. Estimator:'+estimator+' with regularization:'+regul)
    if Latex:
        tikzplotlib_fix_ncols(fig)
        if regul == False:
            tikzplotlib.save(('CostFunctionLS_rho'+str(rho)+'_p'+str(p)+'_Est'+estimator+'_RegulNo'+'.tex'))
        else:
            tikzplotlib.save(('CostFunctionLS_rho'+str(rho)+'_p'+str(p)+'_Est'+estimator+'_Regul'+regul+'.tex'))


    fig = plt.figure()
    plt.xlabel('iterations')
    plt.ylabel('cost functions')
    plt.semilogy(cost_RG_WLS,'-k', label = 'RG')
    plt.semilogy(cost_RCG_WLS,'-r', label = 'RCG')
    #plt.plot(MSE_Corr[:,n],'o-r', label = '2-p InSAR')
    plt.legend()
    plt.grid("True")
    if regul == False:
        plt.title('WLS. rho='+str(rho)+' with p='+str(p)+' n='+str(n)+'. Estimator:'+estimator+' with no regularization')
    else:
        plt.title('WLS. rho='+str(rho)+' with p='+str(p)+' n='+str(n)+'. Estimator:'+estimator+' with regularization:'+regul)
    if Latex:
        tikzplotlib_fix_ncols(fig)
        if regul == False:
            tikzplotlib.save(('CostFunctionWLS_rho'+str(rho)+'_p'+str(p)+'_Est'+estimator+'_RegulNo'+'.tex'))
        else:
            tikzplotlib.save(('CostFunctionWLS_rho'+str(rho)+'_p'+str(p)+'_Est'+estimator+'_Regul'+regul+'.tex'))


    
    plt.show()
