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
from src.optimization import MM_KL_IPL, MM_LS_IPL, RG_comet_IPL

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
    
def oneMC_regul(trial_no,trueCov,trueTheta,p,n,nu,estimator,regul,rank,alpha,b):
    """ one iteration MC for comparing all distances for one estimator and one regularization
    Input :
        * trueCov : true InSAR covariance estimation
        * trueTheta : true theta
        * p : data size
        * n : number of samples
        * nu : scale parameter of the K-distributed data (if 0 Gaussian data are simulated)
        * estimator : 'SCM', 'Tyler' or 'PO' 
        * regul : chosen regularization 'SK', 'LR', 'BW' or None
        * rank, alpha, b : rank, regularization parameter and bandwith parameter
    Output : 
        * error_KL, error_LS, error_WLS, error_Corr: angle error for KL, LS, WLS and 2-p InSAR
    """ 
    np.random.seed(trial_no)

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

    theta_temp = MM_KL_IPL(Sigma_tilde,100,False)
    delta_theta_KL = np.angle(theta_temp)-np.angle(theta_temp[0])

    theta_temp = MM_LS_IPL(Sigma_tilde,100,False)
    delta_theta_LS = np.angle(theta_temp)-np.angle(theta_temp[0])
    
    theta_temp = RG_comet_IPL(Sigma_tilde,1000,False,False,True)
    delta_theta_WLS = np.angle(theta_temp)-np.angle(theta_temp[0])

    delta_theta_Corr = np.zeros((p,1))
    # In correlogram
    for kk in np.arange(1,p):
        delta_theta_Corr[kk] = (-np.angle(np.dot(X[0,:],np.transpose(np.conj(X[kk,:])))))

    error_KL = np.abs((delta_theta_KL - trueTheta))
    error_LS = np.abs((delta_theta_LS - trueTheta))
    error_WLS = np.abs((delta_theta_WLS - trueTheta))
    error_Corr = np.abs((delta_theta_Corr - trueTheta))
    
            
    return [error_KL, error_LS, error_WLS, error_Corr]

def parallel_monte_carlo(trueCov,trueTheta,p,n,nu,estimator,regul,rank,alpha,b,number_of_threads,number_of_trials,Multi):
    """ Fonction for computing MSE - mutliple parallel (or not) monte carlo
        for comparing all distances for one estimator and one regularization
        ----------------------------------------------------------------------------------
    Input :
        * trueCov : true InSAR covariance estimation
        * trueTheta : true theta
        * p : data size
        * n : number of samples
        * nu : scale parameter of the K-distributed data (if 0 Gaussian data are simulated)
        * estimator : 'SCM', 'Tyler' or 'PO' 
        * regul : chosen regularization 'SK', 'LR', 'BW' or None
        * rank, alpha, b : rank, regularization parameter and bandwith parameter
        * number_of_threads : number of threads
        * number_of_trials : number of monte carlo
        * multi : True for parallel computing, False for sequential
    Output : 
        * MSE_KL, MSE_LS, MSE_WLS, error_Corr: MSE for KL, LS, WLS and 2-p InSAR
    """

    if Multi:
        results_parallel = Parallel(n_jobs=number_of_threads)(delayed(oneMC_regul)(iMC,trueCov,trueTheta,p,n,nu,estimator,regul,rank,alpha,b) for iMC in range(number_of_trials))
        results_parallel = np.array(results_parallel)
        MSE_KL = np.mean(results_parallel[:,0], axis=0)
        MSE_LS = np.mean(results_parallel[:,1], axis=0)
        MSE_WLS = np.mean(results_parallel[:,2], axis=0)
        MSE_Corr = np.mean(results_parallel[:,3], axis=0)
        return MSE_KL, MSE_LS, MSE_WLS, MSE_Corr
    else:
        # Results container
        results = []
        for iMC in range(number_of_trials):
            results.append(oneMC_regul(iMC,trueCov,trueTheta,p,n,nu,estimator,regul,rank,alpha,b))

        results = np.array(results)
        MSE_KL = np.mean(results[:,0], axis=0)
        MSE_LS = np.mean(results[:,1], axis=0)
        MSE_WLS = np.mean(results[:,2], axis=0)
        MSE_Corr = np.mean(results[:,3], axis=0)
        return MSE_KL, MSE_LS, MSE_WLS, MSE_Corr

# Main program
if __name__ == "__main__":

    # Paremeters setting
    number_of_threads = -1
    Multi = True
    Latex = False

    p = 10 # data size
    rho = 0.7 # correlation coefficient
    nu = 0 # scale parameter of K-distributed distribution (0 if Gaussian)
    b = 3 # bandwidth parameter
    alpha = 0.8 # coefficient regularization
    rank = 1 # rank of the covariance matrix (p if full-rank)
    maxphase = 2 # in radian
    phasechoice='linear',maxphase 
    estimator = 'SCM' # estimator : 'SCM', 'Tyler' or 'PO' 
    regul = False # regularization: False, LR, SK, BW 

    nMC = 1000 # number of Monte Carlo trials
    vec_L = np.arange(3*p,6*p,3) # number of samples

    # true covariance
    delta_thetasim = phasegeneration(phasechoice,p) #generate phase with either random or linear. for linear, define last phase is needed
    SigmaTrue = La.toeplitz(rho**np.arange(p))
    trueCov = simulate_Covariance(SigmaTrue, delta_thetasim)

    ###############################
    # MC evaluation
    ###############################
    t_beginning = time.time()

    # Distance containers
    MSE_KL = np.zeros((len(vec_L),p))
    MSE_LS = np.zeros((len(vec_L),p))
    MSE_WLS = np.zeros((len(vec_L),p))
    MSE_Corr = np.zeros((len(vec_L),p))
    Temp_MSE_KL = np.zeros(p)
    Temp_MSE_LS = np.zeros(p)
    Temp_MSE_WLS = np.zeros(p)
    Temp_MSE_Corr = np.zeros(p)

    for i_n, n in enumerate(tqdm(vec_L)):
        Temp_MSE_KL,Temp_MSE_LS,Temp_MSE_WLS,Temp_MSE_Corr = \
            np.array(parallel_monte_carlo(
                trueCov,np.transpose(delta_thetasim[np.newaxis,:]),p,n,nu,estimator,regul,rank,alpha,b,number_of_threads,nMC,Multi
            ))
        MSE_KL[i_n,:] = np.squeeze(Temp_MSE_KL)
        MSE_LS[i_n,:] = np.squeeze(Temp_MSE_LS)
        MSE_WLS[i_n,:] = np.squeeze(Temp_MSE_WLS)
        MSE_Corr[i_n,:] = np.squeeze(Temp_MSE_Corr)

    print('Done in %f s'%(time.time()-t_beginning))

    # plot MSE
    for n in range (1,p):

        fig = plt.figure()
        plt.xlabel('n')
        plt.ylabel('MSE')
        plt.plot(vec_L, MSE_KL[:,n],'y', label = 'KL')
        plt.plot(vec_L, MSE_LS[:,n],'b', label = 'LS')
        plt.plot(vec_L, MSE_WLS[:,n],'k', label = 'WLS')
        #plt.plot(vec_L, MSE_Corr[:,n],'o-r', label = '2-p InSAR')
        plt.legend()
        plt.grid("True")
        if regul == False:
            plt.title('At date '+str(n+1)+', rho='+str(rho)+' with p='+str(p)+'. Estimator:'+estimator+' with no regularization')
        else:
            plt.title('At date '+str(n+1)+', rho='+str(rho)+' with p='+str(p)+'. Estimator:'+estimator+' with regularization:'+regul)

        if Latex:
            tikzplotlib_fix_ncols(fig)
            if regul == False:
                tikzplotlib.save(('MSE_fDist_date'+str(n+1)+'_rho'+str(rho)+'_p'+str(p)+'_Est'+estimator+'_RegulNo'+'.tex'))
            else:
                tikzplotlib.save(('MSE_fDist_date'+str(n+1)+'_rho'+str(rho)+'_p'+str(p)+'_Est'+estimator+'_Regul'+regul+'.tex'))

        plt.show()
