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

def oneMC_regul(trial_no,trueCov,trueTheta,p,n,nu,estimator,cost_function,rank,alpha,b):
    """ one iteration MC for comparing all regularization for one estimator and one cost function
    Input :
        * trueCov : true InSAR covariance estimation
        * trueTheta : true theta
        * p : data size
        * n : number of samples
        * nu : scale parameter of the K-distributed data (if 0 Gaussian data are simulated)
        * estimator : 'SCM', 'Tyler' or 'PO' 
        * cost_function : chose cost function 'LS', 'WLS', 'KL'
        * rank, alpha, b : rank, regularization parameter and bandwith parameter
    Output : 
        * error_NoRegul, error_LR, error_SK, error_BW, error_Corr: angle error for NoRegul, LR, SK, BW and 2-p InSAR
    """ 
    np.random.seed(trial_no)

    if nu == 0: 
        X = simulate_gaussiandata(trueCov, p,n)
    else:
        X = simulate_scaledgaussiandata(trueCov, nu, p,n)

    if estimator == 'SCM':
        Sigma_tilde = SCM(X)
        Sigma_tilde_LR = SCM_LR(X,rank)
        Sigma_tilde_SK = regul_linear(Sigma_tilde,alpha)
        Sigma_tilde_BW = bandw(Sigma_tilde,b)

    if estimator == 'PO':
        Sigma_tilde = corr_phase(X)
        Sigma_tilde_LR = corr_phase_LR(X,rank)
        Sigma_tilde_SK = regul_linear(Sigma_tilde,alpha)
        Sigma_tilde_BW = bandw(Sigma_tilde,b)

    if estimator == 'Tyler':
        Sigma_tilde = tyler_estimator_covariance(X, tol=0.001, iter_max=20, return_tau=False)
        Sigma_tilde_LR = tyler_estimator_covariance_LR(X, tol=0.001, iter_max=20, return_tau=False,r=rank)
        Sigma_tilde_SK = regul_linear(Sigma_tilde,alpha)
        Sigma_tilde_BW = bandw(Sigma_tilde,b)

    if cost_function == 'LS':
        theta_temp = MM_LS_IPL(Sigma_tilde,100,False)
        delta_theta_NoRegul = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = MM_LS_IPL(Sigma_tilde_LR,100,False)
        delta_theta_LR = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = MM_LS_IPL(Sigma_tilde_SK,100,False)
        delta_theta_SK = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = MM_LS_IPL(Sigma_tilde_BW,100,False)
        delta_theta_BW = np.angle(theta_temp)-np.angle(theta_temp[0])
        
    if cost_function == 'KL':
        theta_temp = MM_KL_IPL(Sigma_tilde,100,False)
        delta_theta_NoRegul = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = MM_KL_IPL(Sigma_tilde_LR,100,False)
        delta_theta_LR = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = MM_KL_IPL(Sigma_tilde_SK,100,False)
        delta_theta_SK = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = MM_KL_IPL(Sigma_tilde_BW,100,False)
        delta_theta_BW = np.angle(theta_temp)-np.angle(theta_temp[0])

    if cost_function == 'WLS':
        theta_temp = RG_comet_IPL(Sigma_tilde,1000,False,False,True)
        delta_theta_NoRegul = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = RG_comet_IPL(Sigma_tilde_LR,1000,False,False,True)
        delta_theta_LR = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = RG_comet_IPL(Sigma_tilde_SK,1000,False,False,True)
        delta_theta_SK = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = RG_comet_IPL(Sigma_tilde_BW,1000,False,False,True)
        delta_theta_BW = np.angle(theta_temp)-np.angle(theta_temp[0])

    delta_theta_Corr = np.zeros((p,1))
    # In correlogram
    for kk in np.arange(1,p):
        delta_theta_Corr[kk] = (-np.angle(np.dot(X[0,:],np.transpose(np.conj(X[kk,:])))))

    error_NoRegul = np.abs((delta_theta_NoRegul - trueTheta))
    error_LR = np.abs((delta_theta_LR - trueTheta))
    error_SK = np.abs((delta_theta_SK - trueTheta))
    error_BW = np.abs((delta_theta_BW - trueTheta))
    error_Corr = np.abs((delta_theta_Corr - trueTheta))
            
    return [error_NoRegul, error_LR, error_SK, error_BW, error_Corr]

def parallel_monte_carlo(trueCov,trueTheta,p,n,nu,estimator,cost_function,rank,alpha,b,number_of_threads,number_of_trials,Multi):
    """ Fonction for computing MSE - mutliple parallel (or not) monte carlo
        for comparing all regularization for one estimator and one cost function
        ----------------------------------------------------------------------------------
    Input :
        * trueCov : true InSAR covariance estimation
        * trueTheta : true theta
        * p : data size
        * n : number of samples
        * nu : scale parameter of the K-distributed data (if 0 Gaussian data are simulated)
        * estimator : 'SCM', 'Tyler' or 'PO' 
        * cost_function : chose cost function 'LS', 'WLS', 'KL'
        * rank, alpha, b : rank, regularization parameter and bandwith parameter
        * number_of_threads : number of threads
        * number_of_trials : number of monte carlo
        * multi : True for parallel computing, False for sequential
    Output : 
        * MSE_NoRegul, MSE_LR, MSE_SK, MSE_BW, MSE_Corr: MSE for NoRegul, LR, SK, BW and 2-p InSAR
    """

    if Multi:
        results_parallel = Parallel(n_jobs=number_of_threads)(delayed(oneMC_regul)(iMC,trueCov,trueTheta,p,n,nu,estimator,cost_function,rank,alpha,b) for iMC in range(number_of_trials))
        results_parallel = np.array(results_parallel)
        MSE_NoRegul = np.mean(results_parallel[:,0], axis=0)
        MSE_LR = np.mean(results_parallel[:,1], axis=0)
        MSE_SK = np.mean(results_parallel[:,2], axis=0)
        MSE_BW = np.mean(results_parallel[:,3], axis=0)
        MSE_Corr = np.mean(results_parallel[:,4], axis=0)
        return MSE_NoRegul, MSE_LR, MSE_SK, MSE_BW, MSE_Corr
    else:
        # Results container
        results = []
        for iMC in range(number_of_trials):
            results.append(oneMC_regul(iMC,trueCov,trueTheta,p,n,nu,estimator,cost_function,rank,alpha,b))

        results = np.array(results)
        MSE_NoRegul = np.mean(results[:,0], axis=0)
        MSE_LR = np.mean(results[:,1], axis=0)
        MSE_SK = np.mean(results[:,2], axis=0)
        MSE_BW = np.mean(results[:,3], axis=0)
        MSE_Corr = np.mean(results[:,4], axis=0)
        return MSE_NoRegul, MSE_LR, MSE_SK, MSE_BW, MSE_Corr

# Main program
if __name__ == "__main__":

    # Paremeters setting
    number_of_threads = -1
    Multi = True
    Latex = True

    p = 10 # data size
    rho = 0.7 # correlation coefficient
    nu = 0 # scale parameter of K-distributed distribution (0 if Gaussian)
    b = 3 # bandwidth parameter
    alpha = 0.5 # coefficient regularization
    rank = 1 # rank of the covariance matrix (p if full-rank)
    maxphase = 2 # in radian
    phasechoice='linear',maxphase 
    cost = 'KL' # cost function: LS, KL or WLS
    estimator = 'SCM' # estimator : 'SCM', 'Tyler' or 'PO' 

    nMC = 1000 # number of Monte Carlo trials
    vec_L = np.arange(p+1,3*p,3) # number of samples

    # true covariance
    delta_thetasim = phasegeneration(phasechoice,p) #generate phase with either random or linear. for linear, define last phase is needed
    SigmaTrue = La.toeplitz(rho**np.arange(p))
    trueCov = simulate_Covariance(SigmaTrue, delta_thetasim)

    ###############################
    # MC evaluation
    ###############################
    t_beginning = time.time()

    # Distance containers
    MSE_NoRegul = np.zeros((len(vec_L),p))
    MSE_LR = np.zeros((len(vec_L),p))
    MSE_SK = np.zeros((len(vec_L),p))
    MSE_BW = np.zeros((len(vec_L),p))
    MSE_Corr = np.zeros((len(vec_L),p))
    Temp_MSE_NoRegul = np.zeros(p)
    Temp_MSE_LR = np.zeros(p)
    Temp_MSE_SK = np.zeros(p)
    Temp_MSE_BW = np.zeros(p)
    Temp_MSE_Corr = np.zeros(p)

    for i_n, n in enumerate(tqdm(vec_L)):
        Temp_MSE_NoRegul,Temp_MSE_LR,Temp_MSE_SK,Temp_MSE_BW,Temp_MSE_Corr = \
            np.array(parallel_monte_carlo(
                trueCov,np.transpose(delta_thetasim[np.newaxis,:]),p,n,nu,estimator,cost,rank,alpha,b,number_of_threads,nMC,Multi
            ))
        MSE_NoRegul[i_n,:] = np.squeeze(Temp_MSE_NoRegul)
        MSE_LR[i_n,:] = np.squeeze(Temp_MSE_LR)
        MSE_SK[i_n,:] = np.squeeze(Temp_MSE_SK)
        MSE_BW[i_n,:] = np.squeeze(Temp_MSE_BW)
        MSE_Corr[i_n,:] = np.squeeze(Temp_MSE_Corr)

    print('Done in %f s'%(time.time()-t_beginning))

    # plot MSE
    for n in range (1,p):

        fig = plt.figure()
        plt.xlabel('n')
        plt.ylabel('MSE')
        plt.plot(vec_L, MSE_NoRegul[:,n],'y', label = 'No regul')
        plt.plot(vec_L, MSE_LR[:,n],'b', label = 'LR')
        plt.plot(vec_L, MSE_SK[:,n],'k', label = 'SK')
        plt.plot(vec_L, MSE_BW[:,n],'g', label = 'BW')
        #plt.plot(vec_L, MSE_Corr[:,n],'o-r', label = '2-p InSAR')
        plt.legend()
        plt.grid("True")
        plt.title('At date '+str(n+1)+', rho='+str(rho)+' with p='+str(p)+'. Distance:'+cost+' with estimator:'+estimator)

        if Latex:
            tikzplotlib_fix_ncols(fig)
            tikzplotlib.save(('MSE_fRegul_date'+str(n+1)+'_rho'+str(rho)+'_p'+str(p)+'_Dist'+cost+'_Estim'+estimator+'.tex'))

        plt.show()
