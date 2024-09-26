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

def oneMC_estimators(trial_no,trueCov,trueTheta,p,n,nu,regul,cost_function,rank,alpha,b):
    """ one iteration MC for comparing all estimators for one regularization and one cost function
    Input :
        * trueCov : true InSAR covariance estimation
        * trueTheta : true theta
        * p : data size
        * n : number of samples
        * nu : scale parameter of the K-distributed data (if 0 Gaussian data are simulated)
        * regul : chosen regularization 'SK', 'LR', 'BW' or None
        * cost_function : chose cost function 'LS', 'WLS', 'KL'
        * rank, alpha, b : rank, regularization parameter and bandwith parameter
    Output : 
        * delta_theta_SCM, delta_theta_Tyler, delta_theta_PO, delta_theta_Corr: angle error for SCM, Tyler, PO and 2-p InSAR
    """ 
    np.random.seed(trial_no)

    if nu == 0: 
        X = simulate_gaussiandata(trueCov, p,n)
    else:
        X = simulate_scaledgaussiandata(trueCov, nu, p,n)

    if regul == False:
        Sigma_tilde_SCM = SCM(X)
        Sigma_tilde_PO = corr_phase(X)
        Sigma_tilde_Tyler = tyler_estimator_covariance(X, tol=0.001, iter_max=20, return_tau=False)

    if regul == 'SK':
        Sigma_SCM = SCM(X)
        Sigma_PO = corr_phase(X)
        Sigma_Tyler = tyler_estimator_covariance(X, tol=0.001, iter_max=20, return_tau=False)
        Sigma_tilde_SCM = regul_linear(Sigma_SCM,alpha)
        Sigma_tilde_PO = regul_linear(Sigma_PO,alpha)
        Sigma_tilde_Tyler = regul_linear(Sigma_Tyler,alpha)

    if regul == 'BW':
        Sigma_SCM = SCM(X)
        Sigma_PO = corr_phase(X)
        Sigma_Tyler = tyler_estimator_covariance(X, tol=0.001, iter_max=20, return_tau=False)
        Sigma_tilde_SCM = bandw(Sigma_SCM,b)
        Sigma_tilde_PO = bandw(Sigma_PO,b)
        Sigma_tilde_Tyler = bandw(Sigma_Tyler,b)

    if regul == 'LR':
        Sigma_tilde_SCM = SCM_LR(X,rank)
        Sigma_tilde_PO = corr_phase_LR(X,rank)
        Sigma_tilde_Tyler = tyler_estimator_covariance_LR(X, tol=0.001, iter_max=20, return_tau=False,r=rank)


    if cost_function == 'LS':
        theta_temp = MM_LS_IPL(Sigma_tilde_SCM,100,False)
        delta_theta_SCM = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = MM_LS_IPL(Sigma_tilde_PO,100,False)
        delta_theta_PO = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = MM_LS_IPL(Sigma_tilde_Tyler,100,False)
        delta_theta_Tyler = np.angle(theta_temp)-np.angle(theta_temp[0])
        
    if cost_function == 'KL':
        theta_temp = MM_KL_IPL(Sigma_tilde_SCM,100,False)
        delta_theta_SCM = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = MM_KL_IPL(Sigma_tilde_PO,100,False)
        delta_theta_PO = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = MM_KL_IPL(Sigma_tilde_Tyler,100,False)
        delta_theta_Tyler = np.angle(theta_temp)-np.angle(theta_temp[0])

    if cost_function == 'WLS':
        theta_temp = RG_comet_IPL(Sigma_tilde_SCM,1000,False,False,True)
        delta_theta_SCM = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = RG_comet_IPL(Sigma_tilde_PO,1000,False,False,True)
        delta_theta_PO = np.angle(theta_temp)-np.angle(theta_temp[0])
        theta_temp = RG_comet_IPL(Sigma_tilde_Tyler,1000,False,False,True)
        delta_theta_Tyler = np.angle(theta_temp)-np.angle(theta_temp[0])

    delta_theta_Corr = np.zeros((p,1))
    # In correlogram
    for kk in np.arange(1,p):
        delta_theta_Corr[kk] = (-np.angle(np.dot(X[0,:],np.transpose(np.conj(X[kk,:])))))

    error_SCM = np.abs((delta_theta_SCM - trueTheta))
    error_PO = np.abs((delta_theta_PO - trueTheta))
    error_Tyler = np.abs((delta_theta_Tyler - trueTheta))
    error_Corr = np.abs((delta_theta_Corr - trueTheta))
            
    return [error_SCM, error_PO, error_Tyler, error_Corr]

def parallel_monte_carlo(trueCov,trueTheta,p,n,nu,regul,cost_function,rank,alpha,b,number_of_threads,number_of_trials,Multi):
    """ Fonction for computing MSE - mutliple parallel (or not) monte carlo
        for comparing all estimators for one regularization and one cost function
        ----------------------------------------------------------------------------------
    Input :
        * trueCov : true InSAR covariance estimation
        * trueTheta : true theta
        * p : data size
        * n : number of samples
        * nu : scale parameter of the K-distributed data (if 0 Gaussian data are simulated)
        * regul : chosen regularization 'SK', 'LR', 'BW' or None
        * cost_function : chose cost function 'LS', 'WLS', 'KL'
        * rank, alpha, b : rank, regularization parameter and bandwith parameter
        * number_of_threads : number of threads
        * number_of_trials : number of monte carlo
        * multi : True for parallel computing, False for sequential
    Output : 
        * MSE_SCM, MSE_Tyler, MSE_PO, MSE_Corr: MSE for SCM, Tyler, PO and 2-p InSAR
    """

    if Multi:
        results_parallel = Parallel(n_jobs=number_of_threads)(delayed(oneMC_estimators)(iMC,trueCov,trueTheta,p,n,nu,regul,cost_function,rank,alpha,b) for iMC in range(number_of_trials))
        results_parallel = np.array(results_parallel)
        MSE_SCM = np.mean(results_parallel[:,0], axis=0)
        MSE_PO = np.mean(results_parallel[:,1], axis=0)
        MSE_Tyler = np.mean(results_parallel[:,2], axis=0)
        MSE_Corr = np.mean(results_parallel[:,3], axis=0)
        return MSE_SCM, MSE_PO, MSE_Tyler, MSE_Corr
    else:
        # Results container
        results = []
        for iMC in range(number_of_trials):
            results.append(oneMC_estimators(iMC,trueCov,trueTheta,p,n,nu,regul,cost_function,rank,alpha,b))

        results = np.array(results)
        MSE_SCM = np.mean(results[:,0], axis=0)
        MSE_PO = np.mean(results[:,1], axis=0)
        MSE_Tyler = np.mean(results[:,2], axis=0)
        MSE_Corr = np.mean(results[:,3], axis=0)
        return MSE_SCM, MSE_PO, MSE_Tyler, MSE_Corr

# Main program
if __name__ == "__main__":

    # Paremeters setting
    number_of_threads = -1
    Multi = True
    Latex = False

    p = 10 # data size
    rho = 0.7 # correlation coefficient
    nu = 0 # scale parameter of K-distributed distribution (0 if Gaussian)
    b = 9 # bandwidth parameter
    alpha = 0.8 # coefficient regularization
    rank = p # rank of the covariance matrix (p if full-rank)
    maxphase = 2 # in radian
    phasechoice='linear',maxphase 
    cost = 'LS' # cost function: LS, KL or WLS
    regul = False # regularization: False, LR, SK, BW 

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
    MSE_SCM = np.zeros((len(vec_L),p))
    MSE_PO = np.zeros((len(vec_L),p))
    MSE_Tyler = np.zeros((len(vec_L),p))
    MSE_Corr = np.zeros((len(vec_L),p))
    Temp_MSE_SCM = np.zeros(p)
    Temp_MSE_PO = np.zeros(p)
    Temp_MSE_Tyler = np.zeros(p)
    Temp_MSE_Corr = np.zeros(p)

    for i_n, n in enumerate(tqdm(vec_L)):
        Temp_MSE_SCM,Temp_MSE_PO,Temp_MSE_Tyler,Temp_MSE_Corr = \
            np.array(parallel_monte_carlo(
                trueCov,np.transpose(delta_thetasim[np.newaxis,:]),p,n,nu,regul,cost,rank,alpha,b,number_of_threads,nMC,Multi
            ))
        MSE_SCM[i_n,:] = np.squeeze(Temp_MSE_SCM)
        MSE_PO[i_n,:] = np.squeeze(Temp_MSE_PO)
        MSE_Tyler[i_n,:] = np.squeeze(Temp_MSE_Tyler)
        MSE_Corr[i_n,:] = np.squeeze(Temp_MSE_Corr)

    print('Done in %f s'%(time.time()-t_beginning))

    # plot MSE
    for n in range (1,p):

        fig = plt.figure()
        plt.xlabel('n')
        plt.ylabel('MSE')
        plt.plot(vec_L, MSE_SCM[:,n],'y', label = 'SCM')
        plt.plot(vec_L, MSE_Tyler[:,n],'b', label = 'Tyl')
        plt.plot(vec_L, MSE_PO[:,n], 'k',label = 'PO')
        #plt.plot(vec_L, MSE_Corr[:,n],'o-r', label = '2-p InSAR')
        plt.legend()
        plt.grid("True")
        if regul == False:
            plt.title('At date '+str(n+1)+', rho='+str(rho)+' with p='+str(p)+'. Distance:'+cost+' with no regularization')
        else:
            plt.title('At date '+str(n+1)+', rho='+str(rho)+' with p='+str(p)+'. Distance:'+cost+' with regularization:'+regul)

        if Latex:
            tikzplotlib_fix_ncols(fig)
            if regul == False:
                tikzplotlib.save(('MSE_fEstim_date'+str(n+1)+'_rho'+str(rho)+'_p'+str(p)+'_Dist'+cost+'_RegulNo'+'.tex'))
            else:
                tikzplotlib.save(('MSE_fEstim_date'+str(n+1)+'_rho'+str(rho)+'_p'+str(p)+'_Dist'+cost+'_Regul'+regul+'.tex'))

        plt.show()
