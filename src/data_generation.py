# File containing functions to generate InSAR data

import numpy as np
import numpy.linalg as la
import numpy.random as random
import scipy.linalg as La

def simulate_LRCovariance(Sigma, rank):
    """ A function that generates a low-rank Covariance
        Inputs:
            * Sigma full rank covariance matrix
            * rank : rank

        Outputs:
            * low-rank covariance
    """
    u,s,vh = la.svd(Sigma)
    u_signal = u[:,:rank]
    u_noise = u[:,rank:]
    sigma = np.mean(s[rank:])
    return u_signal @ np.diag(s[:rank])@u_signal.conj().T + sigma*u_noise@u_noise.conj().T

def simulate_Covariance(truesigma, truetheta):
    """ A function that generates a Covariance respecting the InSAR structure
        Inputs:
            * truesigma : coherence matrix
            * truetheta : complex vector of phasis

        Outputs:
            * InSAR covariance matrix
    """
    diag_theta = np.diag(np.exp(np.dot(1j,truetheta)))
    truecov = (diag_theta.dot(truesigma).dot(diag_theta.conj().T))
    return truecov

def simulate_gaussiandata(covariance, N,L):
    """ A function that generates complex Gaussian data
        Inputs:
            * covariance : covariance matrix of the gaussian data
            * N, L : data size and number of secondary data

        Outputs:
            * complex gaussian data
    """
    Csqrt = La.sqrtm(covariance)
    X = np.dot(Csqrt*np.sqrt(1/2),(random.randn(N,L) +1j*random.randn(N,L)))
    return X

def simulate_scaledgaussiandata(covariance, nu, N,L):
    """ A function that generates complex K-distributed data
        Inputs:
            * covariance : covariance matrix of the gaussian data
            * nu : scale paramater of the gamma distribution
            * N, L : data size and number of secondary data

        Outputs:
            * complex K-distributed data
    """
    Csqrt = La.sqrtm(covariance)
    X = np.dot(Csqrt*np.sqrt(1/2),(random.randn(N,L) +1j*random.randn(N,L)))

    tau = random.gamma(nu,1/nu, L )
    tau_mat = np.tile(tau,(N,1))
    # Y = (X)* np.sqrt(tau_mat)
    Y = np.multiply(X,np.sqrt(tau_mat))
    return Y

def phasegeneration(choice,N):
    """ A function that generates a complex vector of phasis
        Inputs:
            * choice : random or linear
            * N : data size

        Outputs:
            * complex K-distributed data
    """
    if choice == 'random':
        theta_sim = np.array([random.uniform(-np.pi,np.pi) for i in range(N)])
        delta_thetasim0 = np.array((theta_sim-theta_sim[0]))
        delta_thetasim = np.angle(np.exp(1j*delta_thetasim0))
    elif choice[0]  == 'linear':
        thetastep = choice[1]
        delta_thetasim = np.linspace(0,thetastep,N)
    return delta_thetasim
