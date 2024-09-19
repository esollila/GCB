#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Greedy Capon Beamformer 

REFERENCE:
    Esa Ollila, "Greedy Capon Beamformer", ArXiv preprint, 
    arXiv:2404.15329 [eess.SP], 2024. 
    
@author: esollila
"""
import numpy as np
from scipy.linalg import solve 
from scipy.signal import find_peaks

#%%
def peaks_1D(gamma, Nsources):
    """ 
    peaks_1d(gamma,Nsources)
    fast method to find peaks
    """
    pks    = np.zeros((Nsources))
    locs   = np.zeros((Nsources),dtype = int)
    Ntheta = len(gamma)
    gamma  = gamma.reshape(Ntheta)
    gamma_new = np.zeros((Ntheta+2)) # zero padding on the boundary
    gamma_new[1:Ntheta+1] = gamma
    Ilocs  = np.flip(gamma.argsort(axis = 0))
    npeaks = 0         # current number of peaks found
    local_patch=np.zeros((Nsources))
    for ii in range(Ntheta):
        # local patch area surrounding the current array entry i.e. (r,c)
        # local_patch = gamma_new[(Ilocs[ii]):(Ilocs[ii]+3)];
        # zero the center
        # local_patch[1] = 0;
        local_patch = [gamma_new[(Ilocs[ii])], 0, gamma_new[(Ilocs[ii]+2)]];
        # zero the center
        if sum(gamma[Ilocs[ii]] > local_patch) == 3:
            pks[npeaks] = gamma[Ilocs[ii]];
            locs[npeaks] = Ilocs[ii];
            npeaks = npeaks + 1;
            # if found sufficient peaks, break
            if npeaks == Nsources:
                break;

    return pks, locs

def SCB(A,RY,K,phi_vec):
    """ 
    DOA,pks = SCB(A,RY,K,phi_vec)
    Standard Capon Beamformer (GCB) algorithm 

      INPUT: 
          A       - Steering matrix of M steering vectors, matrix of size: N x M
          RY      - Sample Covariance Matrix (SCM), matrix of size N x N
          K       - the number of sources, a positive integer
          phi_vec - DOA grid in degrees, so elements are in [-90,90), a vector 
                  of length M

      OUTPUT:
          DOA    -  K-vector of estimated DOAs in degrees (ordered: DOA(1) <.)
          pks    -  peak indices (K vector that is subset of {1,...,M})
    """
    
    invRY = solve(RY, np.eye(RY.shape[0]),assume_a='pos') 
    power = 10*np.log10(1/np.real(np.sum(A.conj().T @ invRY *(A.T),axis=1))) 
    locs,_ = find_peaks(power)
    locs = sorted(locs, key=lambda idx: power[idx], reverse=True)
    pks = locs[:K]
    DOA = np.sort(phi_vec[pks])  
    
    return DOA, pks  
#%%    
def GCB(A,RY,K,phi_vec):
    """
    DOA,pks = GCB(A,RY,K,phi_vec)
    Greedy Capon Beamformer (GCB) algorithm proposed in Ollila (2024).

      INPUT: 
          A       - Steering matrix of M steering vectors, matrix of size: N x M
          RY      - Sample Covariance Matrix (SCM), matrix of size N x N
          K       - the number of sources, a positive integer
          phi_vec - DOA grid in degrees, so elements are in [-90,90), a vector 
                  of length M

      OUTPUT:
          DOA    -  K-vector of estimated DOAs in degrees (ordered: DOA(1) <.)
          pks    -  peak indices (K vector that is subset of {1,...,M})

    REFERENCE:
        Esa Ollila, "Greedy Capon Beamformer", ArXiv preprint, 
        arXiv:2404.15329 [eess.SP], 2024. 
        
    Author: Esa Ollila, Aalto University, 2024. 
    """

    N,M=A.shape # number of sensors and dictionary entries

    ## Initialize
    sigc = np.trace(RY) / N
    SigmaYinv = (1/sigc)*np.eye(N)
    Ilocs = np.zeros(K, dtype = int)
    gam = np.zeros(K)


    for k in range(K-1):
      
        ## 1. Calculate the powers 
        B =  SigmaYinv@A # Sigma^-1 a_m , m=1,.., M 
        AB = RY @ B
        P_num = np.maximum(0, np.real(np.sum(np.conj(B) * AB, axis=0)))
        P_denum = np.maximum(0,np.real(np.sum(np.conj(A) * B,axis = 0)))
        P_denum[P_denum <= 1e-18] = 1e-18 # make sure not zero
        P = P_num/P_denum**2   

        ## 2. find k largest peaks and 3. pick index with least coherence

        if k==0:
            indx = np.argmax(P)
        else: 
            _ , locs = peaks_1D(P,k+1)
            coh_mat = np.abs(A[:, locs].conj().T @ A[:, Ilocs[:k]])
            indx = locs[np.argmin(np.max(coh_mat, axis=1))]
            
        ## 4. Update chosen indices

        Ilocs[k] = indx

        ## 5. Estimate the signal power  
        gam[k] = np.maximum(0, P[indx] - 1/P_denum[indx])
        if k == 0:
            gam[k] *= N / (N-1)

        
        ## 6. Update the INCM 
        b = SigmaYinv @ A[:, indx].reshape(-1,1)
        SigmaYinv -= (gam[k] / (1 + gam[k] * P_denum[indx])) * (b @ b.conj().T)

    ## 1. Calculate the powers 
    B = SigmaYinv @ A  # Sigma^-1 a_m , m=1,..,M
    AB = RY @ B
    P_num = np.maximum(0, np.real(np.sum(np.conj(B) * AB, axis=0)))
    P_denum = np.maximum(0,np.real(np.sum(np.conj(A) * B,axis = 0)))
    P_denum[P_denum <= 1e-18] = 1e-18 # make sure not zero
    P = P_num/P_denum**2   
         
    if K==1:
        _ , locs = np.argmax(P)
    else:
        _ ,locs = peaks_1D(P,K)
        idx = np.argsort(locs)
        locs = locs[idx]
        
    DOA = phi_vec[locs]
    
    return DOA,locs
