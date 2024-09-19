import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import gmean
from GCB_base import GCB, SCB

#%%
def sind(degrees):
    return np.sin(np.deg2rad(degrees))
#%% Model parameters 
N = 20     # nr. of sensors in the ULA
M = 1801   # grid size
L = 125    # number of snapshots available 
K = 4      # number of sources

#%% True DOA-s (all except the 1st one is off the grid)
DOA_src1 = -30.1     
DOA_src2 = -20.02
DOA_src3 = -10.02
DOA_src4 = 3.02
DOA_src = np.array([DOA_src1, DOA_src2,DOA_src3,DOA_src4])
A0 = np.exp(-1j*np.pi*np.arange(N).reshape(-1,1)*sind(DOA_src))  #steering matrix 

#%% Create the DOA grid and the respective steering matrix of size N x M: 
# We assume ULA with half a vawelength spacing
dphi = 180/(M-1)           #  % angular resolution
phi_vec = np.arange(-90,90,dphi)     # grid of DOAs
A = np.exp(-1j*np.pi*np.arange(N).reshape(-1,1)*sind(phi_vec)) # steering matrix for grid DOAs

#%% Compute source powers 
# source 2, 3, and 4 have -1, -2 and -5 dB lower power than source 1

SNR = np.arange(-6.5,-12.5,-0.5) 
sigma_vec1 = np.sqrt(10**(SNR/10))
sigma_vec2 = np.sqrt(10**((SNR-1)/10)) 
sigma_vec3 = np.sqrt(10**((SNR-2)/10)) 
sigma_vec4 = np.sqrt(10**((SNR-5)/10)) 
sigmas = np.vstack([sigma_vec1, sigma_vec2, sigma_vec3, sigma_vec4])


#%% Simulation parameters
LL = 5000  # number of MCtrials -> the paper had 15000  
# Note: simulations in the paper were made using matlab codes 


nSNR = len(sigma_vec1)
DOA1 = np.zeros((LL,K,nSNR),dtype=float)
DOA2 = np.zeros((LL,K,nSNR),dtype=float)
AVEcpu  = np.zeros((nSNR,2),dtype=float)

for isnr in range(nSNR):

    sigma = sigmas[:,isnr]
    print('{} / {} , SNR= {:.3f}\n'.format(isnr+1, nSNR, SNR[isnr]))
    rng = np.random.default_rng(12345)
   
    tim1 = 0
    tim2 = 0 

    for ell in range(LL):
     
        #%% Generate the L snapshots 
        s = np.diag(sigma) @ (rng.standard_normal((K,L))+1j*rng.standard_normal((K,L)))/np.sqrt(2)
        noise = (rng.standard_normal((N,L))+1j*rng.standard_normal((N,L)))/np.sqrt(2)
        y = A0 @ s + noise                  
        RY = (1/L)*y @ y.conj().T  # The sample covariance matrix (SCM)        
        
        #%% 1. GCB
        start_time = time.time()
        DOAgbf,_ = GCB(A,RY,K,phi_vec)
        end_time = time.time()
        tim1 = tim1 +  (end_time - start_time)
        DOA1[ell,:,isnr] = DOAgbf     

        ## Find nearest neighbors
        knn_indices = np.argmin(np.abs(DOAgbf  - DOA_src.reshape(-1,1)),axis=1)
    
        if not(np.setdiff1d(np.arange(K),knn_indices).size==0):
        
            indx = np.arange(K)
            idxset = np.zeros(K,dtype=int)
            for i, point in enumerate(DOA_src):
            
                idx = np.argmin(np.abs(DOAgbf[indx] - point))
                idxset[i] = int(indx[idx])
                indx = np.setdiff1d(indx, idxset[i] )
            
            DOA1[ell,:,isnr] = DOAgbf[idxset]
        
        #%% 2. Standard Capon Beamformer (SCB)
        start_time = time.time()
        est2,_ = SCB(A,RY,K,phi_vec)
        end_time = time.time()
        tim2 = tim2 +  (end_time - start_time)
        DOA2[ell,:,isnr] = est2     
        
        ##  Find nearest neighbors
        knn_indices = np.argmin(np.abs(est2  - DOA_src.reshape(-1,1)),axis=1)
        
        if not(np.setdiff1d(np.arange(K),knn_indices).size==0):
            
           indx = np.arange(K)
           idxset = np.zeros(K,dtype=int)
           for i, point in enumerate(DOA_src):
                
               idx = np.argmin(np.abs(est2[indx] - point))
               idxset[i] = int(indx[idx])
               indx = np.setdiff1d(indx, idxset[i] )
                
           DOA2[ell,:,isnr] = est2[idxset]
     
        #%%
        
        if ell % 500 == 0:
            print('.', end='')  # Print a dot without newline
            
    AVEcpu[isnr,:] = np.array([tim1,tim2])/LL
    print(" Done\n")
    
#%% Compute MSE and RMSE
mse = np.mean((DOA1 - DOA_src[np.newaxis, :, np.newaxis])**2, axis=0)
MSE1 = mse.reshape(DOA1.shape[1:])
rmse1 = np.sqrt(np.sum(MSE1, axis=0))

mse = np.mean((DOA2 -  DOA_src[np.newaxis, :, np.newaxis])**2, axis=0)
MSE2 = mse.reshape(DOA2.shape[1:])
rmse2 = np.sqrt(np.sum(MSE2, axis=0))

#%%  Plotting setup
gm = gmean(sigmas**2)
xvals = 10 * np.log10(gm) # Array SNR =  1/K sum_k log10(sigma_k^2) 
msize = 7 # Set marker size
lwid = 0.8
#%%  Plotting
plt.figure(1)
plt.clf()
plt.semilogy(xvals, rmse1, 'bo-', label='GCB', linewidth=lwid, markersize=msize)
plt.semilogy(xvals, rmse2, 'm+-', label='SCB', linewidth=lwid, markersize=msize)
plt.title('MC = %d' % LL)
plt.legend(fontsize=18)
plt.ylabel('RMSE of DOA-s')
plt.xlabel('SNR (dB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
#%% RMSE of DOA estimate of source 1 vs SNR 
# Create a new figure
plt.figure(2)
plt.clf()  # Clear the current figure
plt.semilogy(xvals, np.sqrt(MSE1[0,]), 'bo-', label='GCB', linewidth=lwid, markersize=msize)
plt.semilogy(xvals, np.sqrt(MSE2[0,]), 'm+-', label='SCB', linewidth=lwid, markersize=msize)
plt.legend(fontsize=18)
plt.ylabel('RMSE for estimate of $\\theta_1$')
plt.xlabel('SNR (dB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
#%% RMSE of DOA estimate of source 4 vs SNR 
plt.figure(3)
plt.clf()  # Clear the current figure
plt.semilogy(xvals, np.sqrt(MSE1[3,]), 'bo-', label='GCB', linewidth=1.0, markersize=msize)
plt.semilogy(xvals, np.sqrt(MSE2[3,]), 'm+-', label='SCB', linewidth=1.0, markersize=msize)
plt.legend(fontsize=18)
plt.ylabel('RMSE for estimate of $\\theta_4$')
plt.xlabel('SNR (dB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()