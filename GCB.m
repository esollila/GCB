function [DOA,pks] = GCB(A,RY,K,phi_vec)
% [DOA,pks] = GCB(A,RY,K,phi_vec)
%
% Greedy Capon Beamformer (GCB) algorithm proposed in Ollila (2024).
%
% INPUT: 
%   A       - Steering matrix of M steering vectors, matrix of size: N x M
%   RY      - Sample Covariance Matrix (SCM), matrix of size N x N
%   K       - the number of sources, a positive integer
%   phi_vec - DOA grid in degrees, so elements are in [-90,90), a vector 
%               of length M
%
% OUTPUT:
%   DOA    -  K-vector of estimated DOAs in degrees (ordered: DOA(1) <.)
%   pks    -  peak indices (K vector that is subset of {1,...,M})
%
% REFERENCE:
%   Esa Ollila, "Greedy Capon Beamformer", ArXiv preprint, 
%   arXiv:2404.15329 [eess.SP], 2024. 
% 
% Author: Esa Ollila, Aalto University, 2024. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize variables
N = size(A,1);% number of sensors
M = size(A,2);% number of dictionary entries

%-- assert: 
assert(isequal(round(K),K) && K < M && isreal(K) && K >0, ... 
    ['''K'' must be a positive integer smaller than ', num2str(M)]);

assert(length(phi_vec)==M, 'sizes of matrix A and vector phi_vec do not match');

%% Initialize
sigc = trace(RY)/N;
SigmaYinv = (1/sigc)*eye(N); 
Ilocs = zeros(1,K);
gam = zeros(K,1);

for k = 1:(K-1)
    
    %% 1. Calculate the powers 
    B =  SigmaYinv*A; % \Sigma^-1 a_m , m=1,..,M
    P_num = subplus(real(sum(conj(B).*(RY*B))));
    P_denum = subplus(real(sum(conj(A).*B)));
    P_denum(P_denum<=10^-18) = 10^-18; % make sure not zero
    P = P_num./P_denum.^2;   
    
   %% 2. find k largest peaks and 3. pick index with least coherence
    if k==1
       [~,indx] = max(P);
    else
       [~,pks] = find_peaks_1D(P,k);
       coh_mat = abs(A(:,pks)'*A(:,Ilocs(1:(k-1))));
       [~,indx] = min(max(coh_mat,[],2));
       indx = pks(indx);       
    end  
    %% 4. Update chosen indices
    Ilocs(k) = indx;

    %% 5. Estimate the signal power  
    gam(k) = subplus(P(indx) - 1/(P_denum(indx)));
    if k==1
        gam(k) = gam(k)*(N/(N-1));
    end
    %% 6. Update the INCM 
    b = SigmaYinv*A(:,indx);
    SigmaYinv =  SigmaYinv - (gam(k)/(1+gam(k)*P_denum(indx)))*b*(b');

end

%% 1. Calculate the powers 
B =  SigmaYinv*A; % \Sigma^-1 a_m , m=1,..,M
P_num = subplus(real(sum(conj(B).*(RY*B))));
P_denum = subplus(real(sum(conj(A).*B)));
P_denum(P_denum<=10^-18) = 10^-18; % make sure not zero
P = P_num./P_denum.^2;   

%% 2. Choose the peaks and compute and return the DOAs
if K==1
    [~,pks] = max(P);
else
    [~,pks] = find_peaks_1D(P,K);
    [~,idx] =  sort(pks);
    pks = pks(idx);
end
DOA = phi_vec(pks);

end

function [pks, locs] = find_peaks_1D(gamma, Nsources)
% fast alternative for findpeaks in 1D case
%

% output variables
pks = zeros(Nsources,1);
locs = zeros(Nsources,1);

% zero padding on the boundary
gamma_new = zeros(length(gamma)+2,1);
gamma_new(2:end-1) = gamma;

[~, Ilocs]= sort(gamma,'descend');

% current number of peaks found
npeaks = 0;

for ii = 1:length(Ilocs)
    
    % local patch area surrounding the current array entry i.e. (r,c)
    local_patch = gamma_new(Ilocs(ii):Ilocs(ii)+2);
    
    % zero the center
    local_patch(2) = 0;
    
    if sum(sum(gamma(Ilocs(ii)) > local_patch)) == 3
        npeaks = npeaks + 1;
        
        pks(npeaks) = gamma(Ilocs(ii));
        locs(npeaks) = Ilocs(ii);
        
        % if found sufficient peaks, break
        if npeaks == Nsources
            break;
        end
    end
    
end

% if Nsources not found
if npeaks ~= Nsources
    pks(npeaks+1:Nsources) = [];
    locs(npeaks+1:Nsources) = [];
end

end