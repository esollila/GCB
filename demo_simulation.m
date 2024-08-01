%% DOA estimation performance: Greedy Capon vs. Standard Capon Beamformer
% This simulation repeats the simulation set-up RMSE vs SNR that is
% reporeted in the publication: 
%  
% Esa Ollila, "Greedy Capon Beamformer", ArXiv preprint, 
% arXiv:2404.15329 [eess.SP], Apr. 7, 2024. 
% 
% Author: Esa Ollila, Aalto University

clearvars;
%% Model parameters 
N = 20;   % nr. of sensors in the ULA
M = 1801; % grid size
L = 125;  % number of snapshots available 
K = 4;    % number of sources

% True DOA-s (all except the 1st one is off the grid)
DOA_src1 = -30.1;     
DOA_src2 = -20.02;
DOA_src3 = -10.02;
DOA_src4 = 3.02;
DOA_src = [DOA_src1, DOA_src2,DOA_src3,DOA_src4];
A0 = exp(-1j*pi*(0:(N-1))'*sind(DOA_src)); % true steering matrix 
%% Create the DOA grid and the respective steering matrix of size N x M: 
% We assume ULA with half a vawelength spacing
dphi = 180/(M-1);      % angular resolution
phi_vec = -90:dphi:90; % grid of DOAs
A = exp(-1j*pi*(0:(N-1))'*sind(phi_vec)); % steering matrix for grid DOAs

%% Compute source powers 
% source 2, 3, and 4 have -1, -2 and -5 dB lower power than source 1=

SNR = -6.5:-0.5:-12;
sigma_vec1 = sqrt(10.^(SNR/10)); 
sigma_vec2 = sqrt(10.^((SNR-1)/10)); 
sigma_vec3 = sqrt(10.^((SNR-2)/10)); 
sigma_vec4 = sqrt(10.^((SNR-5)/10)); 
sigmas = [sigma_vec1; sigma_vec2; sigma_vec3; sigma_vec4];

%% Simulation parameters
LL = 15000; % number of MCtrials -> same as in the paper. 
% Put LL smaller for faster computations

nSNR = length(sigma_vec1);
DOA1 = zeros(LL,K,nSNR);
DOA2 = zeros(LL,K,nSNR);
AVEcpu  = zeros(nSNR,2);

for isnr = 1:nSNR

    sigma = sigmas(:,isnr).';
    fprintf('%d/%d , SNR= %.3f\n',isnr,nSNR,SNR(isnr));

    rng("default");
    tim1 = 0; tim2=0; 

    for ell=1:LL
     
        %% Generate the L snapshots 
        s = diag(sigma)*complex(randn(K,L),randn(K,L))/sqrt(2);
        noise = complex(randn(N,L),randn(N,L))/sqrt(2);
        y = A0*s + noise;                  
        RY = (1/L)*y*(y'); % The sample covariance matrix (SCM)
  
        %% 1. Greedy Capon Beamformer (GCB)
        tStart = tic;
        DOAgbf = GCB(A,RY,K,phi_vec);
        timer1 = toc(tStart);
        tim1 = tim1 + timer1;
        DOA1(ell,:,isnr) = DOAgbf;

        if ~isempty(setdiff(1:K,knnsearch(DOAgbf',DOA_src')))       
            idxset = zeros(1,K);
            indx = 1:K;
            for k = 1:(K-1)
                idx = knnsearch(DOAgbf(indx)',DOA_src(k));
                idxset(k) = indx(idx);
                indx = setdiff(indx,idxset(k));
            end
            idxset(K)=indx;
            DOA1(ell,:,isnr) = DOAgbf(idxset);
        end
  
        %% 2.  Standard Capon Beamformer (SCB)

        tStart = tic;
        invRY = RY \ eye(N);   
        pow = 10*log10(1./real(sum(A'*invRY.*(A.'),2)).');
        [~,locs] = findpeaks(pow,'SortStr','descend');
        est2 = sort(phi_vec(locs(1:K)));    
        timer2 = toc(tStart);
        tim2 = tim2 + timer2;
        DOA2(ell,:,isnr) = est2;     
      
        if ~isempty(setdiff(1:K,knnsearch(est2',DOA_src')))        
            idxset = zeros(1,K);
            indx = 1:K;
            for k = 1:(K-1)
                idx = knnsearch(est2(indx)',DOA_src(k));
                idxset(k) = indx(idx);
                indx = setdiff(indx,idxset(k));
            end
            idxset(K)=indx;
            DOA2(ell,:,isnr) = est2(idxset);
        end
  
        if mod(ell,1000) == 0
            fprintf('.');
        end
    end
    AVEcpu(isnr,:) = [tim1,tim2]/LL;  

    fprintf(" Done\n");
    %%
end
%%
fprintf('.')

%%
MSE1 = reshape(mean((DOA1-DOA_src).^2),size(DOA1,[2,3]));
MSE2 = reshape(mean((DOA2-DOA_src).^2),size(DOA2,[2,3]));

%%
gm = geomean(sigmas.^2);
xvals =  10*log10(gm);
msize = 10;

%% RMSE of DOAs vs SNR
figure(1); clf
semilogy(xvals,sqrt(sum(MSE1)),'bo-','DisplayName','GCB','LineWidth',1.0,'MarkerSize',msize); 
hold on;
semilogy(xvals,sqrt(sum(MSE2)),'m+-','DisplayName','SCB','LineWidth',1.0,'MarkerSize',msize);
legend('FontSize',18); 
ylabel('RMSE of DOA-s')
xlabel('SNR (dB)')
grid on; axis tight;axis square;

%% RMSE of DOA estimate of source 1 vs SNR 
figure(2); clf;
semilogy(xvals,sqrt(MSE1(1,:)),'bo-','DisplayName','GCB','LineWidth',1.0,'MarkerSize',msize);
hold on;
semilogy(xvals,sqrt(MSE2(1,:)),'m+-','DisplayName','SCB','LineWidth',1.0,'MarkerSize',msize);

legend('FontSize',18); 
ylabel('RMSE for estimate of \theta_1')
xlabel('SNR (dB)')
grid on; axis tight; axis square;

%% RMSE of DOA estimate of source 4 vs SNR 
figure(3); clf;
semilogy(xvals,sqrt(MSE1(4,:)),'bo-','DisplayName','GCB','LineWidth',1.0,'MarkerSize',msize);
hold on;
semilogy(xvals,sqrt(MSE2(4,:)),'m+-','DisplayName','SCB','LineWidth',1.0,'MarkerSize',msize);

legend('FontSize',18); 
ylabel('RMSE for estimate of \theta_4')
xlabel('SNR (dB)')
grid on; axis tight

