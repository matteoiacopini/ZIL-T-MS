function [Xt, Zt, St, gammas, W, DS0] = DGP_ZIL_MS_tensor_K(I,J,K,Q,rho,R,P,ip,T,meangamma,tau,phi,lambda)
%% DGP_ZIL_MS_tensor_K Simulation of dataset from the ZIL-MS model with K > 1 and common covariates
%
% The observations are binary arrays Xt of size (IxJxK) for each time t = 1,...,T,
% the covariates Zt are common to all the entries of the response array Xt 
% and consist of a (Qx1) vector for each time t = 1,...,T.
%
%    [XT, ZT, ST, GAMMAS, W, DS0] = DGP_ZIL_MS_tensor(I,J,K,Q,RHO,R,P,IP,T,MEANGAMMA,TAU,PHI,LAMBDA)
%    Generates a time series of T binary arrays XT of size (I,J,K) and a time series of T common covariate
%    vectors ZT of size (Q,1) for each time t = 1,...,T, from the ZIL-T-MS model.
%    It also stores the simulated path of the hidden Markov chain, ST, whose transition matrix and 
%    initial probability vector are P and IP, respectively.
%    The coefficient tensor has rank R in each state and the PARAFAC marginals are generated from the
%    prior distribution with hyper-parameters MEANGAMMA,TAU,PHI,LAMBDA, then stored into the cell GAMMAS.
%    The arrays W and DS0 contain the local component of the PARAFAC marginals' covariance and the position
%    of the observables which are drawn from the Dirac component of the ZIL mixture, respectively.
%
% INPUTS
%  I,J,K       1x1   size of observed binary array
%  Q           1x1   number of covariates
%  R           1x1   rank tensor Gamma
%  T           1x1   length time series (observations)
%  rho         Lx1   probability Dirac in sampling X entries
%  P           LxL   transition matrix
%  ip          Lx1   initial probabilities
%  meangamma   Lx1   mean of gammas in each regime
%  tau         1x1   global variance
%  phi         Rx1   component variance
%  lambda      Lx1   local variance hyperparameter
% OUTPUS
%  Xt       IxJxKxT       array of binary observations
%  Zt       QxT           vector of covariates
%  St       Tx1           vector of states
%  gammas   {4,1}(:,R,L)  cell of PARAFAC marginals in each state l=1..L
%  W        (4,R,L)       local variance
%  DS0      (I,J,K,T)     array containing the allocation of observables to the Dirac
%                         such that DS0(i,j,k,t) = 1  if  Xt(i,j,k,t) = 0 and has been 
%                         allocated to the Dirac component of the ZIL mixture
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   L = size(P,1);
   % CHECKS
   if sum(phi) ~= 1
      phi = phi / sum(phi);
   end
   if size(ip,1) == 1
      ip = ip';
   end
   
   % generate COVARIATES z= QxT from stationary VAR(1), entriwise uniform(-lim,lim)
   lag = 1;
   lim = 1.2;
   Zt = VAR_gdp_unif(T, lag, Q, lim);   % Std Normal --> VAR_gdp(T, lag, Q);
   
   % generate PARAMETERS
   gamma1 = zeros(I,R,L);
   gamma2 = zeros(J,R,L);
   gamma3 = zeros(K,R,L);
   gamma4 = zeros(Q,R,L);
   W      = zeros(4,R,L);
   for l=1:L
      W(:,:,l) = exprnd(lambda(l)^2/2, [4,R]); % = Ga(1,theta) [shape/scale] -- E(w) = (?^2/2)^-1
   end
   
   for r=1:R
      for l=1:L
         gamma1(:,r,l) = mvnrnd(meangamma(l)*ones(I,1), tau*phi(r)*W(1,r,l)*eye(I));
         gamma2(:,r,l) = mvnrnd(meangamma(l)*ones(J,1), tau*phi(r)*W(2,r,l)*eye(J));
         gamma3(:,r,l) = mvnrnd(meangamma(l)*ones(K,1), tau*phi(r)*W(3,r,l)*eye(K));
         gamma4(:,r,l) = mvnrnd(meangamma(l)*ones(Q,1), tau*phi(r)*W(4,r,l)*eye(Q));
      end
   end
   gammas = cell(4,1);
   gammas{1} = gamma1;    % IxRxL
   gammas{2} = gamma2;    % JxRxL
   gammas{3} = gamma3;    % KxRxL
   gammas{4} = gamma4;    % QxRxL
   
   % generate STATE St and ARRAY Xt
   Xt    = ones(I,J,K,T);
   DS0   = NaN(I,J,K,T);  % position of 'structural' zeros
   St    = NaN(T,1);
   St(1) = find( cumsum(ip) > rand(1), 1);    % first state (1..L) alternative:   mnrnd(1,ip) * (1:L)'
   for t=1:T
      %%% sample state %%%
      if t > 1
         stepdist = P(St(t-1),:);
         St(t) = find( cumsum(stepdist) > rand(1), 1);
      end
      %%% sample 1st equation %%%
      % build tensor
      Gt  = full( ktensor({gamma1(:,:,St(t)), gamma2(:,:,St(t)), gamma3(:,:,St(t)), gamma4(:,:,St(t))}) );
      psi = double( ttv(Gt, Zt(:,t), 4) );
      % proba (logit) of Bernoulli
      eta = exp(psi) ./ (1 + exp(psi));
      % choose Dirac or logit
      u = rand(I,J,K);
      posDirac = double(u < rho(St(t)));
      DS0(:,:,:,t) = posDirac;
      % posDirac --> 0;  (1-posDirac) --> Bernoulli
      Xt(:,:,:,t) = Xt(:,:,:,t) .* (1-posDirac) .* binornd(1,eta);
   end
end
