function [Xt, Zt, St, DS0] = DGP_ZIL_MS_tensor_input_G(I,J,Q,rho,P,ip,T,G)
%% DGP_ZIL_MS_tensor_input_G Simulate dataset from the ZIL-T-MS model with K=1 and common covariates, starting from a given coefficient tensor provided as input.
%
% The observations are binary arrays Xt of size (IxJ) for each time t = 1,...,T,
% the covariates Zt are common to all the entries of the response array Xt 
% and consist of a (Qx1) vector for each time t = 1,...,T.
%
%    [XT, ZT, ST, DS0] = DGP_ZIL_MS_tensor(I,J,Q,RHO,R,P,IP,T,G)
%    Generates a time series of T binary arrays XT of size (I,J) and a time series of T common covariate
%    vectors ZT of size (Q,1) for each time t = 1,...,T, from the ZIL-T-MS model.
%    The (I,J,Q,L) array G contains the coefficient tensor G(:,:,:,l) in each state l = 1,...,L.
%    It also stores the simulated path of the hidden Markov chain, ST, whose transition matrix and 
%    initial probability vector are P and IP, respectively.
%    The array DS0 contains the position of the observables which are drawn from the Dirac component
%    of the ZIL mixture, respectively.
%
% INPUTS
%  I,J         1x1   size of observed binary array
%  Q           1x1   number of covariates
%  T           1x1   length time series (observations)
%  rho         Lx1   probability Dirac in sampling X entries
%  P           LxL   transition matrix
%  ip          Lx1   initial probabilities
%  G           IxJxQxL coefficient tensor 
% OUTPUS
%  Xt       IxJxT       array of binary observations
%  Zt       QxT         vector of covariates
%  St       Tx1         vector of states
%  DS0      (I,J,T)     array containing the allocation of observables to the Dirac
%                       such that DS0(i,j,t) = 1  if  Xt(i,j,t) = 0 and has been 
%                       allocated to the Dirac component of the ZIL mixture
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % CHECKS
   if size(ip,1) == 1
      ip = ip';
   end
   
   % generate COVARIATES z= QxT from stationary VAR(1), entriwise uniform(-lim,lim)
   lag = 1;
   lim = 1.2;
   Zt = VAR_gdp_unif(T, lag, Q, lim);   % Std Normal --> VAR_gdp(T, lag, Q);
   
   
   % generate STATE St and ARRAY Xt
   Xt    = ones(I,J,T);
   DS0   = NaN(I,J,T);  % position of 'structural' zeros
   St    = NaN(T,1);
   St(1) = find( cumsum(ip) > rand(1), 1);    % first state (1..L) alternative:   mnrnd(1,ip) * (1:L)'
   for t=1:T
      %%% sample state %%%
      if t > 1
         stepdist = P(St(t-1),:);
         St(t) = find( cumsum(stepdist) > rand(1), 1);
      end
      % build tensor
      Gt  = G(:,:,:,St(t));
      psi = double( ttv(Gt, Zt(:,t), 3) );
      % proba (logit) of Bernoulli
      eta = exp(psi) ./ (1 + exp(psi));
      % choose Dirac or logit
      u = rand(I,J);
      posDirac = double(u < rho(St(t)));
      DS0(:,:,t) = posDirac;
      % posDirac --> 0;  (1-posDirac) --> Bernoulli
      Xt(:,:,t) = Xt(:,:,t) .* (1-posDirac) .* binornd(1,eta);
   end
   
end
