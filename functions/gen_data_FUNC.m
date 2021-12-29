function data = gen_data_FUNC(nameArgs)
%% gen_data_FUNC Generate syntetic dataset from ZIL-T-MS model
%
%    DATA = gen_data_FUNC(NAMEARGS) Simulates a synthetic dataset from the ZIL-T-MS model.
%    The parameters of the model are passed as optional name-value pair arguments.
%    The output is a structure, DATA, containing both the parameters used to simulate the
%    data and the time series of the synthetic data.
%
%
% INPUTS (optional)
%  I,J,K   1x1   size of the observed binary array (I,J,K) for each time t=1..T
%  Q       1x1   number of covariates
%  T       1x1   length of the time series (observations)
%  L       1x1   number of latent states
%  R       1x1   CP rank of coefficient tensor G
%  G       IxJxKxQxL coefficient tensor in each state l=1..L
%
% OUTPUS
%  data    stucture containing the fields:
%           * Xt      IxJxKxT    array of binary observations
%           * Zt      QxT        vector of covariates
%           * St      Tx1        vector of latent states
%           * G       IxJxKxQxL  coefficient tensor in each state l=1..L
%           * I,J,K   1x1        size of the observed binary array (I,J,K) for each time t=1..T
%           * Q       1x1        number of covariates
%           * T       1x1        length of the time series (observations)
%           * rho     Lx1        probability of Dirac in sampling the entries of Xt
%           * P       LxL        transition matrix of the hidden Markov chain
%           * ip      Lx1        initial probabilities
%           * Ltrue   1x1        number of latent states
%           * Rtrue   1x1        CP rank of coefficient tensor G
%          If 'G' is not provided by the user, the structure 'data' also contains:
%           * gammas  {3,1}(:,R,L) or {4,1}(:,R,L)  cell of PARAFAC marginals in each state l=1..L
%           * W       (3,R,L) or (4,R,L)            local variance
%           * tau     1x1                           global variance hyperparameter
%           * phi     Rx1                           component variance hyperparameter
%           * lambda  3xL or 4xL                    local variance hyperparameter
%           * DS0     (I,J,K,T);                    position of 'structural' zeros from the Dirac component

% check arguments and set to default values
arguments
   nameArgs.I (1,1) double = 10  % size of the observed binary array (I,J,K) for each time t=1..T
   nameArgs.J (1,1) double = 10
   nameArgs.K (1,1) double = 1
   nameArgs.Q (1,1) double = 3   % number of covariates
   nameArgs.T (1,1) double = 50  % length of the time series (observations)
   nameArgs.L (1,1) double = 2   % number of latent states
   nameArgs.R (1,1) double = 5   % CP rank of coefficient tensor G
   nameArgs.G double = NaN       % coefficient tensor in each state l=1..L
end

I = nameArgs.I;
J = nameArgs.J;
K = nameArgs.K;
Q = nameArgs.Q;
T = nameArgs.T;
L = nameArgs.L;
R = nameArgs.R;
G = nameArgs.G;

% checks
if ~any(isnan(G(:)))
   if ndims(G) == 4
      [I,J,Q,L] = size(G);
   elseif ndims(G) == 5
      [I,J,K,Q,L] = size(G);
   else
      error('G must be (I,J,Q,L) or (I,J,K,Q,L).')
   end
end

if (L < 2)
   error('L must be >= 2.')
end

% set hyperparameters to generate data
if (L == 2)
   P         = [0.8,0.2; 0.3,0.7];
   ip        = ones(L,1) / L;
   rho       = [ 0.7, 0.5];
   meangamma = [unifrnd(-0.6,-0.1),unifrnd(-0.9,0.9)]; %[-0.8, 1.0];
   lambda    = [ 0.5, 1.2];  %[ 1.8, 0.8];
   phi       = ones(R,1)*1/R;
   tau       = 4;
else
   P         = NaN(L,L);
   for l=1:L
      P(l,:) = dirrnd(1*ones(1,L),1);
   end
   ip        = ones(L,1) / L;
   rho       = betarnd(1,1,[L,1]);    % NaN(L,1);
   rho       = sort(rho,'descend');   % identification:  rho(1) > ... > rho(L)
   meangamma = linspace(-0.9,0.9,L);  % NaN(L,1);
   lambda    = sort(linspace(0.5,3,L),'descend');  % NaN(L,1);
   phi       = ones(R,1)*1/R;
   tau       = 4;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate dataset
if ~any(isnan(G(:))) && (K == 1)
   G = tensor(G);
   [Xt,Zt,St,DS0] = DGP_ZIL_MS_tensor_input_G(I,J,Q,rho,P,ip,T,G);
   gammas = NaN;     tau = NaN;     W = NaN;
   lambda = NaN;     phi = NaN;

elseif ~any(isnan(G(:))) && (K >= 2)
   G = tensor(G);
   [Xt,Zt,St,DS0] = DGP_ZIL_MS_tensor_K_input_G(I,J,K,Q,rho,P,ip,T,G);
   gammas = NaN;     tau = NaN;     W = NaN;
   lambda = NaN;     phi = NaN;

elseif isnan(G) && (K == 1)
   [Xt,Zt,St,gammas,W,DS0] = DGP_ZIL_MS_tensor(I,J,Q,rho,R,P,ip,T,meangamma,tau,phi,lambda);
   G = tenzeros(I,J,Q,L);
   for ll=1:L
      G(:,:,:,ll) = ktensor({gammas{1}(:,:,ll),gammas{2}(:,:,ll),gammas{3}(:,:,ll)});
   end

elseif isnan(G) && (K >= 2)
   [Xt,Zt,St,gammas,W,DS0] = DGP_ZIL_MS_tensor_K(I,J,K,Q,rho,R,P,ip,T,meangamma,tau,phi,lambda);
   G = tenzeros(I,J,K,Q,L);
   for ll=1:L
      G(:,:,:,:,ll) = ktensor({gammas{1}(:,:,ll),gammas{2}(:,:,ll),gammas{3}(:,:,ll),gammas{4}(:,:,ll)});
   end
end

% prepare output
data = struct;
data.Xt = Xt;     data.Zt = Zt;
data.I = I;       data.J = J;       data.Q = Q;    data.T = T;
data.Ltrue = L;   data.Rtrue = R;
data.St = St;     data.G = double(G);
data.P = P;       data.rho = rho;   data.ip = ip;
data.gammas = gammas;   data.tau = tau;     data.lambda = lambda;
data.phi = phi;         data.W = W;         data.DS0 = DS0;

end





%% AUXILIARY FUNCTIONS

function [Xt, Zt, St, DS0] = DGP_ZIL_MS_tensor_input_G(I,J,Q,rho,P,ip,T,G)
%% DGP_ZIL_MS_tensor_input_G Simulate dataset for ZIL-T-MS binary tensor model
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



function [Xt, Zt, St, DS0] = DGP_ZIL_MS_tensor_K_input_G(I,J,K,Q,rho,P,ip,T,G)
%% DGP_ZIL_MS_tensor_K_input_G Simulate dataset for ZIL-T-MS binary tensor model
%
% The observations are binary arrays Xt of size (IxJxK) for each time t = 1,...,T,
% the covariates Zt are common to all the entries of the response array Xt 
% and consist of a (Qx1) vector for each time t = 1,...,T.
%
%    [XT, ZT, ST, DS0] = DGP_ZIL_MS_tensor(I,J,K,Q,RHO,R,P,IP,T,G)
%    Generates a time series of T binary arrays XT of size (I,J,K) and a time series of T common covariate
%    vectors ZT of size (Q,1) for each time t = 1,...,T, from the ZIL-T-MS model.
%    The (I,J,K,Q,L) array G contains the coefficient tensor G(:,:,:,l) in each state l = 1,...,L.
%    It also stores the simulated path of the hidden Markov chain, ST, whose transition matrix and 
%    initial probability vector are P and IP, respectively.
%    The array DS0 contains the position of the observables which are drawn from the Dirac component
%    of the ZIL mixture, respectively.
%
% INPUTS
%  I,J,K       1x1   size of observed binary array
%  Q           1x1   number of covariates
%  T           1x1   length time series (observations)
%  rho         Lx1   probability Dirac in sampling X entries
%  P           LxL   transition matrix
%  ip          Lx1   initial probabilities
%  G           IxJxQxL coefficient tensor 
% OUTPUS
%  Xt       IxJxK,T     array of binary observations
%  Zt       QxT         vector of covariates
%  St       Tx1         vector of states
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
      % build tensor
      Gt  = G(:,:,:,:,St(t));
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


function [Xt, Zt, St, gammas, W, DS0] = DGP_ZIL_MS_tensor(I,J,Q,rho,R,P,ip,T,meangamma,tau,phi,lambda)
%% DGP_ZIL_MS_tensor Simulate dataset for ZIL-T-MS binary tensor model
%
% The observations are binary arrays Xt of size (IxJ) for each time t = 1,...,T,
% the covariates Zt are common to all the entries of the response array Xt 
% and consist of a (Qx1) vector for each time t = 1,...,T.
% 
%    [XT, ZT, ST, GAMMAS, W, DS0] = DGP_ZIL_MS_tensor(I,J,Q,RHO,R,P,IP,T,MEANGAMMA,TAU,PHI,LAMBDA)
%    Generates a time series of T binary arrays XT of size (I,J) and a time series of T common covariate
%    vectors ZT of size (Q,1) for each time t = 1,...,T, from the ZIL-T-MS model.
%    It also stores the simulated path of the hidden Markov chain, ST, whose transition matrix and 
%    initial probability vector are P and IP, respectively.
%    The coefficient tensor has rank R in each state and the PARAFAC marginals are generated from the
%    prior distribution with hyper-parameters MEANGAMMA,TAU,PHI,LAMBDA, then stored into the cell GAMMAS.
%    The arrays W and DS0 contain the local component of the PARAFAC marginals' covariance and the position
%    of the observables which are drawn from the Dirac component of the ZIL mixture, respectively.
%
% INPUTS
%  I,J         1x1   size of observed binary matrix
%  Q           1x1   number of covariates
%  R           1x1   CP rank of coefficient tensor G
%  T           1x1   length time series (observations)
%  rho         Lx1   probability Dirac in sampling X entries
%  P           LxL   transition matrix
%  ip          Lx1   initial probabilities
%  meangamma   Lx1   mean of gammas in each regime
%  tau         1x1   global variance
%  phi         Rx1   component variance
%  lambda      Lx1   local variance hyperparameter
% OUTPUS
%  Xt       IxJxT         array of binary observations
%  Zt       QxT           vector of covariates
%  St       Tx1           vector of states
%  gammas   {3,1}(:,R,L)  cell of PARAFAC marginals in each state l=1..L
%  W        (3,R,L)       local variance
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
   gamma3 = zeros(Q,R,L);
   W      = zeros(3,R,L);
   for l=1:L
      W(:,:,l) = exprnd(lambda(l)^2/2, [3,R]); % = Ga(1,theta) [shape/scale] -- E(w) = (?^2/2)^-1
   end
   
   for r=1:R
      for l=1:L
         gamma1(:,r,l) = mvnrnd(meangamma(l)*ones(I,1), tau*phi(r)*W(1,r,l)*eye(I));
         gamma2(:,r,l) = mvnrnd(meangamma(l)*ones(J,1), tau*phi(r)*W(2,r,l)*eye(J));
         gamma3(:,r,l) = mvnrnd(meangamma(l)*ones(Q,1), tau*phi(r)*W(3,r,l)*eye(Q));
      end
   end
   gammas = cell(3,1);
   gammas{1} = gamma1;    % IxRxL
   gammas{2} = gamma2;    % JxRxL
   gammas{3} = gamma3;    % QxRxL
   
   % generate STATE St and ARRAY Xt
   Xt = ones(I,J,T);
   DS0   = NaN(I,J,T);  % position of 'structural' zeros
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
      Gt  = full( ktensor({gamma1(:,:,St(t)), gamma2(:,:,St(t)), gamma3(:,:,St(t))}) );
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


function [Xt, Zt, St, gammas, W, DS0] = DGP_ZIL_MS_tensor_K(I,J,K,Q,rho,R,P,ip,T,meangamma,tau,phi,lambda)
%% DGP_ZIL_MS_tensor_K Simulation of dataset for ZIL-MS binary tensor model
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
%  R           1x1   CP rank of coefficient tensor G
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



function Y = VAR_gdp_unif(T,lags,dim,unif_lim)
%% VAR_gdp_unif   Simulate Stationary VAR(p)
% 
%   Y = VAR_gdp_unif(T,LAGS,DIM,UNIF_LIM) simulates a stationary VAR process with LAGS lags,
%   length T, and cross-sectional size DIM, storing the results into a (DIM,T) matrix Y.
%   The entries of each coefficient matrix of the process are drawn from a uniform distribution
%   on the interval (UNIF_LIM(1), UNIF_LIM(2)).
%
% INPUTS
%   T        1x1  time series length (observations)
%  lags      1x1  #lags of VAR
%  dim       1x1  dimension of yt
%  unif_lim  2x1  limits uniform distr from which entries of A(l) are drawn
%
% OUTPUT
%   Y        dim x T matrix of simulated variables:   Y= [y(1),...,y(T)]    y(t) dimx1
%
% NOTE: noise distribution Normal, covariance matrix IW(eye,dim)/dim
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate #lags matrices of coefficients for Stationary VAR
   coeff_mat= zeros(dim,dim*lags);
   for l=1:lags
      coeff_mat(:, (1+(l-1)*dim):(l*dim) ) = VAR_coeff(dim,unif_lim);
   end
   
   % generate random noise --> get NORMAL with Covariance= IW(eye,N)/N
   SS = iwishrnd(eye(dim)/dim,dim);
   E = randn(T,dim)*chol(SS);    % rows= et~N(0,S) = N(0,I)*L  with S=LL' (Cholesky)
   
   % generate variables -- [1:lags]
   Y = [rand(lags,dim); zeros(T-lags,dim)];
   for k = lags:T      %-- [lags:T]
       ylag = mlag(Y,lags);   % mlag(.): wants/gives row data
       F = ylag(k,:);         % 1x(dim*p): Y(1,t-p)..Y(N,t-p)..Y(1,t-1)..Y(N,t-1)
       Y(k,:) = F*coeff_mat' + E(k,:);
   end
   % transpose: output Y= (dim)xT
   Y=Y';
end


function A = VAR_coeff(dim,lim)
%% VAR_coeff Build Coefficient Matrix for (Stationary) VAR
%
% INPUTS
%  dim   1x1 size of A (dim x dim)
%  lim   1x1 interval for random uniform entries
%
% OUTPUT
%   A    dim x dim   matrix of coefficients for STATIONARY VAR
%
% NOTE: enrties are uniform (-lim,lim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   A = random('uniform',-lim,lim, [dim dim]);
   % STATIONARITY CHECK: if not, regenerate
   ch_coeff = sum(abs(eig(A))>1);
   while ch_coeff > 0
      A = random('uniform',-lim,lim, [dim dim]);
      ch_coeff = sum(abs(eig(A))>1);
   end
end


function [Xlag] = mlag(X,lags)
%% mlag Get lags of matrix X(t)
%
%       X(t) = (dim)x(dim) --> Xlag = (dim)x(dim*p):    Xlag = [X(t-p),...,X(t-1)]
% If lags > 1 --> concatenates matrices vertically:
% IDEA: 'downshift' series, putting zeros on top (initial values)
% NOTE: inserts zeros for unknown values X(T-p)
%
% INPUTS
%   X     matrix, rows are data, first row = x(1):
%                    x(1)= [X(1,1), X(1,2),...,X(1,N)]
%                 X= ... = ...
%                    x(T)= [X(T,1), X(T,2),...,X(T,N)]
%   p     lags (to be computed)
%
% OUTPUT
%   Xlag  matrix of lagged X --> Xlag = [X(t-1),...,X(t-p)]
%                 xlag(1)= [X(1-p,1), X(1-p,2),...,X(1-p,N)]
%              Xlag= ... = ...
%                 xlag(T)= [X(T-p,1), X(T-p,2),...,X(T-p,N)]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
   [Traw,N] = size(X);
   if lags >= Traw
      warning('Number of lags greater than time series. Put zeros.');
   end
   Xlag = zeros(Traw,N*lags);
   for ii=1:lags
       Xlag(lags+1:Traw, (N*(ii-1)+1):N*ii) = X(lags+1-ii:Traw-ii, 1:N);
   end
end
