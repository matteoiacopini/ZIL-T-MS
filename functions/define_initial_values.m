function [INIT] = define_initial_values(Xt,Zt,HYPER,nameArgs)
%% define_initial_values Generate initial values for the model
%
%   [INIT] = define_initial_values(XT,ZT,HYPER,,NAMEARGS) 
%   Defines the initial values for all the parameters of the ZIL-T-MS model, with observed
%   binary arrays XT, covariates ZT, and hyperparameter values contained in the structure HYPER.
%   If the additional inputs necessary for the initialization are not provided 
%   as optional name-value pair arguments, then the function will assign default values.
%   The output is a structure, INIT, containing all the initial values.
%
% INPUTS
%   Xt  (I,J,K,T)   observed binary array, of size (I,J,K,T), with T being the time length
%   Zt  (Q,T)       observed covariates, of size (Q,T), with T being the time length
%   HYPER           structure containing all the hyperparameter values. Fields:
%                     * L          (1,1)   number of states of the hidden chain
%                     * R          (1,1)   rank of the coefficient tensor
%                     * alambda    (L,1)   shape parameter of the Gamma prior for lambda, in each state l=1..L
%                     * blambda    (L,1)   scale parameter of the Gamma prior for lambda, in each state l=1..L
%                     * alphabar   (1,1)   shape parameter of the Dirichlet prior for phi
%                     * ataubar    (1,1)   shape parameter of the Gamma prior for tau
%                     * btaubar    (1,1)   scale parameter of the Gamma prior for tau
%                     * cbar       (L,L)   parameter vector of the Dirichlet prior for each row l=1..L
%                                          of the Markov chain transition matrix
%                     * arhobar    (L,1)   shape parameter 'a' of the Beta prior for rho, in each state l=1..L
%                     * brhobar    (L,1)   shape parameter 'b' of the Beta prior for rho, in each state l=1..L
%                     * ident_constr  (1,1) logical impose identification constraint during MCMC?
%                     * constr_g3     (1,1) logical impose constraint on gamma3?
%
% INPUTS (optional)
%   thrhtd     (1,1)   threshold on network density, used to initilize the latent state vector st
%   IterSA     (1,1)   number of iterations of the simulated annealing
%
% OUTPUT
%   INIT   structure with fields given by the initial values for all the latent variables and parameters. Fields:
%          * sthat  (T,1) latent states
%          * gamma1hat  (I,R,L) CP marginal vector of mode 1, for each r=1..R and state l=1..L
%          * gamma2hat  (J,R,L) CP marginal vector of mode 2, for each r=1..R and state l=1..L
%          * gamma3hat  (K,R,L) CP marginal vector of mode 3, for each r=1..R and state l=1..L
%          * gamma4hat  (Q,R,L) CP marginal vector of mode 4, for each r=1..R and state l=1..L
%          * Ghat       (I,J,K,Q,L) coefficient tensor for state l=1..L
%          * iphat      (L,1) vector of initial probabilities for the latent states
%          * xihat      (L,L) transition matric of the hidden Markov chain
%          * rhohat     (L,1) mixing probability for each state l=1..L
%          * tauhat     (1,1) global variance of the gammas
%          * what       (3,R,L) local variance of the gammas, for each r=1..R and state l=1..L
%          * lambdahat  (L,1) hyperparameter for the local variance of the gammas, for each state l=1..L
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

arguments
   Xt double               % observed binary array, of size (I,J,K,T), with T being the time length
   Zt double               % observed covariates, of size (Q,T), with T being the time length
   HYPER struct            % structure with hyperparameter values
   nameArgs.thrhtd (1,1) double = 0.08;
   nameArgs.IterSA (1,1) double = 80; %200;
end

thrhtd = nameArgs.thrhtd;
IterSA = nameArgs.IterSA;

% recover hyperparameters
ataubar = HYPER.ataubar;
btaubar = HYPER.btaubar;
alambda = HYPER.alambda;
blambda = HYPER.blambda;
L = HYPER.L;
R = HYPER.R;
if ndims(Xt) == 3       % case of matrix-valued observations
   constr_g3 = false;
elseif ndims(Xt) == 4   % case of matrix-valued observations
   constr_g3 = HYPER.constr_g3;
end

% checks
% if size(Xt,ndims(Xt)) ~= size(Zt,ndims(Zt))
%    error('Xt must be (I,J,K,T) and Zt must be (Q,T).')
% end
if constr_g3
   if mod(R,2) == 0
      Rbar = R / 2;
   else
      R = R + 1;
      Rbar = R / 2;
      warning('In the constrained model, R should be an even integer --> now using (R+1).');
   end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ismatrix(Zt)
   Q = size(Zt,1);   % vector of covariates (common to all edges)
else
   Q = size(Zt,4);   % vector of covariates (all or some are edge-specific)
end

if (ndims(Xt) == 3) && (ismatrix(Zt))       % case of matrix-valued observations
   [I,J,T] = size(Xt);
   %%% Initialize chain St %%%
   totdeg = squeeze(sum(Xt(:,:,:),[1,2]));    % total degree
   sthat  = 1 + (totdeg > thrhtd*I*J);
   if all(sthat==1)
      while sum(sthat==1)/T > 0.80
         thrhtd = thrhtd * 0.8;
         sthat = 1 + (totdeg > thrhtd*I*J);
      end
   elseif all(sthat==2)
      while sum(sthat==2)/T > 0.80
         thrhtd = thrhtd * 1.2;
         sthat = 1 + (totdeg > thrhtd*I*J);
      end
   end

   %%% Initialize Gammas %%%
   k  = I^3*2;
   rw = ones(L,1);
   muinit = linspace(-1.0,1.0,L);
   sigma  = ones(L,1)*0.4;
   pen1     = 4.5;
   pen0     = 1.0;
   penNorm  = 0.8;
   target_w = linspace(-1.0,1.0,L);
   [gammaInit] = SimAnnMScl(Xt,Zt,L,R,IterSA,sthat,rw,muinit,sigma,k,pen1,pen0,penNorm,target_w,'loglog');
   clearvars IterSA k rw muinit sigma pen1 pen0 penNorm target_weight thrhtd
   gamma1hat = zeros(I,R,L);
   gamma2hat = zeros(J,R,L);
   gamma3hat = zeros(Q,R,L);
   Ghat = tenzeros(I,J,Q,L);
   for ll=1:L
      gamma1hat(:,:,ll) = gammaInit{1}(:,:,ll);
      gamma2hat(:,:,ll) = gammaInit{2}(:,:,ll);
      gamma3hat(:,:,ll) = gammaInit{3}(:,:,ll);
      Ghat(:,:,:,ll) = ktensor_mod({gamma1hat(:,:,ll), gamma2hat(:,:,ll), gamma3hat(:,:,ll)});
   end
   %%% Initialize other parameters %%%
   iphat  = dirrnd(ones(1,L)*0.8, 1);
   xihat  = ones(L,L)/L;
   rhohat = linspace(0.9,0.1,L);
   tauhat = gamrnd(ataubar, btaubar^(-1)); % Ga(a,b) = Ga(a,1/b) -- rate (paper) <-> scale (MATLAB)
   lambdahat = zeros(L,1);
   what = zeros(3,R,L);
   for ll=1:L
      lambdahat(ll) = gamrnd(alambda(ll),blambda(ll)^(-1));
      what(:,:,ll)  = exprnd(lambdahat(ll)^(-1),[3,R]);  % Exp(l) = Ga(1,l)
   end



elseif (ndims(Xt) == 4) && (ismatrix(Zt))   % case of tensor-valued observations
   [I,J,K,T] = size(Xt);

   %%% Initialize chain St %%%
   totdeg = squeeze(sum(Xt(:,:,1,:),[1,2]));    % total degree on Layer 1 (logReturns)
   sthat  = 1 + (totdeg > thrhtd*I*J);
   if all(sthat==1)
      while sum(sthat==1)/T > 0.80
         thrhtd = thrhtd * 0.8;
         sthat = 1 + (totdeg > thrhtd*I*J);
      end
   elseif all(sthat==2)
      while sum(sthat==2)/T > 0.80
         thrhtd = thrhtd * 1.2;
         sthat = 1 + (totdeg > thrhtd*I*J);
      end
   end

   %%% Initialize Gammas %%%
   k  = I^3*2;
   rw = ones(L,1);
   muinit = linspace(-1.0,1.0,L);
   sigma  = ones(L,1)*0.4;
   pen1     = 4.5;
   pen0     = 1.0;
   penNorm  = 0.8;
   target_w = linspace(-1.0,1.0,L);
   [gammaInit] = SimAnnMScl_tensorData(Xt,Zt,L,R,IterSA,sthat,rw,muinit,sigma,k,pen1,pen0,penNorm,target_w,'loglog');
   clearvars IterSA k rw muinit sigma pen1 pen0 penNorm target_weight thrhtd
   gamma1hat = zeros(I,R,L);
   gamma2hat = zeros(J,R,L);
   gamma3hat = zeros(K,R,L);
   gamma4hat = zeros(Q,R,L);
   Ghat = tenzeros(I,J,K,Q,L);
   for ll=1:L
      gamma1hat(:,:,ll) = gammaInit{1}(:,:,ll);
      gamma2hat(:,:,ll) = gammaInit{2}(:,:,ll);
      gamma3hat(:,:,ll) = gammaInit{3}(:,:,ll);
      gamma4hat(:,:,ll) = gammaInit{4}(:,:,ll);
      if constr_g3
         %%%%%%%%% IMPOSE CONSTRAINT on gamma3hat %%%%%%%%%
         gamma3hat(:,    1:Rbar,ll) = repmat([1;0],1,Rbar);
         gamma3hat(:,Rbar+1:end,ll) = repmat([0;1],1,Rbar);
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      end
      Ghat(:,:,:,:,ll) = ktensor_mod({gamma1hat(:,:,ll), gamma2hat(:,:,ll), gamma3hat(:,:,ll), gamma4hat(:,:,ll)});
   end
   %%% Initialize other parameters %%%
   iphat  = dirrnd(ones(1,L)*0.8, 1);
   xihat  = ones(L,L)/L;
   rhohat = linspace(0.9,0.1,L);
   tauhat = gamrnd(ataubar, btaubar^(-1)); % Ga(a,b) = Ga(a,1/b) -- rate (paper) <-> scale (MATLAB)
   lambdahat = zeros(L,1);
   if constr_g3
      what = zeros(3,R,L);
      for ll=1:L
         lambdahat(ll) = gamrnd(alambda(ll),blambda(ll)^(-1));
         what(:,:,ll)  = exprnd(lambdahat(ll)^(-1),[3,R]);  % Exp(l) = Ga(1,l)
      end
   else
      what = zeros(4,R,L);
      for ll=1:L
         lambdahat(ll) = gamrnd(alambda(ll),blambda(ll)^(-1));
         what(:,:,ll)  = exprnd(lambdahat(ll)^(-1),[4,R]);  % Exp(l) = Ga(1,l)
      end
   end



elseif (ndims(Xt) == 3) && (ndims(Zt) == 4)       % case of matrix-valued observations, edge-specific covariates
   [I,J,T] = size(Xt);
   %%% Initialize chain St %%%
   totdeg = squeeze(sum(Xt(:,:,:),[1,2]));    % total degree
   sthat  = 1 + (totdeg > thrhtd*I*J);
   if all(sthat==1)
      while sum(sthat==1)/T > 0.80
         thrhtd = thrhtd * 0.8;
         sthat = 1 + (totdeg > thrhtd*I*J);
      end
   elseif all(sthat==2)
      while sum(sthat==2)/T > 0.80
         thrhtd = thrhtd * 1.2;
         sthat = 1 + (totdeg > thrhtd*I*J);
      end
   end

   %%% Initialize Gammas %%%
   k  = I^3*2;
   rw = ones(L,1);
   muinit = linspace(-1.0,1.0,L);
   sigma  = ones(L,1)*0.4;
   pen1     = 4.5;
   pen0     = 1.0;
   penNorm  = 0.8;
   target_w = linspace(-1.0,1.0,L);
   [gammaInit] = SimAnnMScl_edge_specific(Xt,Zt,L,R,IterSA,sthat,rw,muinit,sigma,k,pen1,pen0,penNorm,target_w,'loglog');
   clearvars IterSA k rw muinit sigma pen1 pen0 penNorm target_weight thrhtd
   gamma1hat = zeros(I,R,L);
   gamma2hat = zeros(J,R,L);
   gamma3hat = zeros(Q,R,L);
   Ghat = tenzeros(I,J,Q,L);
   for ll=1:L
      gamma1hat(:,:,ll) = gammaInit{1}(:,:,ll);
      gamma2hat(:,:,ll) = gammaInit{2}(:,:,ll);
      gamma3hat(:,:,ll) = gammaInit{3}(:,:,ll);
      Ghat(:,:,:,ll) = ktensor_mod({gamma1hat(:,:,ll), gamma2hat(:,:,ll), gamma3hat(:,:,ll)});
   end
   %%% Initialize other parameters %%%
   iphat  = dirrnd(ones(1,L)*0.8, 1);
   xihat  = ones(L,L)/L;
   rhohat = linspace(0.9,0.1,L);
   tauhat = gamrnd(ataubar, btaubar^(-1)); % Ga(a,b) = Ga(a,1/b) -- rate (paper) <-> scale (MATLAB)
   lambdahat = zeros(L,1);
   what = zeros(3,R,L);
   for ll=1:L
      lambdahat(ll) = gamrnd(alambda(ll),blambda(ll)^(-1));
      what(:,:,ll)  = exprnd(lambdahat(ll)^(-1),[3,R]);  % Exp(l) = Ga(1,l)
   end



elseif (ndims(Xt) == 4) && (~ismatrix(Zt))   % case of tensor-valued observations and edge-specific covariates
   [I,J,K,T] = size(Xt);

   %%% Initialize chain St %%%
   totdeg = squeeze(sum(Xt(:,:,1,:),[1,2]));    % total degree on Layer 1 (logReturns)
   sthat  = 1 + (totdeg > thrhtd*I*J);
   if all(sthat==1)
      while sum(sthat==1)/T > 0.80
         thrhtd = thrhtd * 0.8;
         sthat = 1 + (totdeg > thrhtd*I*J);
      end
   elseif all(sthat==2)
      while sum(sthat==2)/T > 0.80
         thrhtd = thrhtd * 1.2;
         sthat = 1 + (totdeg > thrhtd*I*J);
      end
   end

   %%% Initialize Gammas %%%
   k  = I^3*2;
   rw = ones(L,1);
   muinit = linspace(-1.0,1.0,L);
   sigma  = ones(L,1)*0.4;
   pen1     = 4.5;
   pen0     = 1.0;
   penNorm  = 0.8;
   target_w = linspace(-1.0,1.0,L);
   [gammaInit] = SimAnnMScl_tensorData_edge_specific(Xt,Zt,L,R,IterSA,sthat,rw,muinit,sigma,k,pen1,pen0,penNorm,target_w,'loglog');
   clearvars IterSA k rw muinit sigma pen1 pen0 penNorm target_weight thrhtd
   gamma1hat = zeros(I,R,L);
   gamma2hat = zeros(J,R,L);
   gamma3hat = zeros(K,R,L);
   gamma4hat = zeros(Q,R,L);
   Ghat = tenzeros(I,J,K,Q,L);
   for ll=1:L
      gamma1hat(:,:,ll) = gammaInit{1}(:,:,ll);
      gamma2hat(:,:,ll) = gammaInit{2}(:,:,ll);
      gamma3hat(:,:,ll) = gammaInit{3}(:,:,ll);
      gamma4hat(:,:,ll) = gammaInit{4}(:,:,ll);
      if constr_g3
         %%%%%%%%% IMPOSE CONSTRAINT on gamma3hat %%%%%%%%%
         gamma3hat(:,    1:Rbar,ll) = repmat([1;0],1,Rbar);
         gamma3hat(:,Rbar+1:end,ll) = repmat([0;1],1,Rbar);
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      end
      Ghat(:,:,:,:,ll) = ktensor_mod({gamma1hat(:,:,ll), gamma2hat(:,:,ll), gamma3hat(:,:,ll), gamma4hat(:,:,ll)});
   end
   %%% Initialize other parameters %%%
   iphat  = dirrnd(ones(1,L)*0.8, 1);
   xihat  = ones(L,L)/L;
   rhohat = linspace(0.9,0.1,L);
   tauhat = gamrnd(ataubar, btaubar^(-1)); % Ga(a,b) = Ga(a,1/b) -- rate (paper) <-> scale (MATLAB)
   lambdahat = zeros(L,1);
   if constr_g3
      what = zeros(3,R,L);
      for ll=1:L
         lambdahat(ll) = gamrnd(alambda(ll),blambda(ll)^(-1));
         what(:,:,ll)  = exprnd(lambdahat(ll)^(-1),[3,R]);  % Exp(l) = Ga(1,l)
      end
   else
      what = zeros(4,R,L);
      for ll=1:L
         lambdahat(ll) = gamrnd(alambda(ll),blambda(ll)^(-1));
         what(:,:,ll)  = exprnd(lambdahat(ll)^(-1),[4,R]);  % Exp(l) = Ga(1,l)
      end
   end
end

% create structure with all hyperparameters
INIT = struct;
INIT.sthat     = sthat;
INIT.gamma1hat = gamma1hat;
INIT.gamma2hat = gamma2hat;
INIT.gamma3hat = gamma3hat;
if ndims(Xt) == 4
   INIT.gamma4hat = gamma4hat;
end
INIT.Ghat      = Ghat;
INIT.iphat     = iphat;
INIT.xihat     = xihat;
INIT.rhohat    = rhohat;
INIT.tauhat    = tauhat;
INIT.what      = what;
INIT.lambdahat = lambdahat;
end
