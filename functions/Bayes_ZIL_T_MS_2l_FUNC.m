function [OUT] = Bayes_ZIL_T_MS_2l_FUNC(Xt,Zt, nameArgs)
%% Bayes_ZIL_T_MS_2l_TEST Estimate the Bayesian Zero-Inflated Logit Tensor model with Markov Switching coefficients (ZIL-T-MS) with common covariates and K>1 layers
%
%   [OUT] = Bayes_ZIL_T_MS_2l_edgespec_FUNC(XT,ZT,NAMEARGS)
%   Estimates the Bayesian Zero-Inflated Logit with Markov Switching coefficients model (ZIL-T-MS)
%   with edge-specific covariates.
%   The observations consists in binary arrays XT of size (I,J,K,T) and covariates ZT of size (Q,T).
%   If the additional inputs necessary for the estimation are not provided as optional name-value
%   pair arguments, then the function will assign default values.
%   The output is a structure, OUT, containing the posterior mean of the coefficient tensor, G,
%   and the MAP estimate of the hidden Markov chain, ST.
%
% INPUTS
%  Xt  (I,J,K,T)  array of (I,J,K) binary observations for each time t=1...T
%  Zt  (Q,T)      vector of Q covariates for each time t=1...T
%  nameArgs       optional arguments, specified as ('name',value) pairs. Includes:
%     - 'NumIter'   number of MCMC iterations to be retained
%     - 'burn'      number of burn-in iterations to be discarded
%     - 'thin'      thinning factor (retain 1 iteration every 'thin' ones)
%     - 'irep'      display cumulative computing time once every 'irep' iterations
%     - 'HYPER'     structure containing the hyperparameter values. Fields:
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
%     - 'INIT'      structure containing the initial values of latent variables and parameters. Fields:
%                      * sthat  (T,1) latent states
%                      * gamma1hat  (I,R,L) CP marginal vector of mode 1, for each r=1..R and state l=1..L
%                      * gamma2hat  (J,R,L) CP marginal vector of mode 2, for each r=1..R and state l=1..L
%                      * gamma3hat  (K,R,L) CP marginal vector of mode 3, for each r=1..R and state l=1..L
%                      * gamma4hat  (Q,R,L) CP marginal vector of mode 4, for each r=1..R and state l=1..L
%                      * Ghat       (I,J,K,Q,L) coefficient tensor for state l=1..L
%                      * iphat      (L,1) vector of initial probabilities for the latent states
%                      * xihat      (L,L) transition matric of the hidden Markov chain
%                      * rhohat     (L,1) mixing probability for each state l=1..L
%                      * tauhat     (1,1) global variance of the gammas
%                      * what       (4,R,L) local variance of the gammas, for each r=1..R and state l=1..L
%                      * lambdahat  (L,1) hyperparameter for the local variance of the gammas, for each state l=1..L
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check inputs and set DEFAULT values for optional ones
arguments
   Xt (:,:,:,:) double                     % array (I,J,K,T) of (I,J,K) binary observations for each time t=1...T
   Zt (:,:) double                         % array (Q,T)     of Q covariates for each time t=1...T
   nameArgs.L        (1,1) double = 2      % number of states of the hidden chain
   nameArgs.R        (1,1) double = 5      % coefficient tensor CP rank
   nameArgs.NumIter  (1,1) double = 1000   % number of MCMC iterations to store
   nameArgs.burn     (1,1) double = 100    % number of initial iterations to thrash as burn-in phase
   nameArgs.thin     (1,1) double = 1      % thinning factor: store 1 iteration every 'thin' ones
   nameArgs.irep     (1,1) double = 10     % display cumulative computing time once every 'irep' iterations
   nameArgs.HYPER    struct                % structure with hyperparameter values
   nameArgs.INIT     struct                % structure with initial values
end

NumIter = nameArgs.NumIter;
burn    = nameArgs.burn;
thin    = nameArgs.thin;
irep    = nameArgs.irep;

% checks
if size(Xt,ndims(Xt)) ~= size(Zt,ndims(Zt))
   error('Xt must be (I,J,K,T) and Zt must be (Q,T).')
end

% Baseline parameters
TotIter = NumIter*thin + burn;
[I,J,K,T] = size(Xt);
Q         = size(Zt,1);
dims      = [I,J,K,Q];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Set HYPERPARAMETERS

if ~isfield(nameArgs,'HYPER')
   L = nameArgs.L;
   R = nameArgs.R;
   HYPER = define_hyperparameters(L,R);
else
   HYPER = nameArgs.HYPER;
   L = HYPER.L;
   R = HYPER.R;
end

% extract hyperparameters
alambda  = HYPER.alambda;
blambda  = HYPER.blambda;
alphabar = HYPER.alphabar;
ataubar  = HYPER.ataubar;
btaubar  = HYPER.btaubar;
cbar     = HYPER.cbar;
arhobar  = HYPER.arhobar;
brhobar  = HYPER.brhobar;
% Apply IDENTIFICATION constraint along MCMC iterations?
ident_constr = HYPER.ident_constr;
% CONSTRAINED version?
% gamma3hat(:,1:Rbar,ll) = repmat([1;0],1,Rbar);   gamma3hat(:,Rbar+1:end,ll) = repmat([0;1],1,Rbar); 
constr_g3 = HYPER.constr_g3;


% check
if constr_g3
   if ~mod(R,2) == 0
%       error('R must be an even integer.');  % Rbar = R / 2;
      R = R + 1;
      Rbar = R / 2;
      warning('In the constrained model, R should be an even integer --> now using (R+1).');
   end
end


%% INITIALIZATION

if ~isfield(nameArgs,'INIT')
   INIT = define_initial_values(Xt,Zt,HYPER);
else
   INIT = nameArgs.INIT;
end

% extract initial values
sthat     = INIT.sthat;
gamma1hat = INIT.gamma1hat;
gamma2hat = INIT.gamma2hat;
gamma3hat = INIT.gamma3hat;
gamma4hat = INIT.gamma4hat;
Ghat      = INIT.Ghat;
iphat     = INIT.iphat;
xihat     = INIT.xihat;
rhohat    = INIT.rhohat;
tauhat    = INIT.tauhat;
what      = INIT.what;
lambdahat = INIT.lambdahat;
Tl = cell(L,1);
for ll=1:L
   Tl{ll} = find(sthat == ll);
end

Ghat = double(Ghat);


%% MCMC sampler

%%% Allocate auxiliary variables
psihat = ones(R,1);     %dhat = zeros(I,J,K,T);
dhat = zeros(I*J*K,T);
u    = zeros(I*J*K,T);
uminusg = zeros(I*J*K,T);
acc_lambda = zeros(L,1);
post_mean_G = zeros(size(Ghat));

%%% Allocate OUTPUT in struct %%%%
OUT = struct;
OUT.Xt = Xt;      OUT.Zt = Zt;    OUT.R = R;   OUT.L = L;
OUT.NumIter = NumIter;   OUT.thin = thin;   OUT.burn = burn;
OUT.gamma1hat = zeros(I,R,L,NumIter);
OUT.gamma2hat = zeros(J,R,L,NumIter);
OUT.gamma3hat = zeros(K,R,L,NumIter);
OUT.gamma4hat = zeros(Q,R,L,NumIter);
OUT.sthat     = zeros(T,NumIter);
OUT.rhohat    = zeros(L,NumIter);

vecXt = zeros(I*J*K,T);
for t=1:T
   vecXt(:,t) = reshape(Xt(:,:,:,t),[],1);
end


relabel = 0;   % number of relabelled iterations
it = 0;

tms = zeros(TotIter,10);
tms_lab = {'st','PG','phi','tau','lambda','w','gammas','G','ident','store'};

t0=tic;
for iter=1:TotIter   
   %%%%%%%%%% Update s_t - FFBS %%%%%%%%%%
   tic;
   % Forward Filtering Backward Sampling --> Multi-move of Fruhwirth-Schnatter (2006), ch11.5
   [~,~,~,smoothp,sthat,~] = ffbsX_tensorData(Xt,Zt,rhohat,Ghat,xihat,iphat,0);
   iphat = smoothp(1,:)';     % update initial probability vector
   for ll=1:L
      Tl{ll} = find(sthat == ll);
   end
   tms(iter,1) = toc;
   
   
   tic;
   is = cell(1,T);    js = is;     ss = is;
   rows = 1:I*J*K;
   for t = 1:T
      %%%%%%%%%% Update omega_ij,t - PG %%%%%%%%%%
      % VECTORIZED PGvrnd. Need reshape, matrix IxJQL -> IxJxQ for l=state, j= all; q= all; l= sthat(t) -> 1:JQ or 1+JQ:2JQ
%       Glz   = double( ttv(Ghat(:,:,:,:,sthat(t)), Zt(:,t), 4) );   % select tensor for state t --> IxJQ
      Glz    = ttv_mod(Ghat(:,:,:,:,sthat(t)), Zt(:,t), 4);   % select tensor for state t --> IxJQ
      vecGlz = reshape(Glz,[],1);
      omegahat = PG1rnd00_mex(vecGlz);
%       omegahat = PGvrndCluster(1,1,vecGlz);
%       omegahat = reshape(PGvrndCluster(1,1,reshape(Glz,[],1)), [I,J,K]);
%       omegahat = PGvrndCluster(1,1,reshape(Glz,[],1));
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % speed SPARSE: create new block with values, add to previous (0)
%       rows= 1:I*J*K;     cols= 1+I*J*K*(t-1) : I*J*K*t;
%       addOmega = sparse(rows, cols, omegahat(:), I*J*K, I*J*K*T);
%       addOmega = sparse(rows, cols, omegahat, I*J*K, I*J*K*T);
%       Omegabarbar = Omegabarbar + addOmega;
      cols = 1+I*J*K*(t-1) : I*J*K*t;
      is{t} = rows;  js{t} = cols;  ss{t} = omegahat; % append "on the corner"
      %%%%%%%%%% Update d_ijk,t - inverse CDF %%%%%%%%%%
%       Pnotnorm1 =    rhohat(sthat(t))  * (Xt(:,:,:,t)==0);
%       Pnotnorm0 = (1-rhohat(sthat(t))) * ( (exp(Glz .* Xt(:,:,:,t))) ./ (1+exp(Glz)) );
%       Pnorm1 = Pnotnorm1 ./ (Pnotnorm1+Pnotnorm0);
%       dhat(:,:,:,t) = binornd(1,Pnorm1); % equivalent to:  un= rand(I,J); double(un < Pnorm1);
      Pnotnorm1 =    rhohat(sthat(t))  * (vecXt(:,t)==0);
      Pnotnorm0 = (1-rhohat(sthat(t))) * ( (exp(vecGlz .* vecXt(:,t))) ./ (1+exp(vecGlz)) );
      Pnorm1 = Pnotnorm1 ./ (Pnotnorm1+Pnotnorm0);
      dhat(:,t) = binornd(1,Pnorm1); % equivalent to:  un= rand(I,J); double(un < Pnorm1);
      %%% Necessary for:   gammas %%%
%       Kappa = (1-dhat(:,:,:,t)) .* (Xt(:,:,:,t)-1/2);
%       u(:,t) = reshape( Kappa ./ omegahat, [],1);
      Kappa = (1-dhat(:,t)) .* (vecXt(:,t)-1/2);
      u(:,t) = Kappa ./ omegahat;
   end
   Omegabarbar = sparse([is{:}], [js{:}], [ss{:}]);
   tms(iter,2) = toc;
   
   % Necessary for:   rhohat
   N1 = zeros(L,1);    N0 = zeros(L,1);    Ns = zeros(L,L);
   for ll=1:L
      N1(ll) = sum(sum( dhat(:,sthat==ll) ==1 ));
      N0(ll) = sum(sum( dhat(:,sthat==ll) ==0 ));
%       N1(ll) = sum(sum(sum(sum( dhat(:,:,:,sthat==ll) ==1 ))));
%       N0(ll) = sum(sum(sum(sum( dhat(:,:,:,sthat==ll) ==0 ))));
      for kk=1:L
         % number transitions each state (i,j) = i --> j
         Ns(ll,kk) = sum((sthat(1:T-1)==ll) & (sthat(2:T)==kk));
      end
      %%%%%%%%%% Update xi_l. - Dir %%%%%%%%%%
      xihat(ll,:) = dirrnd(cbar(ll,:) + Ns(ll,:), 1)';
   end
   % Used later for updating rho_l
	arpost = arhobar + N1;  % Lx1
   brpost = brhobar + N0;  % Lx1

   
   tic;
   %%%%%%%%%% Update phi_r - GiG %%%%%%%%%%
   pppost = alphabar-sum(dims);
   appost = 2*btaubar;
   if constr_g3
      for r=1:R
         bppost = 0;
         for ll=1:L
            bppost = bppost + (gamma1hat(:,r,ll)'*gamma1hat(:,r,ll)) / what(1,r,ll) ...
                            + (gamma2hat(:,r,ll)'*gamma2hat(:,r,ll)) / what(2,r,ll) ...
                            + (gamma4hat(:,r,ll)'*gamma4hat(:,r,ll)) / what(3,r,ll);
         end
         % Gibbs
         psihat(r) = gigrnd(pppost,appost,bppost,1);
      end
   else
      for r=1:R
         bppost = 0;
         for ll=1:L
            bppost = bppost + (gamma1hat(:,r,ll)'*gamma1hat(:,r,ll)) / what(1,r,ll) ...
                            + (gamma2hat(:,r,ll)'*gamma2hat(:,r,ll)) / what(2,r,ll) ...
                            + (gamma3hat(:,r,ll)'*gamma3hat(:,r,ll)) / what(3,r,ll) ...
                            + (gamma4hat(:,r,ll)'*gamma4hat(:,r,ll)) / what(4,r,ll);
         end
         % Gibbs
         psihat(r) = gigrnd(pppost,appost,bppost,1);
      end
   end
   phihat = psihat / sum(psihat);   % phi: normalise psi
   tms(iter,3) = toc;
   
   tic;
   %%%%%%%%%% Update tau - GiG %%%%%%%%%%
   ptpost = (alphabar-sum(dims))*R;
   atpost = 2*btaubar;
   if constr_g3
      pw1 = repmat(phihat,1,L) .* squeeze(what(1,:,:));
      pw2 = repmat(phihat,1,L) .* squeeze(what(2,:,:));
      pw4 = repmat(phihat,1,L) .* squeeze(what(3,:,:));
      btpost = 0;
      for r=1:R
         for ll=1:L
            btpost = btpost + (gamma1hat(:,r,ll)'*gamma1hat(:,r,ll)) / pw1(r,ll) + ...
                              (gamma2hat(:,r,ll)'*gamma2hat(:,r,ll)) / pw2(r,ll) + ...
                              (gamma4hat(:,r,ll)'*gamma4hat(:,r,ll)) / pw4(r,ll);
         end
      end
   else
      pw1 = repmat(phihat,1,L) .* squeeze(what(1,:,:));
      pw2 = repmat(phihat,1,L) .* squeeze(what(2,:,:));
      pw3 = repmat(phihat,1,L) .* squeeze(what(3,:,:));
      pw4 = repmat(phihat,1,L) .* squeeze(what(4,:,:));
      btpost = 0;
      for r=1:R
         for ll=1:L
            btpost = btpost + (gamma1hat(:,r,ll)'*gamma1hat(:,r,ll)) / pw1(r,ll) + ...
                              (gamma2hat(:,r,ll)'*gamma2hat(:,r,ll)) / pw2(r,ll) + ...
                              (gamma3hat(:,r,ll)'*gamma3hat(:,r,ll)) / pw3(r,ll) + ...
                              (gamma4hat(:,r,ll)'*gamma4hat(:,r,ll)) / pw4(r,ll);
         end
      end
   end
   % Gibbs
   tauhat = gigrnd(ptpost, atpost, btpost, 1);
   tms(iter,4) = toc;

   tic;
   if constr_g3
      what_lambda = what(1:3,:,:);
   else
      what_lambda = what(1:4,:,:);
   end
   for ll=1:L
      %%%%%%%%%% Update lambda_l - Ga %%%%%%%%%%
      eps_lmbd_hmc = 10^(-4);    L_lmbd = 10^4 *6.0;  % 3.0
      while (lambdahat(ll) <= 0) || ~isreal(acc_lambda(ll))
         [lambdahat(ll),acc_lambda(ll)] = HMClambda(lambdahat(ll),alambda(ll),blambda(ll),what_lambda(:,:,ll),...
                                                      1,eps_lmbd_hmc,L_lmbd,1,1);
%          [lambdahat(ll),acc_lambda(ll)] = HMClambda(lambdahat(ll),alambda(ll),blambda(ll),what(:,:,ll),...
%                                                     1,eps_lmbd_hmc,L_lmbd,1,1);
      end
   end
   tms(iter,5) = toc;
   
   tic;
   %%%%%%%%%% Update w_hrl - GiG %%%%%%%%%%
   pw1post = 1-I/2;    pw2post = 1-J/2;    pw3post = 1-K/2;    pw4post = 1-Q/2; % higher --> smaller out
   awpost  = lambdahat.^2;   % low influence
   tauphi  = tauhat .* phihat;
   for r=1:R
      for ll=1:L
         bw1postl = gamma1hat(:,r,ll)'*gamma1hat(:,r,ll) / tauphi(r);
         bw2postl = gamma2hat(:,r,ll)'*gamma2hat(:,r,ll) / tauphi(r);
         bw3postl = gamma3hat(:,r,ll)'*gamma3hat(:,r,ll) / tauphi(r);
         bw4postl = gamma4hat(:,r,ll)'*gamma4hat(:,r,ll) / tauphi(r);
         % Gibbs
         what(1,r,ll) = gigrnd(pw1post, awpost(ll), bw1postl, 1);   % NOTE: big b --> big output
         what(2,r,ll) = gigrnd(pw2post, awpost(ll), bw2postl, 1);
         if constr_g3
            what(3,r,ll) = gigrnd(pw4post, awpost(ll), bw4postl, 1);
         else
            what(3,r,ll) = gigrnd(pw3post, awpost(ll), bw3postl, 1);
         end
         what(4,r,ll) = gigrnd(pw4post, awpost(ll), bw4postl, 1);
      end
   end
   tms(iter,6) = toc;
   
   %%%% SAME timing as plain code
%    [gamma1hat,gamma2hat,gamma3hat,gamma4hat] = update_gammas(gamma1hat,gamma2hat,gamma3hat,gamma4hat,Zt,Tl,...
%                                                              Omegabarbar,u, tauhat,phihat,what);

   tic;
   %%%%%%%%%% Update gamma_hlr - N %%%%%%%%%%
   %%% necessary for all gamma_hlr %%%
   for r=1:R
      vvv = setdiff(1:R, r);   % (v < r) ALREADY updated -- (v > r) NOT uptdated yet
      for ll=1:L
      times_l = Tl{ll}';
%       GlrL1 = tenzeros(dims);   % build tensor, except marginal r,l
      % build tensor, except marginal r,l
      GlrL1 = ktensor_mod({gamma1hat(:,vvv,ll),gamma2hat(:,vvv,ll),gamma3hat(:,vvv,ll),gamma4hat(:,vvv,ll)});
%       for v=1:R
%          if (v ~= r)        % (v < r) ALREADY updated -- (v > r) NOT uptdated yet
% %             GlrL1 = GlrL1 + full( ktensor({gamma1hat(:,v,ll),gamma2hat(:,v,ll),gamma3hat(:,v,ll),gamma4hat(:,v,ll)}) );
%             GlrL1 = GlrL1 + ktensor_mod({gamma1hat(:,v,ll),gamma2hat(:,v,ll),gamma3hat(:,v,ll),gamma4hat(:,v,ll)});
%          end
%          if v < r        % ALREADY updated --> HAT
%             GlrL1 = GlrL1 + full( ktensor({gamma1hat(:,v,ll),gamma2hat(:,v,ll),gamma3hat(:,v,ll),gamma4hat(:,v,ll)}) );
%          elseif v > r    % NOT updated yet --> OLD
%             GlrL1 = GlrL1 + full( ktensor({gamma1hat(:,v,ll),gamma2hat(:,v,ll),gamma3hat(:,v,ll),gamma4hat(:,v,ll)}) );
%          end
%       end


      %%%%%%%%%% update gamma_1lr %%%%%%%%%%
      gam4Zt        = zeros(T,1);
      sumSigma1L1   = sparse(I,I);
      sumMuSigma1L1 = zeros(I,1);
      kron32L1      = sparse(kron(gamma3hat(:,r,ll), kron(gamma2hat(:,r,ll), eye(I))));
      S1likeInvL1   = sparse((tauhat * phihat(r) * what(1,r,ll) * eye(I)))^(-1);
      for t = times_l   % sum according to regime
%       for t=1:T
%          if (sthat(t) == ll)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % AUX variables necessary for gamma1, gamma2, gamma3
%             vecGlrZt = double( ttv(GlrL1, Zt(:,t), 4) );
            vecGlrZt = ttv_mod(GlrL1, Zt(:,t), 4);
            uminusg(:,t) = u(:,t) - vecGlrZt(:);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            gam4Zt(t) = gamma4hat(:,r,ll)' * Zt(:,t);
            tmp = Omegabarbar(:,1+I*J*K*(t-1):I*J*K*t) * kron32L1;
            sumSigma1L1   = sumSigma1L1   + (kron32L1' * tmp * gam4Zt(t)^2);
            sumMuSigma1L1 = sumMuSigma1L1 + (gam4Zt(t) * uminusg(:,t)' * tmp)';
%          end
      end
%       Sigma1postInvL1 = S1likeInvL1 + sumSigma1L1;
%       Sigma1postL1 = Sigma1postInvL1 \ eye(I);
%       mu1postL1    = Sigma1postInvL1 \ (S1likeInvL1*mu1prior(:,ll) + sumMuSigma1L1);
%       gamma1hat(:,r,ll) = mvnrnd(mu1postL1, Sigma1postL1);
      C = chol(S1likeInvL1 + sumSigma1L1,'lower');    % Cholesky of posterior precision matrix
      xhat = C \ sumMuSigma1L1;                       % posterior mean
      gamma1hat(:,r,ll) = C'\(xhat + randn(I,1));


      %%%%%%%%%% update gamma_2lr %%%%%%%%%%
      kron31L1 = sparse(kron(gamma3hat(:,r,ll), kron(eye(J), gamma1hat(:,r,ll))));
      sumSigma2L1   = sparse(J,J);
      sumMuSigma2L1 = zeros(J,1);
      S2likeInvL1   = sparse((tauhat * phihat(r) * what(2,r,ll) * eye(J)))^(-1);
      for t = times_l   % sum according to regime
%       for t=1:T
%          if (sthat(t) == ll)
            tmp = Omegabarbar(:, 1+I*J*K*(t-1):I*J*K*t) * kron31L1;
            sumSigma2L1   = sumSigma2L1   + (kron31L1' * tmp * gam4Zt(t)^2);
            sumMuSigma2L1 = sumMuSigma2L1 + (gam4Zt(t) * uminusg(:,t)' * tmp)';
%          end
      end
%       Sigma2postInvL1 = S2likeInvL1 + sumSigma2L1;
%       Sigma2postL1 = Sigma2postInvL1 \ eye(J);
%       mu2postL1    = Sigma2postInvL1 \ (S2likeInvL1*mu2prior(:,ll) + sumMuSigma2L1);
%       gamma2hat(:,r,ll) = mvnrnd(mu2postL1, Sigma2postL1);
      C = chol(S2likeInvL1 + sumSigma2L1,'lower');    % Cholesky of posterior precision matrix
      xhat = C \ sumMuSigma2L1;                       % posterior mean
      gamma2hat(:,r,ll) = C'\(xhat + randn(J,1));

      
      %%%%%%%%%% update gamma_3lr %%%%%%%%%%
      if ~constr_g3
         kron21L1 = sparse(kron(eye(K), kron(gamma2hat(:,r,ll), gamma1hat(:,r,ll))));
         sumSigma3L1   = sparse(K,K);
         sumMuSigma3L1 = zeros(K,1);
         S3likeInvL1   = sparse((tauhat * phihat(r) * what(3,r,ll) * eye(K)))^(-1);
         for t = times_l   % sum according to regime
   %       for t=1:T
   %          if (sthat(t) == ll)
               tmp = Omegabarbar(:, 1+I*J*K*(t-1):I*J*K*t) * kron21L1;
               sumSigma3L1   = sumSigma3L1   + kron21L1' * tmp * gam4Zt(t)^2;
               sumMuSigma3L1 = sumMuSigma3L1 + (gam4Zt(t) * uminusg(:,t)' * tmp)';
   %          end
         end
   %       Sigma3postInvL1 = S3likeInvL1 + sumSigma3L1;
   %       Sigma3postL1 = Sigma3postInvL1 \ eye(K);
   %       mu3postL1    = Sigma3postInvL1 \ (S3likeInvL1*mu3prior(:,ll) + sumMuSigma3L1);
   %       gamma3hat(:,r,ll) = mvnrnd(mu3postL1, Sigma3postL1);
         C = chol(S3likeInvL1 + sumSigma3L1,'lower');    % Cholesky of posterior precision matrix
         xhat = C \ sumMuSigma3L1;                       % posterior mean
         gamma3hat(:,r,ll) = C'\(xhat + randn(K,1));
      end
      

      %%%%%%%%%% update gamma_4lr %%%%%%%%%%
      sumSigma4L1   = zeros(Q,Q);
      sumMuSigma4L1 = zeros(Q,1);
      if constr_g3
         S4likeInvL1   = sparse((tauhat * phihat(r) * what(3,r,ll) * eye(Q)))^(-1);
      else
         S4likeInvL1   = sparse((tauhat * phihat(r) * what(4,r,ll) * eye(Q)))^(-1);
      end
      vec_gam123 = reshape( ktensor_mod({gamma1hat(:,r,ll),gamma2hat(:,r,ll),gamma3hat(:,r,ll)}), [], 1);
%       vec_gam123 = reshape( double(ktensor({gamma1hat(:,r,ll),gamma2hat(:,r,ll),gamma3hat(:,r,ll)})), [], 1);
      for t = times_l   % sum according to regime
%       for t=1:T
%          if (sthat(t) == ll)
%             Ztgam2 = Zt(:,t) * gamma2hat(:,r,ll)';
%             Sigma4Inv_rlt = Ztgam2 * kron31L1' * Omegabarbar(:,1+I*J*K*(t-1):I*J*K*t) * kron31L1 * Ztgam2';
%             mu4_rlt       = (uminusg(:,t)' * Omegabarbar(:,1+I*J*K*(t-1):I*J*K*t) * kron31L1 * Ztgam2')';
            tmp = Omegabarbar(:,1+I*J*K*(t-1):I*J*K*t) * vec_gam123 * Zt(:,t)';
            sumSigma4L1   = sumSigma4L1   + (Zt(:,t) * vec_gam123' * tmp);
            sumMuSigma4L1 = sumMuSigma4L1 + (uminusg(:,t)' * tmp)';
%          end
      end
%       Sigma4postInvL1 = S4likeInvL1 + sumSigma4L1;
%       Sigma4postL1 = MatPosDef(MatSym(Sigma4postInvL1 \ eye(Q)));
%       mu4postL1    = Sigma4postInvL1 \ (S4likeInvL1*mu4prior(:,ll) + sumMuSigma4L1);
%       gamma4hat(:,r,ll) = mvnrnd(mu4postL1, Sigma4postL1);
      C = chol(S4likeInvL1 + sumSigma4L1,'lower');    % Cholesky of posterior precision matrix
      xhat = C \ sumMuSigma4L1;                       % posterior mean
      gamma4hat(:,r,ll) = C'\(xhat + randn(Q,1));
      end


%       %%%%%%%%%% update gamma_1lr %%%%%%%%%%
%       gam4Zt        = zeros(T,1);
%       sumSigma1L1   = sparse(I,I);
%       sumMuSigma1L1 = zeros(I,1);
%       kron32L1      = sparse(kron(gamma3hat(:,r,ll), kron(gamma2hat(:,r,ll), eye(I))));
%       S1likeInvL1   = sparse((tauhat * phihat(r) * what(1,r,ll) * eye(I)))^(-1);
%       for t=1:T
%          % sum according to regime
%          if (sthat(t) == ll)
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             % AUX variables necessary for gamma1, gamma2, gamma3
%             vecGlrZt = double( ttv(GlrL1, Zt(:,t), 4) );
%             uminusg(:,t) = u(:,t) - vecGlrZt(:);
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             gam4Zt(t) = gamma4hat(:,r,ll)' * Zt(:,t);
%             sumSigma1L1   = sumSigma1L1   + (kron32L1' * Omegabarbar(:,1+I*J*K*(t-1):I*J*K*t) * kron32L1 * gam4Zt(t)^2);
%             sumMuSigma1L1 = sumMuSigma1L1 + (gam4Zt(t) * uminusg(:,t)' * Omegabarbar(:,1+I*J*K*(t-1):I*J*K*t) * kron32L1)';
%          end
%       end
% %       Sigma1postInvL1 = S1likeInvL1 + sumSigma1L1;
% %       Sigma1postL1 = Sigma1postInvL1 \ eye(I);
% %       mu1postL1    = Sigma1postInvL1 \ (S1likeInvL1*mu1prior(:,ll) + sumMuSigma1L1);
% %       gamma1hat(:,r,ll) = mvnrnd(mu1postL1, Sigma1postL1);
%       C = chol(S1likeInvL1 + sumSigma1L1,'lower');    % Cholesky of posterior precision matrix
%       xhat = C \ sumMuSigma1L1;                       % posterior mean
%       gamma1hat(:,r,ll) = C'\(xhat + randn(I,1));
% 
% 
%       %%%%%%%%%% update gamma_2lr %%%%%%%%%%
%       kron31L1 = sparse(kron(gamma3hat(:,r,ll), kron(eye(J), gamma1hat(:,r,ll))));
%       sumSigma2L1   = sparse(J,J);
%       sumMuSigma2L1 = zeros(J,1);
%       S2likeInvL1   = sparse((tauhat * phihat(r) * what(2,r,ll) * eye(J)))^(-1);
%       for t=1:T
%          % sum according to regime
%          if (sthat(t) == ll)
%             sumSigma2L1   = sumSigma2L1   + (kron31L1' * Omegabarbar(:, 1+I*J*K*(t-1):I*J*K*t) * kron31L1 * gam4Zt(t)^2);
%             sumMuSigma2L1 = sumMuSigma2L1 + (gam4Zt(t) * uminusg(:,t)' * Omegabarbar(:,1+I*J*K*(t-1):I*J*K*t) * kron31L1)';
%          end
%       end
% %       Sigma2postInvL1 = S2likeInvL1 + sumSigma2L1;
% %       Sigma2postL1 = Sigma2postInvL1 \ eye(J);
% %       mu2postL1    = Sigma2postInvL1 \ (S2likeInvL1*mu2prior(:,ll) + sumMuSigma2L1);
% %       gamma2hat(:,r,ll) = mvnrnd(mu2postL1, Sigma2postL1);
%       C = chol(S2likeInvL1 + sumSigma2L1,'lower');    % Cholesky of posterior precision matrix
%       xhat = C \ sumMuSigma2L1;                       % posterior mean
%       gamma2hat(:,r,ll) = C'\(xhat + randn(J,1));
% 
%       
%       %%%%%%%%%% update gamma_3lr -- FIXED %%%%%%%%%%
% %       gamma3hat(:,r,ll) = gamma3hat(:,r,ll);
%       
%       
%       %%%%%%%%%% update gamma_4lr %%%%%%%%%%
%       sumSigma4L1   = zeros(Q,Q);
%       sumMuSigma4L1 = zeros(Q,1);
%       S4likeInvL1   = sparse((tauhat * phihat(r) * what(3,r,ll) * eye(Q)))^(-1);
%       vec_gam123 = reshape( double(ktensor({gamma1hat(:,r,ll),gamma2hat(:,r,ll),gamma3hat(:,r,ll)})), [], 1);
%       for t=1:T
%          % sum according to regime
%          if (sthat(t) == ll)
% %             Ztgam2 = Zt(:,t) * gamma2hat(:,r,ll)';
% %             Sigma4Inv_rlt = Ztgam2 * kron31L1' * Omegabarbar(:,1+I*J*K*(t-1):I*J*K*t) * kron31L1 * Ztgam2';
% %             mu4_rlt       = (uminusg(:,t)' * Omegabarbar(:,1+I*J*K*(t-1):I*J*K*t) * kron31L1 * Ztgam2')';
%             sumSigma4L1   = sumSigma4L1   + (Zt(:,t) * vec_gam123' * Omegabarbar(:,1+I*J*K*(t-1):I*J*K*t) * vec_gam123 * Zt(:,t)');
%             sumMuSigma4L1 = sumMuSigma4L1 + (uminusg(:,t)' * Omegabarbar(:,1+I*J*K*(t-1):I*J*K*t) * vec_gam123 * Zt(:,t)')';
%          end
%       end
% %       Sigma4postInvL1 = S4likeInvL1 + sumSigma4L1;
% %       Sigma4postL1 = MatPosDef(MatSym(Sigma4postInvL1 \ eye(Q)));
% %       mu4postL1    = Sigma4postInvL1 \ (S4likeInvL1*mu4prior(:,ll) + sumMuSigma4L1);
% %       gamma4hat(:,r,ll) = mvnrnd(mu4postL1, Sigma4postL1);
%       C = chol(S4likeInvL1 + sumSigma4L1,'lower');    % Cholesky of posterior precision matrix
%       xhat = C \ sumMuSigma4L1;                       % posterior mean
%       gamma4hat(:,r,ll) = C'\(xhat + randn(Q,1));
%       end
   end
   tms(iter,7) = toc;

   tic;
   %%% compute norm,mean of reconstructed tensor %%%
   for ll=1:L
%       Ghat(:,:,:,:,ll) = full(ktensor({gamma1hat(:,:,ll),gamma2hat(:,:,ll),gamma3hat(:,:,ll),gamma4hat(:,:,ll)}));
      Ghat(:,:,:,:,ll) = ktensor_mod({gamma1hat(:,:,ll),gamma2hat(:,:,ll),gamma3hat(:,:,ll),gamma4hat(:,:,ll)});
   end
   tms(iter,8) = toc;
   
   
   %%%%%%%%%% Update rho_l - Be %%%%%%%%%%
   rhohat = betarnd(arpost,brpost);
   
   
   
   tic;
   %%%%%%%%%% IDENTIFICATION constraint %%%%%%%%%%
   if ident_constr
   % check identifcation constrainnt on mixture probability
   [rho_test, idx] = sort(rhohat,'descend');
   if (rho_test ~= rhohat)
      relabel = relabel + 1;
      
      % re-label states
      tmp = sthat;
      for ll=1:L
         tmp(sthat == ll) = find(idx == ll);
      end
      sthat = tmp;
      
      % re-arrange state-dependent parameters
      rhohat  = rhohat(idx);        % mixing probability
      xihat   = xihat(idx,:);       % transition matrix
      iphat   = iphat(idx);         % initial probability
%       smoothp = smoothp(:,idx);     % smoothed probabilities
      gamma1hat = gamma1hat(:,:,idx);  % CP marginals
      gamma2hat = gamma2hat(:,:,idx);
      gamma3hat = gamma3hat(:,:,idx);
      gamma4hat = gamma4hat(:,:,idx);
      Ghat      = Ghat(:,:,:,:,idx);   % whole tensor coefficient
%       normG     = normG(idx);          % nomr of tensor coefficient
      lambdahat = lambdahat(idx);   % hierarchial variance parameter
      what      = what(:,:,idx);    % hierarchial variance parameter
   end
   end
   tms(iter,9) = toc;
   
   
   tic;
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%% STORE current iteration %%%%
   if (iter > burn)   &&   (mod(iter,thin) == 0)
      it = it + 1;
      OUT.gamma1hat(:,:,:,it) = gamma1hat;
      OUT.gamma2hat(:,:,:,it) = gamma2hat;
      OUT.gamma3hat(:,:,:,it) = gamma3hat;
      OUT.gamma4hat(:,:,:,it) = gamma4hat;
      OUT.sthat(:,it)         = sthat;
      OUT.rhohat(:,it)        = rhohat;
      
      % compute post mean of coefficient tensor 'during' MCMC iterations
      post_mean_G = post_mean_G + Ghat;
%       if it == 1
%          post_mean_G = Ghat;
%       else
%          for ll=1:L
%             post_mean_G(:,:,:,:,ll) = Ghat(:,:,:,:,ll) / it  +  ((it-1)/it) * post_mean_G(:,:,:,:,ll);
%          end
%       end
   end
   tms(iter,10) = toc;
   
   %%% Display current iteration %%%
   if mod(iter,irep) == 0
      disp(['Iteration: ',num2str(iter),' of ',num2str(TotIter),'. Elapsed time is ',num2str(toc(t0)),' seconds.']);
   end
end
% save total computing time of MCMC
OUT.comp_time = ['Computing time = ',num2str(toc(t0)),' seconds.'];

OUT.tms_tab = array2table(tms,'VariableNames',tms_lab);

% compute posterior means
post_mean_G = post_mean_G / NumIter;
MAP_st = zeros(T,1);
for tt=1:T; [~,MAP_st(tt)] = max(histcounts(OUT.sthat(tt,:),[0:L+10]));  MAP_st(tt) = MAP_st(tt) - 1; end
OUT.MAP_st = MAP_st;
OUT.post_mean_G = post_mean_G;

end
