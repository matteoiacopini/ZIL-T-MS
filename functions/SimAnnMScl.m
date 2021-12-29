function [Gammas] = SimAnnMScl(DataX,DataZ,L,R,IterSA,state,rw,mu,sigma,k,pen1,pen0,penNorm,target_weight,cool)
%% SimAnnMScl  Initialization of gammas via Simulated Annealing, K=1 and common covariates
%
%  [GAMMAS] = SimAnnMScl(DATAX,DATAZ,L,R,ITETRSA,STATE,RW,MU,SIGMA,K,PEN1,PEN0,PENNORM,TARGET_WEIGHT,COOL)
%  Runs a Simulated Annealing algorithm to inittialize the PARAFAC marginals, GAMMAS,
%  for a ZIL-T-MS model. For each t=1,...,T, the observed binary arrays (DATAX) are 
%  of size (I,J) and the covariates (DATAZ) are of size (Q,1).
%  The PARAFAC rank is R and the number of states of the hidden chain (provided in 
%  STATES) is L.
%  The Simulated Annealing is run for ITERSA iterations, with a cooling scheme
%  defined by the string COOL (log, loglog) and scale K.
%  The hyperparameters (RW,MU,SIGMA) determine the type of proposal, its mean and
%  variance.
%  The hyperparameters (PEN1,PEN0,PENNORM,TARGET_WEIGHT) represent penalties that
%  determine the target function.
%
% INPUTS
%   DataX   IxJxT    observed binary matrix Xt for each time
%   DataZ   QxT      observed vector of covariates
%   L       1x1      #regimes
%   R       1x1      tensor rank
%   iterSA  1x1      number iterations Simulated Annealing
%   state   Tx1      hidden chain
%   rw      2x1      1 = random walk proposal
%   sigma   1x1      variance proposal
%   k       1x1      constant in temperature
%   pen1    1x1      penalty for x=1 missed
%   pen0    1x1      penalty for x=0 missed
%   penNorm 1x1      penalty for sum( norm-1( gamma123 ) )
%   cool    1x1      cooling schedule (log, loglog)
%
% OUTPUT
%   Gammas= {3}(gamma1, gamma2, gamma3)=  marginals   IxRxL JxRxL QxRxL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % Allocate variables
   if ismatrix(DataZ)
      Q = size(DataZ,1);
   elseif ndims(DataZ) == 4
      Q = size(DataZ,3);
   end
   [I,J,T] = size(DataX);         dims = [I,J,Q];
   gamma1Init = zeros(dims(1), R, L);
   gamma2Init = zeros(dims(2), R, L);
   gamma3Init = zeros(dims(3), R, L);
   gamma1Sim  = zeros(dims(1), R, L);
   gamma2Sim  = zeros(dims(2), R, L);
   gamma3Sim  = zeros(dims(3), R, L);
%    alpha = NaN(IterSA,L);    accproba = NaN(IterSA,L);    accepted = NaN(IterSA,L);   seriesf = NaN(IterSA,L);
%    fn = NaN(IterSA,L);    fo = NaN(IterSA,L);     qn = NaN(IterSA,L);     qo = NaN(IterSA,L);
%    GOld = NaN(I,J,T);     HOld = NaN(I,J,T);      GNew = NaN(I,J,T);      HNew = NaN(I,J,T);
   fold = NaN(L,1);       fnew = NaN(L,1);        Times = NaN(L,1);       % edgest= NaN(T,1);
   logPropOld = zeros(L,1);      logPropNew = zeros(L,1);
%    Tims = zeros(L,1);
   
   % PARAMETERS SimAnn: mean proposal
   mu1 = NaN(dims(1),L);    mu2 = NaN(dims(2),L);    mu3 = NaN(dims(3),L);
   Tl = cell(L,1);
   for ll=1:L
      Times(ll) = sum(state == ll);
      Tl{ll}  = find(state == ll);
      mu1(:,ll) = mu(ll) * ones(dims(1),1);
      mu2(:,ll) = mu(ll) * ones(dims(2),1);
      mu3(:,ll) = mu(ll) * ones(dims(3),1);
   end
   
   % TEMPERATURE T(j) (decreasing): implies --> Dirac, needed for sampling in set close to mean
   if strcmp(cool,'log')
      Temp = k * 1./( 1+log(1:IterSA) );  % 1/iter    1/(1+log(iter))
   elseif strcmp(cool,'loglog')
      Temp = k * 1./( 1+log(1+log(1:IterSA)) );
   end
   
   
   %%% SIMULATED ANNEALING %%%

   ttt = tic; % start timing

   for iter=1:IterSA
      if ~mod(iter,100)
         fprintf('Iteration #%d\n',iter);
      end
      
      %%%%%%%%%%%% first iteration: initialization of B_sa %%%%%%%%%%%%
      if iter==1
         lpg1 = 0;   lpg2 = 0;   lpg3 = 0;
         for l=1:L
            normgs = 0;
            for s=1:R
               if rw(l)==0    % INDEPENDENT proposal: iter(0, s*I)
                  gamma1Init(:,s,l) = mvnrnd(mu1(:,l), sigma(l) * eye(dims(1)), 1);
                  gamma2Init(:,s,l) = mvnrnd(mu2(:,l), sigma(l) * eye(dims(2)), 1);
                  gamma3Init(:,s,l) = mvnrnd(mu3(:,l), sigma(l) * eye(dims(3)), 1);
                  lpg1 = lpg1 + log(mvnpdf(gamma1Init(:,s,l), mu1(:,l), sigma(l) * eye(dims(1))));
                  lpg2 = lpg2 + log(mvnpdf(gamma2Init(:,s,l), mu2(:,l), sigma(l) * eye(dims(2))));
                  lpg3 = lpg3 + log(mvnpdf(gamma3Init(:,s,l), mu3(:,l), sigma(l) * eye(dims(3))));
                  % computes (log version of) proposal density
                  logPropOld(l) = lpg1 + lpg2 + lpg3;
               else           % RANDOM WALK proposal: iter(mu, s*I)
                  gamma1Init(:,s,l) = mvnrnd(mu1(:,l), sigma(l) * eye(dims(1)), 1);
                  gamma2Init(:,s,l) = mvnrnd(mu2(:,l), sigma(l) * eye(dims(2)), 1);
                  gamma3Init(:,s,l) = mvnrnd(mu3(:,l), sigma(l) * eye(dims(3)), 1);
               end
               normgs = normgs + norm(gamma1Init(:,s,l),1) + norm(gamma2Init(:,s,l),1) + norm(gamma3Init(:,s,l),1);
            end
%             Gsa   = double( ktensor({gamma1Init(:,:,l), gamma2Init(:,:,l), gamma3Init(:,:,l)}) );
            Gsa   = ktensor_mod({gamma1Init(:,:,l), gamma2Init(:,:,l), gamma3Init(:,:,l)});
            normG = fnorm(Gsa -target_weight(l)*ones(dims),1);
%             Gsa   = tensor(Gsa);
            % store initial and compute tensor
            fsaOld = zeros(I,J,T);
            for t = Tl{ll}'
%             for t=1:T
%                if (state(t) == l)
%                   GOld(:,:,t) = double( ttv(Gsa, DataZ(:,t), 3) );
                  GOld = ttv_mod(Gsa, DataZ(:,t), 3);
                  HOld = DataX(:,:,t) - double( GOld > 0 );
                  fsaOld(HOld ==  1, t) = pen1;
                  fsaOld(HOld == -1, t) = pen0;
%                   fsaOld(:,:,t) = fsaOld(:,:,t) + penNorm*normG; % normgs
%                end
            end
%             fold(l) = sum(fsaOld(:)) / (I*J*Times(l));
            fold(l) = (sum(fsaOld(:)) / (I*J*Times(l))) + penNorm*normG; % normgs
         end
      end
      
      
      %%% all iterations > 1 %%%
      lpg1 = 0;   lpg2 = 0;   lpg3 = 0;
      for l=1:L
         normgs = 0;
         for s=1:R
            if rw(l)==0    % INDPENDENT proposal
               gamma1Sim(:,s,l) = mvnrnd(mu1(:,l), sigma(l) * eye(dims(1)), 1);
               gamma2Sim(:,s,l) = mvnrnd(mu2(:,l), sigma(l) * eye(dims(2)), 1);
               gamma3Sim(:,s,l) = mvnrnd(mu3(:,l), sigma(l) * eye(dims(3)), 1);
               lpg1 = lpg1 + log(mvnpdf(gamma1Sim(:,s,l), mu1(:,l), sigma(l) * eye(dims(1))));
               lpg2 = lpg2 + log(mvnpdf(gamma2Sim(:,s,l), mu2(:,l), sigma(l) * eye(dims(2))));
               lpg3 = lpg3 + log(mvnpdf(gamma3Sim(:,s,l), mu3(:,l), sigma(l) * eye(dims(3))));
               % computes (log version of) proposal density
               logPropNew(l) = lpg1 + lpg2 + lpg3;
            else           % RANDOM WALK proposal
               gamma1Sim(:,s,l) = mvnrnd(gamma1Init(:,s,l), sigma(l) * eye(dims(1)), 1);
               gamma2Sim(:,s,l) = mvnrnd(gamma2Init(:,s,l), sigma(l) * eye(dims(2)), 1);
               gamma3Sim(:,s,l) = mvnrnd(gamma3Init(:,s,l), sigma(l) * eye(dims(3)), 1);
            end
            normgs = normgs + norm(gamma1Sim(:,s,l),1) + norm(gamma2Sim(:,s,l),1) + norm(gamma3Sim(:,s,l),1);
         end
%          Gnew  = double( ktensor({gamma1Sim(:,:,l), gamma2Sim(:,:,l), gamma3Sim(:,:,l)}) );
         Gnew  = ktensor_mod({gamma1Sim(:,:,l), gamma2Sim(:,:,l), gamma3Sim(:,:,l)});
         normG = fnorm(Gnew -target_weight(l)*ones(dims),1);
%          Gnew  = tensor(Gnew);
         % store proposal and compute tensor
         fsaNew = zeros(I,J,T);
         for t = Tl{ll}'
%          for t=1:T
%             if (state(t) == l)
%                GNew = double( ttv(Gnew, DataZ(:,t), 3) );
               GNew = ttv_mod(Gnew, DataZ(:,t), 3);
               HNew = DataX(:,:,t) - double( GNew > 0 );
               fsaNew(HNew ==  1, t) = pen1;
               fsaNew(HNew == -1, t) = pen0;
%                fsaNew(:,:,t) = fsaNew(:,:,t) + penNorm*normG; % normgs
%             end
         end
%          fnew(l) = sum(fsaNew(:)) / (I*J*Times(l));
         fnew(l) = (sum(fsaNew(:)) / (I*J*Times(l)))  +  penNorm*normG;


         % acceptance probability
         alpha    = exp(-(fnew(l) - fold(l)) / Temp(iter) + logPropOld(l) - logPropNew(l));
         accproba = min(1, alpha);

         if rand(1) < accproba
%             accepted(iter,l) = 1;
            fold(l) = fnew(l);                       % update function
            gamma1Init(:,:,l) = gamma1Sim(:,:,l);    % update gammas
            gamma2Init(:,:,l) = gamma2Sim(:,:,l);
            gamma3Init(:,:,l) = gamma3Sim(:,:,l);
            logPropOld(l) = logPropNew(l);   % (RW case) log proposal density
         end
      end
   end
   
   % SAVE marginals (gamma1, gamma2, gamma3)
   Gammas = cell(3,1);
   Gammas{1} = gamma1Init;   % IxRxL
   Gammas{2} = gamma2Init;   % JxRxL
   Gammas{3} = gamma3Init;   % QxRxL
   
   fprintf('Time Simulated Annealing  %.2f min\n',toc(ttt)/60);

end
