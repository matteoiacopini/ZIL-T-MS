function [Gammas] = SimAnnMScl_tensorData_edge_specific(DataX,DataZ,L,R,IterSA,state,rw,mu,sigma,k,pen1,pen0,penNorm,target_weight,cool)
%% SimAnnMScl_tensorData_edge_specific  Initialistion of gammas via Simulated Annealing, K > 1 and edge-specific covariates covariates
%
%  [GAMMAS] = SimAnnMScl_tensorData_edge_specific(DATAX,DATAZ,L,R,ITETRSA,STATE,RW,MU,SIGMA,K,PEN1,PEN0,PENNORM,TARGET_WEIGHT,COOL)
%  Runs a Simulated Annealing algorithm to inittialize the PARAFAC marginals, GAMMAS,
%  for a ZIL-T-MS model. For each t=1,...,T, the observed binary arrays (DATAX) are 
%  of size (I,J,K) and the covariates (DATAZ) are of size (I,J,K,Q).
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
%   DataX   IxJxKxT   observed binary matrix Xt for each time
%   DataZ   IxJxKxQxT observed vector of covariates
%   L       1x1       #regimes
%   R       1x1       tensor rank
%   iterSA  1x1       number iterations Simulated Annealing
%   state   Tx1       hidden chain
%   rw      2x1       1 = random walk proposal
%   sigma   1x1       variance proposal
%   k       1x1       constant in temperature
%   pen1    1x1       penalty for x=1 missed
%   pen0    1x1       penalty for x=0 missed
%   penNorm 1x1       penalty for sum( norm-1( gamma123 ) )
%   cool    1x1       cooling schedule (log, loglog)
% OUTPUT
%   Gammas= {4}(gamma1, gamma2, gamma3, gamma4)  marginals   IxRxL JxRxL KxRxL QxRxL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % Allocate variables
   if ismatrix(DataZ)
      Q = size(DataZ,1);
   elseif ndims(DataZ) == 5
      Q = size(DataZ,4);
   end
   [I,J,K,T] = size(DataX);    dims = [I,J,K,Q];
   gamma1Init = zeros(dims(1), R, L);      g1 = NaN(dims(1), R, L);
   gamma2Init = zeros(dims(2), R, L);      g2 = NaN(dims(2), R, L);
   gamma3Init = zeros(dims(3), R, L);      g3 = NaN(dims(3), R, L);
   gamma4Init = zeros(dims(4), R, L);      g4 = NaN(dims(4), R, L);
   gamma1Sim  = zeros(dims(1), R, L);
   gamma2Sim  = zeros(dims(2), R, L);
   gamma3Sim  = zeros(dims(3), R, L);
   gamma4Sim  = zeros(dims(4), R, L);
%    alpha = NaN(IterSA,L);    accproba = NaN(IterSA,L);    accepted = NaN(IterSA,L);   seriesf = NaN(IterSA,L);
%    fn = NaN(IterSA,L);    fo = NaN(IterSA,L);     qn = NaN(IterSA,L);     qo = NaN(IterSA,L);
   GOld = NaN(I,J,K,T);     HOld = NaN(I,J,K,T);      GNew = NaN(I,J,K,T);      HNew = NaN(I,J,K,T);
   fold = NaN(L,1);       fnew = NaN(L,1);        Times = NaN(L,1);       % edgest= NaN(T,1);
   logPropOld = zeros(L,1);      logPropNew = zeros(L,1);
   Tims = zeros(L,1);
   
   % PARAMETERS SimAnn: mean proposal
   mu1 = NaN(dims(1),L);    mu2 = NaN(dims(2),L);    mu3 = NaN(dims(3),L);    mu4 = NaN(dims(4),L);
   for ll=1:L
      Times(ll) = sum(state == ll);
      mu1(:,ll) = mu(ll) * ones(dims(1),1);
      mu2(:,ll) = mu(ll) * ones(dims(2),1);
      mu3(:,ll) = mu(ll) * ones(dims(3),1);
      mu4(:,ll) = mu(ll) * ones(dims(4),1);
   end
   
   % TEMPERATURE T(j) (decreasing): implies --> Dirac, needed for sampling in set close to mean
   if strcmp(cool,'log')
      Temp = k * 1./( 1+log(1:IterSA) );  % 1/iter    1/(1+log(iter))
   elseif strcmp(cool,'loglog')
      Temp = k * 1./( 1+log(1+log(1:IterSA)) );
   end
   
   
   %%%%%%%%%%%%%%%% SIMULATED ANNEALING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
   %%%% Dynamic Plot of distance + acc rates %%%%
%    g = figure(800);   g = uipanel('Parent',g,'BorderType','none');   g.Title ='Simulated Annealing';
%    g.TitlePosition = 'centertop';    g.FontSize = 12;    g.FontWeight = 'bold';
%    subplot(2,2,1,'Parent',g);  h11=plot(seriesf(1,1),'Color',[1,0.55,0]);  title('Sparse $(s_t=1)$: function value');
%    hold on; h11b=plot(accepted(1,1),'r*'); hold off;
%    subplot(2,2,3,'Parent',g);  h21=plot(seriesf(1,2),'Color',[1,0.55,0]);  title('Dense $(s_t=2)$: function value');
%    hold on; h21b=plot(accepted(1,2),'r*'); hold off;
%    subplot(2,2,2,'Parent',g);  h12=plot(cumsum(accproba(1,1))./(1)');  ylim([0,1]);
%    title('Sparse $(s_t=1)$: acceptance probability');
%    subplot(2,2,4,'Parent',g);  h22=plot(cumsum(accproba(1,2))./(1)');  ylim([0,1]);
%    title('Dense $(s_t=2)$: acceptance probability');
%    set(gcf,'DoubleBuffer','On');    % Prevents flickering
   

   ttt = tic; % start timing

   for iter=1:IterSA
      if ~mod(iter,100)
         fprintf('Iteration #%d\n',iter);
      end
      
      %%%%%%%%%%%% first iteration: initialization of B_sa %%%%%%%%%%%%
      if iter==1
         lpg1 = 0;   lpg2 = 0;   lpg3 = 0;   lpg4 = 0;
         for l=1:L
            normgs = 0;
            for s=1:R
               if rw(l)==0    % INDEPENDENT proposal: iter(0, s*I)
                  g1(:,s,l) = mvnrnd(mu1(:,l), sigma(l) * eye(dims(1)), 1);
                  g2(:,s,l) = mvnrnd(mu2(:,l), sigma(l) * eye(dims(2)), 1);
                  g3(:,s,l) = mvnrnd(mu3(:,l), sigma(l) * eye(dims(3)), 1);
                  g4(:,s,l) = mvnrnd(mu4(:,l), sigma(l) * eye(dims(4)), 1);
                  lpg1 = lpg1 + log(mvnpdf(g1(:,s,l), mu1(:,l), sigma(l) * eye(dims(1))));
                  lpg2 = lpg2 + log(mvnpdf(g2(:,s,l), mu2(:,l), sigma(l) * eye(dims(2))));
                  lpg3 = lpg3 + log(mvnpdf(g3(:,s,l), mu3(:,l), sigma(l) * eye(dims(3))));
                  lpg4 = lpg4 + log(mvnpdf(g4(:,s,l), mu4(:,l), sigma(l) * eye(dims(4))));
                  % computes (log version of) proposal density
                  logPropOld(l) = lpg1 + lpg2 + lpg3 + lpg4;
               else           % RANDOM WALK proposal: iter(mu, s*I)
                  g1(:,s,l) = mvnrnd(mu1(:,l), sigma(l) * eye(dims(1)), 1);
                  g2(:,s,l) = mvnrnd(mu2(:,l), sigma(l) * eye(dims(2)), 1);
                  g3(:,s,l) = mvnrnd(mu3(:,l), sigma(l) * eye(dims(3)), 1);
                  g4(:,s,l) = mvnrnd(mu4(:,l), sigma(l) * eye(dims(4)), 1);
               end
               normgs = normgs + norm(g1(:,s,l),1) + norm(g2(:,s,l),1) + norm(g3(:,s,l),1) + norm(g4(:,s,l),1);
            end
            Gsa   = double( ktensor({g1(:,:,l), g2(:,:,l), g3(:,:,l), g4(:,:,l)}) );
            normG = fnorm(Gsa -target_weight(l)*ones(dims),1);
            Gsa   = tensor(Gsa);
            % store initial and compute tensor
            gamma1Init(:,:,l) = g1(:,:,l);
            gamma2Init(:,:,l) = g2(:,:,l);
            gamma3Init(:,:,l) = g3(:,:,l);
            gamma4Init(:,:,l) = g4(:,:,l);
            fsaOld = zeros(I,J,K,T);
            for t=1:T
               if (state(t) == l)
%                   GOld(:,:,:,t) = double( ttv(Gsa, DataZ(:,:,:,:,t), 4) );
                  GOld(:,:,:,t) = double( ttv(Gsa .* DataZ(:,:,:,:,t), ones(Q,1), 4) );
                  HOld(:,:,:,t) = DataX(:,:,:,t) - double( GOld(:,:,:,t) > 0 );
                  pos1Old = (HOld(:,:,:,t) ==  1);
                  pos0Old = (HOld(:,:,:,t) == -1);
                  fsaOld(pos1Old,t) = pen1;
                  fsaOld(pos0Old,t) = pen0;
                  fsaOld(:,:,:,t) = fsaOld(:,:,:,t) + penNorm*normG; % normgs
               end
            end
            fold(l) = sum(fsaOld(:)) / (I*J*K*Times(l));
         end
      end
      
      
      %%% all iterations > 1 %%%
      lpg1 = 0;   lpg2 = 0;   lpg3 = 0;   lpg4 = 0;
      for l=1:L
         normgs = 0;
         for s=1:R
            if rw(l)==0    % INDPENDENT proposal
               g1(:,s,l) = mvnrnd(mu1(:,l), sigma(l) * eye(dims(1)), 1);
               g2(:,s,l) = mvnrnd(mu2(:,l), sigma(l) * eye(dims(2)), 1);
               g3(:,s,l) = mvnrnd(mu3(:,l), sigma(l) * eye(dims(3)), 1);
               g4(:,s,l) = mvnrnd(mu4(:,l), sigma(l) * eye(dims(4)), 1);
               lpg1 = lpg1 + log(mvnpdf(g1(:,s,l), mu1(:,l), sigma(l) * eye(dims(1))));
               lpg2 = lpg2 + log(mvnpdf(g2(:,s,l), mu2(:,l), sigma(l) * eye(dims(2))));
               lpg3 = lpg3 + log(mvnpdf(g3(:,s,l), mu3(:,l), sigma(l) * eye(dims(3))));
               lpg4 = lpg4 + log(mvnpdf(g4(:,s,l), mu4(:,l), sigma(l) * eye(dims(4))));
               % computes (log version of) proposal density
               logPropNew(l) = lpg1 + lpg2 + lpg3 + lpg4;
            else           % RANDOM WALK proposal
               g1(:,s,l) = mvnrnd(gamma1Init(:,s,l), sigma(l) * eye(dims(1)), 1);
               g2(:,s,l) = mvnrnd(gamma2Init(:,s,l), sigma(l) * eye(dims(2)), 1);
               g3(:,s,l) = mvnrnd(gamma3Init(:,s,l), sigma(l) * eye(dims(3)), 1);
               g4(:,s,l) = mvnrnd(gamma4Init(:,s,l), sigma(l) * eye(dims(4)), 1);
            end
            normgs = normgs + norm(g1(:,s,l),1) + norm(g2(:,s,l),1) + norm(g3(:,s,l),1) + norm(g4(:,s,l),1);
         end
         Gnew  = double( ktensor({g1(:,:,l), g2(:,:,l), g3(:,:,l), g4(:,:,l)}) );
         normG = fnorm(Gnew -target_weight(l)*ones(dims),1);
         Gnew  = tensor(Gnew);
         % store proposal and compute tensor
         gamma1Sim(:,:,l) = g1(:,:,l);
         gamma2Sim(:,:,l) = g2(:,:,l);
         gamma3Sim(:,:,l) = g3(:,:,l);
         gamma4Sim(:,:,l) = g4(:,:,l);
         fsaNew = zeros(I,J,K,T);
         for t=1:T
            if (state(t) == l)
%                GNew(:,:,:,t) = double( ttv(Gnew, DataZ(:,t), 3) );
               GNew(:,:,:,t) = double( ttv(Gnew .* DataZ(:,:,:,:,t), ones(Q,1), 4) );
               HNew(:,:,:,t) = DataX(:,:,:,t) - double( GNew(:,:,:,t) > 0 );
               pos1New = (HNew(:,:,:,t) ==  1);
               pos0New = (HNew(:,:,:,t) == -1);
               fsaNew(pos1New,t) = pen1;
               fsaNew(pos0New,t) = pen0;
               fsaNew(:,:,:,t) = fsaNew(:,:,:,t) + penNorm*normG; % normgs
            end
         end
         fnew(l) = sum(fsaNew(:)) / (I*J*K*Times(l));


         %%%%%%%% ACCEPTANCE RATIO %%%%%%%%
         % store current iteration values
%          fn(iter,l) = fnew(l);          fo(iter,l) = fold(l);
%          qn(iter,l) = logPropNew(l);    qo(iter,l) = logPropOld(l);
         % acceptance probability
         alpha    = exp(-(fnew(l) - fold(l)) / Temp(iter) + logPropOld(l) - logPropNew(l));
         accproba = min(1, alpha);

         if rand(1) < accproba    % draws rand at each l
            % ACCEPT MOVE --> update parameters
%             accepted(iter,l) = 1;
            fold(l) = fnew(l);                       % update function
            gamma1Init(:,:,l) = gamma1Sim(:,:,l);    % update gammas
            gamma2Init(:,:,l) = gamma2Sim(:,:,l);
            gamma3Init(:,:,l) = gamma3Sim(:,:,l);
            gamma4Init(:,:,l) = gamma4Sim(:,:,l);
            logPropOld(l) = logPropNew(l);   % (RW case) log proposal density
         end
         % store current state of function
%          seriesf(iter,l) = fold(l);
      end
      
      %%%% Update PLOT %%%%
%       set(h11,'Xdata',1:iter,'Ydata',seriesf(1:iter,1));
%       set(h11b,'Xdata',1:iter,'Ydata',accepted(1:iter,1).*seriesf(1:iter,1));
%       set(h21,'Xdata',1:iter,'Ydata',seriesf(1:iter,2));
%       set(h21b,'Xdata',1:iter,'Ydata',accepted(1:iter,2).*seriesf(1:iter,2));
%       set(h12,'Xdata',1:iter,'Ydata',cumsum(accproba(1:iter,1))./(1:iter)');
%       set(h22,'Xdata',1:iter,'Ydata',cumsum(accproba(1:iter,2))./(1:iter)');
%       drawnow;
   end
   
   % SAVE marginals (gamma1, gamma2, gamma3)
   Gammas = cell(4,1);
   Gammas{1} = gamma1Init;   % IxRxL
   Gammas{2} = gamma2Init;   % JxRxL
   Gammas{3} = gamma3Init;   % KxRxL
   Gammas{4} = gamma4Init;   % QxRxL
   
   fprintf('Time Simulated Annealing  %.2f min\n',toc(ttt)/60);

end
