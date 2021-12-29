function [lkl,filtp,filts,smoothp,smooths,clkl] = ffbsX_tensorData_edge_specific(Xt,Zt,rho,G,P,ip,scale_clkl)
%% ffbsX_tensorData_edge_specific Forward-Filtering Backward-Sampling for ZIL-MS binary tensor model with tensor observations and edge-specific covariates
%
% Samples trajectory (s1,...,sT):
% 1) computes conditional log-likelihood (specific for each model)
% 2) runs FFBS algorithm: FF forward filtering; BS backward smoothing and sampling
%
%   [LKL,FILTP,FILTS,SMOOTHP,SMOOTHS,CLKL] = ffbsX(XT,ZT,RHO,G,P,IP,SCALE_CLKL)
%   Applies the Forward-Filtering Backward-Sampling algorithm to estimate the hidden
%   Markov chain of the ZIL-T-MS model.
%   Takes as inputs the observed binary arrays XT and covariates ZT, the values of the
%   coefficient tensor, G, and the ZIL mixing probability, RHO, in each state l=1,...,L,
%   and the transiton matrix P and initial probability IP.
%   Returns the filtered and smoothed probabilities (FILTP, SMOOTHP), the 
%   filtered and smoothed states (FILTS, SMOOTHS), the log-likelihood (LKL), and
%   the conditional likelihood values in each state (CLKL).
%   If SCALE_CLKL == 1, a correction is applied to CLKL to deal with large differences
%   among the entris of CLKL.
%
% INPUTS
%  Xt   IxJxKxT   --> xt IxJxK  tensor observation
%  Zt   QxT       --> zt Qx1  vector covariates
%  G    IxJxKxQxL --> G(:,:,:,:,l) tensor IxJxKxQ
%  P    LxL       --> transition probabilities
%  ipm  Lx1       --> initial probabilities
%  scale_clkl = 1 --> rescale clkl until difference proba between states < 2.5
% OUTPUTS
%  lkl      Tx1 unconditional log-likelihood   log(p(Xt,yt|Xt-1,yt-1,..,X1,y1))
%  filtp    TxL filtered probabilities
%  filts    Tx1 filtered states
%  smooths  Tx1 smoothed states (smooth(t)= 1..L)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Regimes cycle: computes log-conditional likelihood, then 'Hamilton' for FFBS   
[~,~,~,T] = size(Xt);      % T = length time series (observations)
L = size(P,1);             % L = #regimes
clkl = zeros(T,L);
X1 = Xt==1;
X0 = Xt==0;
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%% CONDITIONAL LOG-LIKELIHOOD (depends on model) %%%%%%%%%%%%%%
   %             ( p(X1,y1|s1=1) ... p(X1,y1|s1=L) )
   %   clkl= log (       ...               ...     )
   %             ( p(XT,yT|sT=1) ... p(XT,yT|sT=L) )
   for t=1:T
      for l=1:L
         %%% conditional likelihood (Xt|st,-) -- discrete 0-1 %%%
%          Gz = double( ttv(G(:,:,:,:,l), Zt(:,t), 4) );
         Gz = sum( double(G(:,:,:,:,l) .* Zt(:,:,:,:,t)), 4);
         % CORRECT (equivalent) for big exponent:
         % log(1+X) = log((1+X)/X *X) = log(1 + 1/X) + log(X) = log(1+1/exp(Gz))+Gz
         psmall = Gz <= 15;      Gzsmall = Gz.*psmall;
         pbig   = Gz >  15;      %Gzbig   = Gz.*pbig;
         if sum(pbig(:)) > 0
            % P(x_ij = 1|-) = (1-rho) * exp(Gz)/(1+exp(Gz))
            % P(x_ij = 0|-) = rho + (1-rho) * 1/(1+exp(Gz)) = (1+rho*exp(Gz)) / (1+exp(Gz))
            expGxsmall = exp(Gzsmall);
            logden = log(1+expGxsmall).*psmall + Gz.*pbig;
            logpX0 = log(1+rho(l)*expGxsmall).*psmall + (log(rho(l))+Gz).*pbig - logden;
         else
            % P(x_ij = 1|-) = (1-rho) * exp(Gz)/(1+exp(Gz))
            % P(x_ij = 0|-) = rho + (1-rho) * 1/(1+exp(Gz)) = (1+rho*exp(Gz)) / (1+exp(Gz))
            expGx = exp(Gz);
            logden = log(1+expGx);
            logpX0 = log(1+rho(l)*expGx) - logden;
         end
         logpX1 = log(1-rho(l)) + Gz - logden;
         % conditional likelihood (Xt|st,-)
         clkl(t,l) = sum(logpX1(X1(:,:,:,t))) + sum(logpX0(X0(:,:,:,t)));   % sum since likelihood factorises for each entry
%          logpX(:,:,:,l)  = (logpX1(:,:,:,l) .* (Xt(:,:,:,t)==1)) + (logpX0(:,:,:,l) .* (Xt(:,:,:,t)==0));
%          clkl_X(t,l) = clkl_X(t,l) + sum(logpX(:,:,:,l),'all');     % sum since likelihood factorises for each entry
      end
   end
   
   if sum(isnan(clkl(:))) ~= 0
      error('FFBS: NaN in clkl.');
   end
   
   %%% NOTE: if difference of clkl over states too big --> filter/smooth prob ~ 0/1 --> stuck into same state
   %%%       Need difference < 2.
   if scale_clkl == 1
      D_clkl = abs( diff(clkl,1,2) );  % difference of clkl between states
      if sum(D_clkl > 2.5) > 0   % 2.5 small value since it is logarithmic scale
         while sum(D_clkl > 2.5) > 0
            pos = D_clkl > 2.5;
            clkl(pos,:) = clkl(pos,:) / 2;
            D_clkl = abs( diff(clkl,1,2) );
         end
      end
   end
   
   %%%%%%%% Hamilton filter (forward-filter backward-sampler) %%%%%%%%
   % INPUTS:  conditional log-like; transition matrix; initial proba
   % OUTPUTS: log-like; filter proba + states; smooth proba + states
   [lkl,filtp,filts,smoothp,smooths] = Hamilton(clkl,P,ip);
   
end
