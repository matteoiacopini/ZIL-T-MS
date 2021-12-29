function [out,accrate] = HMClambda(start, a, b, w, SigM, eps, L, epsrnd, alpha)
%% HMClambda  Draws lambda from its posterior full conditional distribution using a Hamiltonian Monte Carlo step
%
% THEORY: Hamiltonian dynamics used to propose a new value in Metropolis-Hastings
% The target is the joint H(q,p) --> q interest (position), p auxiliary (momentum)
% Propose only p (implies change H(q,p) and marginal q) from symmetric --> disappears,
% thus, only acceptance ratio H(q,p)-H(q*,p*) [start and end of trajectory]
%
%
%   [OUT,ACCRATE] = HMClambda(START,A,B,W,SIGM,EPS,L,EPSRND,ALPHA)
%   Samples onne value (OUT) from the posterior full conditional distribution of lambda
%   using a Hamiltonian Monte Carlo step, with L leapfrog steps of length EPS.
%   The parameters of the full confitional are passed as inputs (A,B,W), together with
%   the value of lambda at the current iteration of the MCMC algoritthm (START).
%   If EPSRND == 1, then a randomized EPS is used, following Neal (2011).
%   If ALPHA == 1, then a tempering method is used, following Neal (2011).
%   The output ACCRATE records the acceptance probability of the move.
%
%
% NOTE: Gamma(shape,scale)   ---   1/b^a*G(a) * x^(a-1) * exp(-x/b)
%
% INPUTS
%   start   1x1 initial value for position
%   a       1x1 parameter target distribution (Gamma)
%   b       1x1 parameter target distribution (Gamma)
%   w       RxL parameter of target distribution (prod[exp(...)])
%   eps     1x1 leapfrog step length (default)
%   L       1x1 #steps (default)
%   epsrnd  1-0 randomize epsilon (uniform +/-20%) -- (Neal 2011)
%   alpha   1x1 tempering if =/= 1 -- (Neal 2011)
%
% OUTPUTS
%   out     1x1 new position
%
% NOTES
%   1) eps,L  -> calibrate: eps*L ~ length proposed move --> bigger if longer distances
%             sould be covered. Too long = lower acceptance.
%   2) uses 'log ...' and 'gradlog ....' for:   log(pdf) + grad(log(pdf)) of ....
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if nargin < 5
      SigM = 1;
      eps = 0.013;
      L = 100;
      epsrnd = 1;
      alpha  = 1;   % no tempering if == 1
   end
   if epsrnd == 1   % random epsilon (Neal 2011)
      eps = unifrnd(eps*0.8,eps*1.2);  % eps +/-20%
   end
   R = size(w,1);
   
   q = start;
   Mom = chol(SigM) * randn([length(q),1]);  % MOMENTUM ~ N(0,1)
   %%%% initialPos = log(pdf) -- MVN(start,Mu,Sigma); %%%%
   initialPos = log_target_pdf(q,a,b,w,R);  %log(target pdf)
   initialMom = (Mom' * SigM^(-1) * Mom) / 2;
   
%    % MOMENTUM: Half step (at beginning of trajectory)
%    Mom = alpha^(1/2)*Mom + eps * gradlogMVN(q,Mu,Sigma) / 2;
   % Alternate Full steps for Position and Momentum
   for l=1:L
      %%% TEMPERING %%%
      if     l<=L/2           % 1st half of trajectory
         Mom = Mom * alpha^(1/2);
      elseif l>L/2 && l~=L    % 2nd half of trajectory
         Mom = Mom / alpha^(1/2);
      end
      
      % MOMENTUM: Half step (at beginning of trajectory)
      % Mom = Mom + eps * gradlogMVN(q,Mu,Sigma) / 2;
      Mom = Mom + eps * grad_log_target_pdf(q,a,b,w,R) / 2;
      % POSITION: Full step
      q = q + eps * SigM^(-1) * Mom;
      % MOMENTUM: Half step (at end of trajectory)
      % Mom = Mom + eps * gradlogMVN(q,Mu,Sigma) / 2;
      Mom = Mom + eps * grad_log_target_pdf(q,a,b,w,R) / 2;
      
      %%% TEMPERING %%%
      if     l<=L/2           % 1st half of trajectory
         Mom = Mom * alpha^(1/2);
      elseif l>L/2 && l~=L    % 2nd half of trajectory
         Mom = Mom / alpha^(1/2);
      end
   end
%    % MOMENTUM: Half step (at end of trajectory)
%    Mom = Mom + eps * gradlogMVN(q,Mu,Sigma) / 2;
   % Negate momentum at end of trajectory to make the proposal symmetric (theory)
   Mom = -Mom;
   % Evaluate MH acceptance ratio for Hamiltonian dynamics proposal move (q,p)->(q,p)^*
   %%%% propPos = log(pdf) -- logMVN(q,Mu,Sigma); %%%%
   propPos = log_target_pdf(q,a,b,w,R);  %log(Gammapdf)
   propMom = (Mom' * SigM^(-1) * Mom) / 2;
   % Accept/reject the state at end of trajectory
   fold = -initialPos + initialMom;    % theory: exp(-U(x)), U(x)= logpdf -> change sign
   fnew = -propPos + propMom;
   accrate = min(1, exp(-(fold-fnew)));
   if rand(1) < accrate
      out = q;      % accept
   else
      out = start;  % reject
   end
   
end


%%%%%%%%%%%%%%%% AUXILIARY functions %%%%%%%%%%%%%%%%
%%%%%%%%% logpdf of ... %%%%%%%%%
function [logpdf] = log_target_pdf(x,a,b,w,R)
   % pdf:  b^a/G(a) * lambda^(a-1) * exp(-b*lambda) * prod_{h,r}[ lambda^2/2 * exp(-lambda^2/2 * w_{h,r}) ]
   logpdf = a*log(b) - gammaln(a) + (a+8*R-1)*log(x) -b*x -4*R*log(2) -(x^2/2 * sum(sum(w)));
   % target pdf PROPORTIONAL:  lambda^(a+8R-1) * exp(-b*lambda -lambda^2/2 * sum_{h,r} w_{h,r}) 
%    logpdf = (a+8*R-1)*log(x) -b*x -x^2/2 * sum(sum(w));
end

%%%%%%%%% grad(logpdf) of ... %%%%%%%%%
function [glpdf] = grad_log_target_pdf(x,a,b,w,R)
   glpdf = (a+8*R-1)/x -b -(x * sum(sum(w)));
end

