function [lkl, filtp, filts, smoothp, smooths] = Hamilton(clkl,P,ip)
%% Hamilton Runs the Hamilton filter/smoother for Markov Switching/Mixture model.
% See the Forward-Filter Backward-Smoother algorithm [Fruhwirth-Schnatter(2006)]
%
%   [LKL,FILTP,FILTS,SMOOTHP,SMOOTHS] = Hamilton(CLKL,P,IP)
%   Applies the Hamilton filter/smoother using as inputs a (TxL) matrix of 
%   conditional likelihood values in each state l=1,...,L and time t=1,...,T,
%   where the hidden Markov chain has transiton matrix P and initial probability IP.
%   Returns the filtered and smoothed probabilities (FILTP, SMOOTHP), the 
%   filtered and smoothed states (FILTS, SMOOTHS), and the log-likelihood (LKL).
%
% INPUTS
%     ip = Lx1 vector initial probabilities
%     P  = LxL matrix transition probabilities (row t -> col t+1)
%     clkl = TxL conditional log-likelihood (time series T for each regime L):
%                 log(y1|s1=1,--) ... log(y1|s1=L,--)
%                       ...       ...       ...
%                 log(yT|sT=1,--) ... log(yT|sT=L,--)
%
% OUTPUTS
%     lkl     = Tx1 unconditional log-likelihood   logp(yt|y1,..,yt-1)
%     filtp   = TxL filtered probabilities
%     filts   = Tx1 filtered states
%     smooths = Tx1 smoothed states (smooth(t)= 1..L)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   [T,L] = size(clkl);
   lkl     = zeros(T,1);
   filtp   = zeros(T,L);      smoothp = zeros(T,L);
   filts   = zeros(T,1);      smooths = zeros(T,1);   
   
   %%%%%% (Forward) FILTERING of states ---- [Alg 11.1 p320 FS(2006)] %%%%%%
   st1t1 = ip;
   for t=1:T
       stt1 = P' * st1t1;  % Lx1
       % log(.): numerical treatment low probabilities [p322 FS(2006)]
       clklmax = max( clkl(t,stt1>0) );   % max conditional log-like (within those with positive proba)
       stt = stt1 .* ( exp(clkl(t,:) - clklmax)' ); % p^*(yt|t1..t-1)
       nc  = sum(stt);
       stt = stt/nc;    % normalise probabilities
       filtp(t,:) = stt';           % filtered probabilities
       lkl(t) = clklmax + log(nc);  % unconditional log-likelihood
       st1t1 = stt;                 % state vector update (instead of saving every t, update vector with P(St))
       [~,maxpos] = max(stt);
       filts(t)   = maxpos;         % filtered state = MaxAPosteriori (highest prob)
   end
   
   %%%%%% (Backward) SMOOTHING and SAMPLING of states ---- [Alg 11.5 p343 FS(2006)] %%%%%%
   % choose state T according to filtered prob T (since smoothed prob T = filtered prob T)
   smoothp(T,:) = filtp(T,:);    % smoothed proba
%    smooths(T)   = mnrnd(1,smoothp(T,:)') * (1:L)';    % mnrnd(1,p) = multinomial
   smooths(T) = find(cumsum(smoothp(T,:)) > rand(1),1,'first');  % inverse CDF (faster than mnrnd)
   for t=(T-1):-1:1
       p1 = (filtp(t,:)') .* P(:,smooths(t+1));
       p1 = p1/sum(p1);
       smoothp(t,:) = p1;        % smoothed proba
       % choose state t according to smoothed prob t
%        smooths(t) = mnrnd(1,smoothp(t,:)') * (1:L)';
       smooths(t) = find(cumsum(smoothp(t,:)) > rand(1),1,'first');  % inverse CDF (faster than mnrnd)
   end
   
end
