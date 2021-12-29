function Y = VAR_gdp_unif(T,lags,dim,unif_lim)
%% VAR_gdp_unif   Simulate a stationary VAR(p)
% 
%   Y = VAR_gdp_unif(T,LAGS,DIM,UNIF_LIM)
%   Simulates a stationary VAR process with LAGS lags, length T, and cross-sectional size DIM,
%   storing the results into a (DIM,T) matrix Y.
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



%% Additional functions
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
