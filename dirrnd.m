function sample = dirrnd(p,numSim)
%% dirrnd Samples from a Dirichlet distribution
%
%    SAMPLE = dirrnd(P,NUMSIM)  Draws NUMSIM vectors independently from
%    a Dirichlet distribution with parameter vector P.
% 
% INPUTS
%  p       Nx1  vector of positive values
%  numSim  1x1  number of draws (each draw is a vector of length N)
%
% OUTPUTS
%  sample  N x numSim   matrix of simulated vectors, each column is an independent draw from Dirichlet(P)
%
% PROBLEM:
% - gamrnd(.,.): for small values of scale, draws 0. Since x~Dir(a) is obtained
%                stardardizing x_i~gamma(.,.): x_i= x_i/sum(x). If draw from gamma
%                is 0, then x_i=0.
%                ERROR: support(gamma)= (0,+inf)   supp(dirichlet)= (0,1)
%                Error is spread also over "dirpdf" since it expects x_i=(0,1)
% SOLUTION: substitute small/big values with eps/(1-eps); default eps=10^-5
%           then, re-normalize in order to have sum(x)=1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   if nargin < 2
      numSim = 1;
   end
   % check INPUT
   if ~isvector(p)
      error('Parameter "p" must be a vector.');
   end
   if length(p) < 2
      error('Parameter "p" must be a vector of length >= 2.');
   end
   for i=1:length(p)
      if p(i) <= 0
         error('Parameter "p" must have positive entries.');
      end
   end
   if length(numSim) > 1
      error('Parameter "numSim" must be a scalar.');
   end
   
   % next code considers "p" a row vector; hence transform in it if it's not
   [pr,~] = size(p);
   if pr ~= 1
      p = p';
   end
   
   %########################################################################
   % sample from Dirichlet(p) in each ROW (sum to 1)
   dim_p = length(p);
   % matrix, vector a in each row:   size(mat)= [numSim, length(p)]
   mat = repmat(p, numSim, 1);
   sample = gamrnd(mat,1,[numSim,dim_p]);
   sample = sample ./ repmat(sum(sample,2),1,dim_p);
   % transpose, for having sample in each COLUMN (sum to 1)
   sample= sample';
   
   %########################################################################
   % check OUTPUT
   % 1) each entry is in (0,1) [avoid boundary values]
   eps = 10^(-5);    % 10^-5= 0.00001
   for j=1:numSim
      % find elements (in each sample) close to 0 or 1 --> out: logical arrays x0,x1
      x0 = sample(:,j) < eps;
      x1 = sample(:,j) > 1-eps;
      % apply corrections: set to small/big value
      sample(x0,j) = eps;
      sample(x1,j) = 1-eps;
   
   % 2) each column sums to 1
      if sum(sample(:,j)) ~= 1
         % re-normalize sampled column, so that it will sum to 1
         %disp('Sum of sampled "x" is not 1, proceed with corrections.');
         sample(:,j) = sample(:,j) ./ sum(sample(:,j));
      end
   end
   
end
