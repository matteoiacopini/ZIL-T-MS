function [Ainv] = MatInv(A)
%% MatInv Computes the matrix inverse using numerical corrections if it is close to singular
%
% Inverse matrix:   IF rcond(A)< 10^(-9) --> invert (A + eye(.)*eps)
%
%   AINV = MatInv(A) Computes the inverse of A applying a numerical correction when A is
%   close to singular. Specifically, it adds a perturbation to the main diagonal of A, which
%   depends on the magnitude of the enttries of A.
%
% INPUTS
%   A     NxN  square matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % compute reciprocal conditioning number: high -> close to singular
   n = size(A,1);
   c = rcond(A);
   thresh_c = 10^(-9);     % default arbitrary small value
   % compare rcond with threshold level
   if c > thresh_c
      Ainv = A \ eye(n);
      
   else
%       fprintf('Reverse cond number LOW= %d.\n',c);
      % set PERTURBATION
      eps = max(max(abs(A))) * 1e-10;
      Ainv = (A + eye(n)*eps) \ eye(n);
   end
   
end