function [Apd] = MatPosDef(A)
%% MatPosDef Returns a Positive Definite version of matrix A, applying a numerical correction if necessary
%
%    APD = MatPosDef(A) For a (theoretically) Positive Definite (PD) matrix A, the function
%    MatPosDef first checks if A is actually PD and, if not, it applies a numerical correction
%    to its eigenvalues to make it PD.
%
% INPUTS
%   A      square matrix
%
% OUTPUTS
%   Apd    proposed positive definite version of A
%
% PROPERTIES of POSITIVE DEFINITE matrices
% - symmetric
% - ALL eigenvalues > 0
% - all principal minors > 0
% - unique Cholesky decomposition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   [~,p] = chol(A);
   if p == 0  % already POSITIVE DEFINITE
      Apd = A;
       
   else       % NOT POSITIVE DEFINITE -- change Eigenvalues
      [V,D] = eig(A);
      d = diag(D);
      d(d<=0) = 1e-14;     % arbitrary small value > 0
      Apd = V * diag(d) * V';
   end
   
end
