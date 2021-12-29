function [Asym] = MatSym(A)
%% MatSym Symmetrise a square matrix
%
%    ASYM = MatSym(A) Returns a symmetric version of the square matrix A
%
%
% INPUTS
%  A         NxN  square matrix
%
% OUTPUTS
%  Asym      NxN  "symmetrized" version of A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % check if symmetric
   if issymmetric(A)
      Asym = A;
      
   else
      new_sym1 = (A*A')/2;
      new_sym2 = diag(diag(A)) + triu(A,1) + triu(A,1)';     % CLUSTER v2015b
      % compute norm (absolute value)
      if fnorm(new_sym1-A, 1) < fnorm(new_sym2-A, 1)
         Asym = new_sym1;
      else
         Asym = new_sym2;
      end
%       maxdiff = max(max(abs(Asym-A)));
%       if maxdiff > 10^3
%          error('Symmetrise: too big difference');
%       end
%          fprintf('Symmetrise: max distance in/out: %d\n',maxdiff);

%       % CHECK post-symmetry
%       if issymmetric(new_sym) == 0
%          error('Output matrix is NOT symmetric.');
%       end
   end
end
