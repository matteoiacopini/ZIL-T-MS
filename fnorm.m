function fn = fnorm(B,t)
%% fnorm Computes different tyoes of norms for vector, matrix, tensor inputs
%
%  FN = fnorm(B,T)  Computes the norm of B (tensor, matrix or vector).
%  The type of norm that is computed is defined by the code T.
%
% INPUTS
%   B         vector, matrix, tensor
%   t   1x1   code 1...4
%
% OUTPUT
%   fn  1x1 norm of the tensor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % VEC, MAT, TENS - Max
   switch t         
      % VEC, MAT, TENS - "sum of Abs"
      case 1
         vB= reshape(B,[],1);
         fn= sum(abs(vB));
         return
         
      % VEC, MAT, TENS - "Frobenius"
      case 2
         vB= reshape(B,[],1);
         fn= sqrt(sum(abs(vB).^2));
         return
         
      % MAT - "max norm"
      case 3
         fn= max(max(abs(B)));
         return
      
      % MAT - "average of abs"
      case 4
         [nr,nc]= size(B);
         fn= sum(sum(abs(B)))/(nr*nc);
         return
   end
   
end