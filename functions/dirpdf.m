function pdf = dirpdf(x,p,logarithm)
%% dirpdf Evaluates the density of a Dirichlet distrtibution
%
%    PDF = dirpdf(X,P,LOGARITHM) computes the density of a Dirichlet distribution
%    of parameter P at the point X. Returns the log-density if LOGARITHM == 1.
%
% INPUTS
%  x          Nx1  vector of values at which evaluate PDF
%  p          Nx1  vector of probabilities (parameters of distribution)
%  logarithm  1x1  if 1 returns "log(pdf)", otherwise "pdf"
%
% OUTPUTS
%  PDF        1x1  scalar value of PDF Dir(x|p)
%
% PROBLEM
%   does NOT handle values that underflowed to zero (i.e.: coming from the 
%   dsitribution with small parameter values). p(j) small -> x(j) small -> pdf BIG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % check INPUT
   if length(p) < 2
      error('Parameter "p" must be a vector of length >= 2.');
   end
   % check that both are vectors, with same length
   if isvector(x) ~=1 || isvector(p) ~= 1
      error('Both "x" and "p" must be vectors.')
   end
   if length(x) ~= length(p)
      error('Vectors "x" and "p" must have the same length.');
   end
   % check positive entries of parameter vector "p"; check "x" in (0,1)
   for i=1:length(p)
      if p(i) <= 0
         error('Parameter "p" must have positive entries.');
      end
      if x(i) <= 0 || x(i) >= 1
         error('Vector of "x" values must have entries in (0,1).');
      end
   end
   % check vector "x" sums to number very close to 1 (allow small discrepacy)
   if abs( sum(x)-1 ) > 10^(-10)
      error('Vector of "x" values does not sum to 1.');
   end
   
   % transform both into column vectors (need both in the same format)
   [~, xc] = size(x);     [~, pc] = size(p);
   if xc ~= 1
      x = x';
   end
   if pc ~= 1
      p = p';
   end
   
   
   % take exp{log(...)} for reduce numerical errors
   % exp{ -(sum[log(Gamma(p_i))] - log[Gamma(sum(p_i))]) + sum[(p_i-1) * log(x_i)] }
   l1 = -( sum(gammaln(p)) - gammaln(sum(p)) );
   l2 = sum( (p-1) .* log(x) );
   if logarithm==1
      pdf = l1 + l2;
   else
      pdf = exp(l1 + l2);
   end
   
end


%################################################################################
%####################### Check "bad entries" of "x" #############################
%    tol = 0.000001;   % tolerance level for "x"
%    sum_x = sum(x);
%    % CHECK 1: checks each row globally (sum)
%    % Badx1 --> sum too small (almost 0) or too big (slightly >1) --> out: logical array
%    Badx1 = (sum_x<tol | sum_x>1+tol);
%    % CHECK 2: checks each row infividually (entrywise)
%    % Badx2 --> counts #x satisfying "bad conditions" in each column --> out: logical array
%    %           with 1 where count is >0 (exist at leat one entry of "x" badly behaved)
%    Badx2 = sum(x>(1+tol) | x<tol)>0;
%    % Badx --> logical array, with 1 when either Badx1 or Badx2 has a non-zero entry.
%    %          out: matrix with all the "bad entries (rows)"
%    Badx = Badx1|Badx2;
%    % substitutes to "badly behaved (rows)" artificial rows,
%    %             with 1 in first position and 0 elsewhere
%    x(Badx,:) = [ones(sum(Badx),1), zeros(sum(Badx),d-1)];
%    
%    a = p;
%    % Bada -->counts rows where "a" has very small entries (< tolerance)
%    Bada = sum(a<(0+tol),2)>0;
%    % substitutes to "badly behaved (rows)" artificial rows,
%    %             with all 1
%    a(Bada,:) = ones(sum(Bada),d);
%    
%    % WARNING if there is at leat one "badly behaved (row)" in "x" or "a"
%    if sum(Badx)>0
%        warning('dirpdf:ValuesOutOfBound',': "x" has values out of bound.');
%    end
%    if sum(Bada)>0
%        warning('dirpdf:ValuesOutOfBound',': "p" has values out of bound.');
%    end   
%################################################################################
