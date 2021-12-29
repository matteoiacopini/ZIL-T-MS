function X = gigrnd(p, a, b, numSim)
%% gigrnd Samples from a Generalized Inverse Gaussian distribution
%
% Implementation of the Devroye (2014) algorithm
% p(x | p,a,b) = (a/b)^(p/2)/2/besselk(p,sqrt(a*b))*x^(p-1)*exp(-(a*x + b/x)/2)
%
%     X = gigrnd(P,A,B,NUMSIM)  draws numSim independent random numbers from 
%     a Generalized Inverse Gaussian distribution with parameters (p,a,b), or GiG(p,a,b)
%
% INPUTS
%   p  1x1  real
%   a  1x1  real, positive 
%   b  1x1  real, positive
%
% OUTPUTS:
%   X  numSim x 1  vector numSim independent draws from a GiG(p,a,b) distribution
%
% NOTES:
%  p == -1/2  --> Inverse Gaussian distribution
%  b == 0     --> Gamma distribution:  x^(k-1)*exp(-x/theta)   with:  k=p  theta=a/2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % check arguments
   if a <= 0
      fprintf('a= %d\n',a);
      error('Parameter "a" must be a>0');
   end
   if b <= 0
      fprintf('b= %d\n',b);
      error('Parameter "b" must be b>0');
   end
   % check NaN
   if isnan(p) == 1
      error('Parameter "p" is NaN.');
   end
   if isnan(a) == 1
      error('Parameter "a" is NaN.');
   end   
   if isnan(b) == 1
      error('Parameter "b" is NaN.');
   end
   
   
   % Setup: sample from the two parameter version of the GIG(alpha,omega)
   lambda = p;
   omega = sqrt(a*b);
   alpha = sqrt(omega^2 + lambda^2) - lambda;

   % Find t,s
   x = -Psi(1, alpha, lambda);   % want it in[0.5,2]
   if((x >= 0.5) && (x <= 2)) 	% keep 1 since already in[0.5,2]
       t = 1;
       s = 1;
   elseif(x > 2)
       t = sqrt(2 / (alpha + lambda));
       s = sqrt(4 / (alpha*cosh(1) + lambda));
   elseif(x < 0.5)
       t = log(4 / (alpha + 2*lambda));   
       s = min(1 / lambda, log(1 + 1/alpha + sqrt(1/alpha^2 + 2/alpha)));  
   end

   % Generation
   eta = -Psi(t, alpha, lambda);
   zeta = -dPsi(t, alpha, lambda);
   theta = -Psi(-s, alpha, lambda);
   xi = dPsi(-s, alpha, lambda);
   p = 1/xi;
   r = 1/zeta;
   td = t - r*eta;
   sd = s - p*theta;
   q = td + sd;

   X = zeros(numSim, 1);
   for sample = 1:numSim
       done = false;
       while(~done)
           U = rand(1); 
           V = rand(1); 
           W = rand(1);
           if(U < (q / (p + q + r)))
               X(sample) = -sd + q*V;
           elseif(U < ((q + r) / (p + q + r)))
               X(sample) = td - r*log(V);
           else
               X(sample) = -sd + p*log(V);
           end

           % Are we done?
           f1 = exp(-eta - zeta*(X(sample)-t));
           f2 = exp(-theta + xi*(X(sample)+s));
           if((W*g(X(sample), sd, td, f1, f2)) <= exp(Psi(X(sample), alpha, lambda)))
               done = true;
           end
       end
   end
   
   % X has density: f(x) propto exp(psi(x))
   % Transform X back to the three parameter GIG(p,a,b)
   X = exp(X) * (lambda / omega + sqrt(1 + (lambda/omega)^2));
   X = X ./ sqrt(a/b);

end

% ###################### Addiitonal functions ######################
% NOTE: call it Psi since MATLAB function "psi" already exists
function f = Psi(x, alpha, lambda)
    f = -lambda*(exp(x) - x - 1) -alpha*(cosh(x) - 1);
end

function f = dPsi(x, alpha, lambda)
    f = -lambda*(exp(x) - 1) -alpha*sinh(x);
end

function f = g(x, sd, td, f1, f2)
   % g(x)= chi(x) in paper
   a = 0; b = 0; c = 0;
   if((x >= -sd) && (x <= td))
       a = 1;
   elseif(x > td)
       b = f1;
   elseif(x < -sd)
       c = f2;   
   end

f = a + b + c;
end
