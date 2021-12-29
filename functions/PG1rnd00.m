function sim = PG1rnd00(Z00)
%% PG1rnd00 Sample from the Polya-Gamma distribution PG(1,Z00)
%
%   SIM = PG1rnd00(Z00) Draws one random number from the Polya-Gamma distribution
%   with parameters a=1 and b=Z00, that is, PG(1,Z00).
%   If Z00 is a vector of length N, then PG1rnd00 draws one random number from PG(1,Z00(j)), 
%   for each j=1,...,N.
%
% INPUTS
%  Z00   Nx1  vector of parameter values
%
% OUTPUTS
%  SIM   Nx1  vector of draws from the distribution PG(1,Z00)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   nn = length(Z00);
   sim = zeros(nn,1);
   for i=1:nn
   Z1 = Z00(i);
   TRUNC = 0.64;
   Z = abs(Z1) * 0.5;
   fz = pi^2 / 8 + Z^2 / 2;

   while 1
      if unifrnd(0,1) < masstexpon(Z)
         %%% Truncated Exponential
         X = TRUNC + exprnd(1) / fz;
      else
         %%% Truncated Inverse Gaussian
         ZZ = abs(Z);
         mu = 1/ZZ;
         X = TRUNC + 1;
         if mu > TRUNC
            alpha = 0.0;
            while unifrnd(0,1) > alpha
               E = exprnd(1,2,1);
               while E(1)^2 > 2*E(2)/TRUNC
                  E = exprnd(1,2,1);
               end
               X = TRUNC / (1 + TRUNC*E(1))^2;
               alpha = exp(-0.5 * ZZ^2 * X);
            end
         else
            while X > TRUNC
               lambda = 1.0;
               Y = normrnd(0,1)^2;
               X = mu + 0.5 * mu^2 / lambda * Y - ...
                  0.5 * mu / lambda * sqrt(4 * mu * lambda * Y + (mu * Y)^2);
               if unifrnd(0,1) > mu / (mu + X)
                  X = mu^2 / X;
               end
            end
         end

      end

      % C = cosh(Z) * exp( -0.5 * Z^2 * X )
      % Don't need to multiply everything by C, since it cancels in inequality.
      S = acoef(0,X);
      Y = unifrnd(0,1)*S;
      n = 0;

      while 1
         n = n + 1;
         if mod(n,2)
            S = S - acoef(n,X);
            if Y <= S
               break
            else
               S = S + acoef(n,X);
               if Y > S
                  break
               end
            end
         end
      end

      if Y <= S
         break
      end
   end

   sim(i)= 0.25 * X;
   end
end
