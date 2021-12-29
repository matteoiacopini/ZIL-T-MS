function [HYPER] = define_hyperparameters(L,R,nameArgs)
%% define_hyperparameters Set the hyperparameters for the model (set to default if missing)
%
%   [HYPER] = define_hyperparameters(L,R,NAMEARGS) 
%   Defines the hyperparameters for the prior distributions used in the ZIL-T-MS model,
%   with PARAFAC rank R and L states.
%   If the hyperparameters are not provided as optional name-value pair arguments, then
%   the function will assign default values.
%   The output is a structure, HYPER, containing all the hyperparameters.
% 
% INPUTS (optional)
%   L          (1,1)   number of states of the hidden chain
%   R          (1,1)   rank of the coefficient tensor
%   alambda    (L,1)   shape parameter of the Gamma prior for lambda, in each state l=1..L
%   blambda    (L,1)   scale parameter of the Gamma prior for lambda, in each state l=1..L
%   alphabar   (1,1)   shape parameter of the Dirichlet prior for phi
%   btaubar    (1,1)   scale parameter of the Gamma prior for tau
%   cbar       (L,L)   parameter vector of the Dirichlet prior for each row l=1..L of the Markov chain transition matrix
%   arhobar    (L,1)   shape parameter 'a' of the Beta prior for rho, in each state l=1..L
%   brhobar    (L,1)   shape parameter 'b' of the Beta prior for rho, in each state l=1..L
%   ident_constr  (1,1) logical impose identification constraint during MCMC?
%   constr_g3     (1,1) logical impose constraint on gamma3?
%
% OUTPUT
%   HYPER      structure with fields given by all the inputs.
%              If optional inputs are missing, they are set to default values. Fields:
%               * L          (1,1)   number of states of the hidden chain
%               * R          (1,1)   rank of the coefficient tensor
%               * alambda    (L,1)   shape parameter of the Gamma prior for lambda, in each state l=1..L
%               * blambda    (L,1)   scale parameter of the Gamma prior for lambda, in each state l=1..L
%               * alphabar   (1,1)   shape parameter of the Dirichlet prior for phi
%               * ataubar    (1,1)   shape parameter of the Gamma prior for tau
%               * btaubar    (1,1)   scale parameter of the Gamma prior for tau
%               * cbar       (L,L)   parameter vector of the Dirichlet prior for each row l=1..L
%                                    of the Markov chain transition matrix
%               * arhobar    (L,1)   shape parameter 'a' of the Beta prior for rho, in each state l=1..L
%               * brhobar    (L,1)   shape parameter 'b' of the Beta prior for rho, in each state l=1..L
%               * ident_constr  (1,1) logical impose identification constraint during MCMC?
%               * constr_g3     (1,1) logical impose constraint on gamma3?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

arguments
   L (1,1) double = 2;   % number of states of the hidden chain
   R (1,1) double = 5;   % rank of the coefficient tensor
   nameArgs.alambda  (:,1) double = ones(L,1) * 36.0/1.0;          % [36.0/1.0;  36.0/1.0]; % mu=0.09
   nameArgs.blambda  (:,1) double = ones(L,1) * (1.0/6.0).^(-1);   % [1.0/6.0;  1.0/6.0].^(-1);
   nameArgs.alphabar (1,1) double = 2.0 * 1e-0; %1e-6;           % 3 - 0.5, affects: phi_r, tau
   nameArgs.btaubar  (1,1) double = 2^(-1);
   nameArgs.cbar     (:,:) double = ones(L,L) * 4.0;
   nameArgs.arhobar  (:,1) double = ones(L,1);
   nameArgs.brhobar  (:,1) double = ones(L,1);
   nameArgs.ident_constr (1,1) logical = true;
   nameArgs.constr_g3    (1,1) logical = false;
%    nameArgs.mu1prior (:,:) double = zeros(I,L);
%    nameArgs.mu2prior (:,:) double = zeros(J,L);
%    nameArgs.mu3prior (:,:) double = zeros(K,L);
%    nameArgs.mu4prior (:,:) double = zeros(Q,L);
end

alambda  = nameArgs.alambda;
blambda  = nameArgs.blambda;
alphabar = nameArgs.alphabar;
ataubar  = alphabar * R;  % constraint in posterior
btaubar  = nameArgs.btaubar;
cbar     = nameArgs.cbar;
arhobar  = nameArgs.arhobar;
brhobar  = nameArgs.brhobar;
ident_constr = nameArgs.ident_constr;
constr_g3    = nameArgs.constr_g3;
% mu1prior = nameArgs.mu1prior;
% mu2prior = nameArgs.mu2prior;
% mu3prior = nameArgs.mu3prior;
% mu4prior = nameArgs.mu4prior;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% create structure with all hyperparameters
HYPER = struct;
HYPER.L        = L;
HYPER.R        = R;
HYPER.alambda  = alambda;
HYPER.blambda  = blambda;
HYPER.alphabar = alphabar;
HYPER.ataubar  = ataubar;
HYPER.btaubar  = btaubar;
HYPER.cbar     = cbar;
HYPER.arhobar  = arhobar;
HYPER.brhobar  = brhobar;
HYPER.ident_constr = ident_constr;
HYPER.constr_g3    = constr_g3;
% HYPER.mu1prior = mu1prior;
% HYPER.mu2prior = mu2prior;
% HYPER.mu3prior = mu3prior;
% HYPER.mu4prior = mu4prior;
end