function [OUT,data] = simulation_FUNC(nameArgs)
%% simulation_FUNC Generates synthetic binary temporal network dataset, then estimates the Tensor MS-ZIL model
%
%   [OUT,DATA] = simulation_FUNC(NAMEARGS)
%   Generates a synthetic dataset from the ZIL-T-MS model, using the setting specified by the
%   hyperparameter values provided as optional name-value pair arguments.
%   If no inout is provided, then the function will assign default values to the hyperparameters.
%   The output consists in two structures: DATA contains the synthetic data and the value of the 
%   parameters used to generate them; OUT contains the estimation results.
%
% INPUTS (optional)
%  I,J,K   1x1   size of the observed binary array (I,J,K) for each time t=1..T
%  Q       1x1   number of covariates
%  T       1x1   length of the time series (observations)
%  L       1x1   number of latent states
%  R       1x1   CP rank of coefficient tensor G
%  type_coeff  string  Code for generating the dataset. Must be one of the following:
%                        'col', 'row', 'block', 'sparse', 'base' in lowercase letters
%  NumIter   1x1  number of MCMC iterations to store
%  burn      1x1  number of initial iterations to thrash as burn-in phase
%  thin      1x1  thinning factor: store 1 iteration every 'thin' ones
%  irep      1x1  display cumulative computing time once every 'irep' iterations
%  save_file logical   save data and MCMC results in separate .mat files?
%
% OUTPUTS
%  data   structure with the synthetic dataset
%  OUT    structure with the estimation results of the MCMC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

arguments
   nameArgs.I (1,1) double = 10  % size of the observed binary array (I,J,K) for each time t=1..T
   nameArgs.J (1,1) double = 10
   nameArgs.K (1,1) double = 1
   nameArgs.Q (1,1) double = 3   % number of covariates
   nameArgs.T (1,1) double = 50  % length of the time series (observations)
   nameArgs.L (1,1) double = 2   % number of latent states
   nameArgs.R (1,1) double = 5   % CP rank of coefficient tensor G
   nameArgs.type_coeff char = 'base'      % type of coefficient tensor G (block, row, col, sparse, base)
   nameArgs.NumIter  (1,1) double = 1000  % number of MCMC iterations to store
   nameArgs.burn     (1,1) double = 100   % number of initial iterations to thrash as burn-in phase
   nameArgs.thin     (1,1) double = 1     % thinning factor: store 1 iteration every 'thin' ones
   nameArgs.irep     (1,1) double = 10    % display cumulative computing time once every 'irep' iterations
   nameArgs.save_file logical = false     % save data and MCMC results in separate .mat files?
end

I = nameArgs.I;
J = nameArgs.J;
K = nameArgs.K;
Q = nameArgs.Q;
T = nameArgs.T;
L = nameArgs.L;
R = nameArgs.R;
type_coeff = nameArgs.type_coeff;
NumIter = nameArgs.NumIter;
burn    = nameArgs.burn;
thin    = nameArgs.thin;
irep    = nameArgs.irep;
save_file = nameArgs.save_file;

% checks
if ~any( strcmpi(type_coeff,{'col','row','block','sparse','base'}) )
   error(['The code for generating the dataset must be one of the following: '...
          '''col'', ''row'', ''block'', ''sparse'', ''base''']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


disp(['%%%%%%%%% Running simulation ''', type_coeff, ''' with size (I,J,K)= (',...
      num2str(I),',',num2str(J),',',num2str(K),'), Q= ',num2str(Q),' covariates, T= ',num2str(T),' periods %%%%%%%%%']);

% generate data
if strcmpi(type_coeff,'col')
   % partial heterogeneity 'cross columns' -- matricized tensor IxJQ has horizontal stripes
   G = zeros(I,J,K,Q,L);
   for k=1:K
      G(    1:I/2, :, k, :, 1) =  0.1;
      G(I/2+1:end, :, k, :, 1) = -0.1;
      G(    1:I/2, :, k, :, 2) =  0.5;
      G(I/2+1:end, :, k, :, 2) = -0.5;
   end
elseif strcmpi(type_coeff,'row')
   % partial heterogeneity 'cross rows' -- matricized tensor IxJQ has horizontal stripes
   G = zeros(I,J,K,Q,L);
   for k=1:K
      G(:,     1:J/2, k, :, 1) =  0.1;
      G(:, J/2+1:end, k, :, 1) = -0.1;
      G(:,    1:J/2,  k, :, 2) =  0.5;
      G(:, J/2+1:end, k, :, 2) = -0.5;
   end
elseif strcmpi(type_coeff,'block')
   % partial heterogeneity 'blocks' -- matricized tensor IxJQ has blocks on each slice
   G = zeros(I,J,K,Q,L);
   for k=1:K
      G(1:I/2,     1:I/2,     k, :, 1) =  0.1;
      G(I/2+1:end, I/2+1:end, k, :, 1) = -0.1;
      G(1:I/2,     1:I/2,     k, :, 2) =  0.5;
      G(I/2+1:end, I/2+1:end, k, :, 2) = -0.5;
   end
elseif strcmpi(type_coeff,'sparse')
   % partial heterogeneity 'blocks' -- matricized tensor IxJQ has blocks on each slice
   G = zeros(I,J,K,Q,L);    n1_1 = floor(I*J*Q *1/2);    n1_2 = floor(I*J*Q *3/4);
   for k=1:K
      pos1_1 = randsample(I*J*Q,n1_1);   [I1_1,J1_1,Q1_1] = ind2sub([I,J,Q],pos1_1);
      pos1_2 = randsample(I*J*Q,n1_2);   [I1_2,J1_2,Q1_2] = ind2sub([I,J,Q],pos1_2);
      for q=1:length(Q1_1)
         G(I1_1(q), J1_1(q), k, Q1_1(q), 1) = -0.2 + 0.1*randn(1);
         G(I1_2(q), J1_2(q), k, Q1_2(q), 2) =  0.4 + 0.1*randn(1);
      end
   end
elseif strcmpi(type_coeff,'base')
   % automatically generated in the function 'gen_data_TEST'
   G = NaN;
end
% remove singleton dimension
G = squeeze(G);

% generate synthetic data
data = gen_data_FUNC('I',I, 'J',J, 'K',K, 'Q',Q, 'T',T, 'L',L, 'R',R, 'G',G);

% save synthetic data
if save_file
   file_name = ['T',num2str(T),'_I',num2str(I),'_J',num2str(J),'_K',num2str(K),'_Q',num2str(Q),'_L',num2str(L),'.mat'];
   save(['simdata_',file_name],'-struct','data');
end


%% MCMC estimation

Xt = data.Xt;
Zt = data.Zt;

if (K == 1)
   HYPER = define_hyperparameters(L,R);
   OUT = Bayes_ZIL_T_MS_1l_FUNC(Xt,Zt, 'HYPER',HYPER, 'NumIter',NumIter,'burn',burn,'thin',thin,'irep',irep);
else
   HYPER = define_hyperparameters(L,R);
   OUT = Bayes_ZIL_T_MS_2l_FUNC(Xt,Zt, 'HYPER',HYPER, 'NumIter',NumIter,'burn',burn,'thin',thin,'irep',irep);
end
OUT.data = data;
OUT.NumIter = NumIter;
OUT.thin    = thin;
OUT.burn    = burn;

% save MCMC output
if save_file
   save(['results_',file_name,'.mat'],'-struct','OUT');
end

end
