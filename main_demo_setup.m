%% DEMO script for ZIL-T-MS model with vector covariate
% (1) load a synthetic dataset
% (2) estimate the ZIL-T-MS model
% (3) make some plots

clear;  close all;
clc;  

% path = matlab.desktop.editor.getActiveFilename;  sf=strfind(path,'\');
% restoredefaultpath;
% addpath(genpath(path(1:sf(end)))); cd(path(1:sf(end)));

disp('%%%%%%%%% Running script file:  main_demo.m   %%%%%%%%%');

addpath(genpath('./functions'));
addpath('./Input');



%% Estimation

% settings for generating synthetic data
load('simdata_T100_I10_J10_K1_Q3_L2.mat');


% define hyperparameters
L = 2;      % number of states of the hidden chain
R = 5;      % rank of the coefficient tensor
cbar = ones(L,L) * 4.0;    % parameter vector of the Dirichlet prior for each row l=1..L
                           % of the Markov chain transition matrix
arhobar = ones(L,1);       % shape parameter 'a' of the Beta prior for rho, in each state l=1..L
brhobar = ones(L,1);       % shape parameter 'b' of the Beta prior for rho, in each state l=1..L
ident_constr = true;       % impose identification constraint during MCMC?
HYPER = define_hyperparameters(L,R, 'cbar',cbar,'arhobar',arhobar,'brhobar',brhobar,...
                                    'ident_constr',ident_constr);


% generate initial values
thrhtd = 0.08;    % threshold on network density, used to initilize the latent state vector st
IterSA = 80;      % number of iterations of the simulated annealing
INIT = define_initial_values(Xt,Zt,HYPER, 'thrhtd',thrhtd,'IterSA',IterSA);


% settings for the MCMC algorithm
NumIter = 1000;   % number of MCMC samples to be stored
burn    = 100;    % number of MCMC samples to be discarded
irep    = 100;    % display computing time every 'irep' iterations

% generate data and estimate the model
OUT = Bayes_ZIL_T_MS_1l_TEST(Xt,Zt, 'NumIter',NumIter,'burn',burn,'irep',irep,...
                                    'HYPER',HYPER,'INIT',INIT);



%% Plots

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex','defaultLegendInterpreter','latex');
set(groot,'DefaultAxesFontSize',18,'DefaultTextFontSize',18);


% get number of states in the hidden chain
L = size(OUT.rhohat,1);

%%%% RHO - ZIL mixing parameter

% plot posterior distribution, trace plot, and ACF for the ZIL mixing parameter (RHO)
figure;
for l=1:L
   rl = OUT.rhohat(l,:)';  xl = length(rl);

   % posterior distribution
   subplot(3,L,l);
      histogram(rl,'Normalization','proba','FaceColor',[1,1,1]*.3*l,'EdgeAlpha',0);
      hold on; scatter(rho(l),0,50,'dk','filled'); % xline(mean(rl),'--k','linew',1);
      title(['$\rho_{',num2str(l),'}$'],'Interpreter','latex');
   % trace plot + progressive mean
   subplot(3,L,L+l);
      plot(rl,'Color',[1,1,1]*.8); xlim([1,xl]);
      hold on; plot(cumsum(rl)./(1:xl)','k'); hold off;
   % ACF
   subplot(3,L,2*L+l);
      autocorr(rl,50);  xlabel(''); ylabel(''); title(''); xlim([0,50]);
end



%%%% St - hidden Markov chain

% plot MAP estimate and trace plot for the hidden Markov chain (St)
figure;
% MAP estimate
subplot(2,2,1:2);
stairs(St + L,'-k','linew',1.0);
hold on; stairs(OUT.MAP_st,'-r','linew',1.0); ylim([0,2*L+2]); hold off;
ylabel('$s_t$');  xlabel('time');
set(gca,'YTick',1:2*L,'YTickLabel',repmat(num2cell(1:L),[1,2]));
legend({'true','estimate'},'Orientation','horizontal','location','north');
% trace plot
subplot(2,2,3:4);
contourf(OUT.sthat); colormap(flipud(gray)); colorbar;
ylabel('time');  xlabel('MCMC iterations');



%%%% G - coefficient tensor

% get true value of the coefficient tensor
Gtrue = G;

% get number of layers
if ndims(Xt) == 3
   K = 1;
elseif ndims(Xt) == 4
   K = size(Xt,3);
end

% randomly pick some entries of the coefficient tensor for detailed plots
NN = 8;   pos = unidrnd(I*J*K*Q,[NN,1]);

% collect entries of the coefficient tensor along MCMC iterations
Gentry = zeros(NN,L,NumIter);  GG = zeros(I*J*K*Q,NumIter,L);
for i=1:NumIter
   for l=1:L
      if (K==1)
         tmp = ktensor_mod({OUT.gamma1hat(:,:,l,i),OUT.gamma2hat(:,:,l,i),OUT.gamma3hat(:,:,l,i)});
      else
         tmp = ktensor_mod({OUT.gamma1hat(:,:,l,i),OUT.gamma2hat(:,:,l,i),OUT.gamma3hat(:,:,l,i),OUT.gamma4hat(:,:,l,i)});
      end
      Gentry(:,l,i) = tmp(pos);
      GG(:,i,l) = tmp(:);
   end
end

%%%% (3) plot posterior distribution, trace plot, and ACF for randomly selected coefficients (G)
figure;
for p=1:NN
   for l=1:L
      sg = squeeze(Gentry(p,l,:));
      if (K==1)
         Gtl = Gtrue(:,:,:,l);
         [ii,jj,qq] = ind2sub([I,J,Q],pos(p));
         ylab = ['$G_{',num2str(ii),',',num2str(jj),',',num2str(qq),'}$'];
      else
         Gtl = Gtrue(:,:,:,:,l);
         [ii,jj,kk,qq] = ind2sub([I,J,K,Q],pos(p));
         ylab = ['$G_{',num2str(ii),',',num2str(jj),',',num2str(kk),',',num2str(qq),'}$'];
      end

      % posterior distribution
      subplot(NN,L*3,((p-1)*L+(l-1))*3+1);
         histogram(sg,'normalization','proba','FaceColor',[1,1,1]*.8,'EdgeAlpha',0);
         hold on; scatter(Gtl(pos(p)),0,40,'dk','filled'); hold off;
         ylabel(ylab,'Interpreter','latex');
      % trace plot + progressive mean
      subplot(NN,L*3,((p-1)*L+(l-1))*3+2);
         plot(sg,'Color',[1,1,1]*.8);
         hold on; plot(cumsum(sg)./(1:length(sg))','k'); hold off; xlim([1,NumIter]);
         if (p==1) && (l==1); title('\textbf{State 1}','Interpreter','latex'); end
         if (p==1) && (l==2); title('\textbf{State 2}','Interpreter','latex'); end
      % ACF
      subplot(NN,L*3,((p-1)*L+(l-1))*3+3);
         autocorr(sg,50);  xlabel(''); ylabel(''); title(''); xlim([0,50]);
   end
end
