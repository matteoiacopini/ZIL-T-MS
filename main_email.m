%% Estimate the ZIL-T-MS model on an Email dataset

clear;  close all;
clc;  

% path = matlab.desktop.editor.getActiveFilename;  sf=strfind(path,'\');
% restoredefaultpath;
% addpath(genpath(path(1:sf(end)))); cd(path(1:sf(end)));

disp('%%%%%%%%% Running script file:  main_email.m   %%%%%%%%%');

addpath(genpath('./functions'));
addpath('./Input');

% LOAD data
load('Data_email.mat');


% MCMC settings
L       = 3;      % number of states
R       = 5;      % CP tensor rank (actual is Rbar = R/2, due to constraint on coefficient tensor)
NumIter = 3000;   % number of MCMC samples to be stored
burn    = 1000;   % number of MCMC samples to be discarded


% define hyperparameters
HYPER = define_hyperparameters(L,R,'constr_g3',false,'ident_constr',true);

% run MCMC algorithm
OUT = Bayes_ZIL_T_MS_2l_FUNC(Xt,Zt, 'R',R,'L',L,'NumIter',NumIter,'burn',burn, 'HYPER',HYPER);



%% Plots

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex','defaultLegendInterpreter','latex');
set(groot,'DefaultAxesFontSize',18,'DefaultTextFontSize',18);


%%%% RHO - ZIL mixing parameter

% plot posterior distribution and trace plot for the ZIL mixing parameter (RHO)
figure;
for l=1:L
   rl = OUT.rhohat(l,:)';  xl = length(rl);

   % posterior distribution
   subplot(2,L,l);
      histogram(rl,'Normalization','proba','FaceColor',[1,1,1]*.3*l,'EdgeAlpha',0);
      xline(mean(rl),'--k','linew',1);
      title(['$\rho_{',num2str(l),'}$'],'Interpreter','latex');
   % trace plot + progressive mean
   subplot(2,L,L+l);
      plot(rl,'Color',[1,1,1]*.8); xlim([1,xl]);
      hold on; plot(cumsum(rl)./(1:xl)','k'); hold off;
end



%%%% St - hidden Markov chain

% plot MAP estimate and trace plot for the hidden Markov chain (St)
figure;
% MAP estimate
subplot(2,2,1:2);
hold on; stairs(OUT.MAP_st,'-r','linew',1.0); ylim([0,L+1]); hold off;
ylabel('$s_t$');  xlabel('time');
set(gca,'YTick',1:L,'YTickLabel',repmat(num2cell(1:L),[1,2]));
legend({'estimate'},'Orientation','horizontal','location','north');
% trace plot
subplot(2,2,3:4);
contourf(OUT.sthat); colormap(flipud(gray)); colorbar;
ylabel('time');  xlabel('MCMC iterations');



%%%% G - coefficient tensor

% selects the index of the covariate to display (q = 1,2,3)
% and the response layer (k = 1,2)
q = 2;
k = 1;

% collect entries of the coefficient tensor along MCMC iterations
GG = zeros(NumIter,I,J,L);
for i=1:NumIter
   for l=1:L
      if (K==1)
         tmp = ktensor_mod({OUT.gamma1hat(:,:,l,i),OUT.gamma2hat(:,:,l,i),OUT.gamma3hat(q,:,l,i)});
      else
         tmp = ktensor_mod({OUT.gamma1hat(:,:,l,i),OUT.gamma2hat(:,:,l,i),OUT.gamma3hat(k,:,l,i),OUT.gamma4hat(q,:,l,i)});
      end
      GG(i,:,:,l) = tmp;
   end
end

Ghat = zeros(I,J,L);
for l=1:L
   Ghat(:,:,l) = squeeze( mean(GG(:,:,:,l),1) );
end

%%%% (3) plot estimated impact of covariate q on layer k, across states l=1..L
figure;
for l=1:L
   ylab = ['$G_{:,:,',num2str(k),',',num2str(q),',',num2str(l),'}$'];
   s=subplot(L,1,l);
   imagesc(Ghat(:,:,l)); colormap(s,bwr);
   ylabel(ylab,'Interpreter','latex');
end
