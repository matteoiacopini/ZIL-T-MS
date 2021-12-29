%% DEMO script for ZIL-T-MS model with vector covariate
% (1) generate a synthetic dataset
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



%% Simulation and estimation

% settings for generating synthetic data
I = 10;   J = 10;   K = 1;    % size of observed binary array Xt (I,J,K) at each time t=1..T
Q = 3;     % number of covariates
T = 100;   % number of time series observations
R = 5;     % PARAFAC rank of the coefficient tensor
L = 2;     % number of states for the hidden Markov chain

% code for setting the coefficient tensor -- see Section 6.XX of the Supplement
%  - 'block'   block-wise structure 
%  - 'row'     constant coefficients across columns
%  - 'col'     constant coefficients across rows
%  - 'sparse'  sparse random structure
%  - 'base'    random structure drawn from the prior
type_coeff = 'block';   

% settings for the MCMC algorithm
NumIter = 1000;   % number of MCMC samples to be stored
burn    = 100;    % number of MCMC samples to be discarded
irep    = 100;    % display computing time every 'irep' iterations

% generate data and estimate the model
[OUT,data] = simulation_FUNC('I',I,'J',J,'K',K,'Q',Q,'T',T,'R',R,'L',L, 'type_coeff',type_coeff, ...
                                  'NumIter',NumIter,'burn',burn,'irep',irep);



%% Plots

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex','defaultLegendInterpreter','latex');
set(groot,'DefaultAxesFontSize',18,'DefaultTextFontSize',18);


%%%% RHO - ZIL mixing parameter

% plot posterior distribution, trace plot, and ACF for the ZIL mixing parameter (RHO)
figure;
for l=1:L
   rl = OUT.rhohat(l,:)';  xl = length(rl);

   % posterior distribution
   subplot(3,L,l);
      histogram(rl,'Normalization','proba','FaceColor',[1,1,1]*.3*l,'EdgeAlpha',0);
      hold on; scatter(data.rho(l),0,50,'dk','filled'); % xline(mean(rl),'--k','linew',1);
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
stairs(data.St + L,'-k','linew',1.0);
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
Gtrue = data.G;

% selects the entries of the coefficient tensor to display
if (K == 1)
   pos_all = [1, 2, 2;   % coordinates (i,j,q)
              2, 1, 3;
              3, 2, 2;
              ];
   pos = sub2ind([I,J,Q], pos_all(:,1),pos_all(:,2),pos_all(:,3));
elseif (K > 1)
   pos_all = [1, 2, 1, 2;   % coordinates (i,j,k,q)
              2, 1, 1, 3;
              3, 2, 2, 2;
              ];
   pos = sub2ind([I,J,K,Q], pos_all(:,1),pos_all(:,2),pos_all(:,3),pos_all(:,4));
end

% alternatively, randomly pick some entries of the coefficient tensor for detailed plots
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
