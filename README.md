# ZIL-T-MS

This archive contains data and codes for replicating the analysis in
Billio, M, Casarin, R., and Iacopini, M. (2022), _Bayesian Markov Switching Tensor Regression for Time-varying Networks_, Journal of the American Statistical Association 119(545), 109–121 (https://doi.org/10.1080/01621459.2022.2102502)


### Scripts description

Tested on MATLAB 2021b and MATLAB 2022a

The following MATALB script files are used to estimate the ZIL-T-MS model on the datasets studied in the main paper (Section 3, Section 4, and Supplement):

* main_financial.m
<br> Estimates the ZIL-T-MS model on the financial network dataset.

* main_email.m
<br> Estimates the ZIL-T-MS model on the email network dataset.

* main_simulation.m
<br> Generates a synthetic dataset and estimates the ZIL-T-MS model.



The following MATALB script file illustrates the use of the estimation procedure for the ZIL-T-MS model:

* main_demo.m
<br> Demo script, loads a synthetic dataset and estimates the ZIL-T-MS model.

* main_demo_setup.m
<br> Demo script, loads a synthetic dataset, sets the hyper-parameters of the prior and the initial values of the MCMC procedure, then estimates the ZIL-T-MS model.


&nsp;


### Functions
-------------------------------------------------------------------------------

The MATALB codes for running the Bayesian Zero-Inflated Logit Tensor model with Markov Switching coefficients (ZIL-T-MS) are:

* Bayes_ZIL_T_MS_2l_edgespec_FUNC.m
  <br> Multi-layer and edge-specific covariates -- Estimates the Bayesian ZIL-T-MS model, when for each time t=1,...,T the observed binary arrays Xt are of size (I,J,K), with K > 1, and the covariates Zt are edge-specific, that is of size (I,J,K,Q).

* Bayes_ZIL_T_MS_2l_FUNC.m
  <br> Multi-layer and common covariates -- Estimates the Bayesian ZIL-T-MS model, when for each time t=1,...,T the observed binary arrays Xt are of size (I,J,K), with K > 1, and the covariates Zt are common, that is of size (Q,1).

* Bayes_ZIL_T_MS_1l_edgespec_FUNC.m
  <br> Single-layer and edge-specific covariates -- Estimates the Bayesian ZIL-T-MS model, when for each time t=1,...,T the observed binary arrays Xt are of size (I,J) and the covariates Zt are edge-specific, that is of size (I,J,K,Q).

* Bayes_ZIL_T_MS_1l_FUNC.m
  <br> Single-layer and common covariates -- Estimates the Bayesian ZIL-T-MS model, when for each time t=1,...,T the observed binary arrays Xt are of size (I,J) and the covariates Zt are common, that is of size (Q,1).


### USAGE
The functions require two mandatory input arguments (Xt, Zt) that should be arranged as numerical arrays as follows:
- Xt (single-layer, that is matrix data):
   <br> binary array of size (I,J,T), such that Xt(:,:,t) represents the adjacency matrix at time t=1,...,T
- Xt (multi-layer, that is tensor data):
   <br> binary array of size (I,J,K,T), such that Xt(:,:,:,t) represents the adjacency tensor at time t=1,...,T
- Zt (common covariates):
   <br> array of size (Q,T), such that Zt(:,t) represents the vector covariates at time t=1,...,T
- Zt (edge-specific covariates):
   <br> array of size (I,J,K,Q,T), such that Zt(:,:,:,:,t) represents the tensor covariates at time t=1,...,T

Then, according to the input data, the estimation of the ZIL-T-MS model is performed as described below, where the default settings of the MCMC are used (alternative values of the MCMC parameters can be provided as name-value pairs):
- If Xt (matrix data) and Zt (common covariates)
  <br> OUT = Bayes_ZIL_T_MS_1l_FUNC(Xt,Zt)
- If Xt (matrix data) and Zt (edge-specific covariates)
  <br> OUT = Bayes_ZIL_T_MS_1l_edgespec_FUNC(Xt,Zt)
- If Xt (tensor data) and Zt (common covariates)
  <br> OUT = Bayes_ZIL_T_MS_2l_FUNC(Xt,Zt)
- If Xt (tensor data) and Zt (edge-specific covariates)
  <br> OUT = Bayes_ZIL_T_MS_2l_edgespec_FUNC(Xt,Zt)


NOTE
Use   help 'name_function'
to get further information about the arguments (mandatory and optional) of the
function 'name_function'.





-------------------------------------------------------------------------------

               %%%%%%%%%%%%%%   Other FUNCTIONS   %%%%%%%%%%%%%%

-------------------------------------------------------------------------------

The following functions can be used to set the hyperparameters of the model and
prior distributions, and to initialize the parameters of the ZIL-T-MS model:

* define_hyperparameters.m
Defines the hyperparameters for the model, setting them to default values if not
provided by the user.

* define_initial_values.m
Generates the initial values for the MCMC procedure.


NOTE
Use   help 'name_function'   to get further information about the arguments (mandatory and optional) of the function 'name_function'.



-------------------------------------------------------------------------------


The main functions used to generate synthetic datasets from the model are:

* gen_data_FUNC.m
  <br> main wrapper function that generates a synthetic dataset, starting from the hyperparameters provided by the user or, in absence of inputs from the user, using default hyperparameters values.

* DGP_ZIL_MS_tensor.m
  <br> Generate a synthetic dataset from the ZIL-T-MS model with K=1 and common covariates.

* DGP_ZIL_MS_tensor_input_G.m
  <br> Generate a synthetic dataset from the ZIL-T-MS model with K=1 and common covariates, starting from a given coefficient tensor provided as input.

* DGP_ZIL_MS_tensor_K.m
  <br> Generate a synthetic dataset from the ZIL-T-MS model with K > 1 and common covariates.

* DGP_ZIL_MS_tensor_K_input_G.m
  <br> Generate a synthetic dataset from the ZIL-T-MS model with K > 1 and common covariates, starting from a given coefficient tensor provided as input.

* VAR_gdp_unif.m
  <br> Simulates a stationary VAR(p) process.

* simulation_FUNC.m
  <br> Generates a synthetic dataset and performs the estimation of the ZIL-T-MS model.


-------------------------------------------------------------------------------


Additional MATLAB codes files include:

* tensor_toolbox package
  <br> Contains several functions for tensor calculus. Available at http://www.kolda.net/post/new-tensor-toolbox-website/

* bwr.m
  <br> Returns a M-by-3 matrix containing red-white-blue colormap (corresponding to (+ 0 -) values).

* dirpdf.m
  <br> Evaluates the density of a Dirichlet distrtibution.

* dirrnd.m
  <br> Samples from a Dirichlet distribution.

* ffbsX.m
  <br> Forward-Filtering Backward-Sampling for ZIL-T-MS model with matrix observations and common covariates.

* ffbsX_edge_specific.m
  <br> Forward-Filtering Backward-Sampling for ZIL-T-MS model with matrix observations and edge-specific covariates.

* ffbsX_tensorData.m
  <br> Forward-Filtering Backward-Sampling for ZIL-T-MS model with tensor observations and common covariates.

* ffbsX_tensorData_edge_specific.m
  <br> Forward-Filtering Backward-Sampling for ZIL-T-MS model with tensor observations and edge-specific covariates.

* fnorm.m
  <br> Computes the norm of B (tensor, matrix or vector).

* gigrnd.m
  <br> Samples from a Generalized Inverse Gaussian distribution.

* Hamilton.m
  <br> Runs the Hamilton filter/smoother for Markov Switching/Mixture model.

* HMClambda.m
  <br> Draws lambda from its posterior full conditional distribution using a Hamiltonian Monte Carlo step.

* ktensor_mod.m
  <br> Creates a tensor from its PARAFAC marginals.

* MatInv.m
  <br> Computes the matrix inverse using numerical corrections if it is closeto singular.

* MatPosDef.m
  <br> Returns a Positive Definite version of a matrix, applying a numericalcorrection if necessary.

* MatSym.m
  <br> Returns a symmetric version of a square matrix.

* PGrnd00.m and PGrnd00_mex.mexmaci64 and PGrnd00_mex.mexw64
  <br> Draws one random number from the Polya-Gamma distribution with parameters a=1 and b=Z00, that is, PG(1,Z00).

* SimAnnMScl.m
  <br> Initialize the PARAFAC marginals via Simulated Annealing, for the case where the observed binary array is of size (I,J,K), with K=1, and the covariates are of size (Q,1), common to all entries.

* SimAnnMScl_edge_specific.m
  <br> Initialize the PARAFAC marginals via Simulated Annealing, for the case where the observed binary array is of size (I,J,K), with K=1, and the covariates are of size (I,J,Q), edge-specific.

* SimAnnMScl_tensorData.m
  <br> Initialize the PARAFAC marginals via Simulated Annealing, for the case where the observed binary array is of size (I,J,K), with K > 1, and the covariates are of size (Q,1), common to all entries.

* SimAnnMScl_tensorData_edge_specific.m
  <br> Initialize the PARAFAC marginals via Simulated Annealing, for the case where the observed binary array is of size (I,J,K), with K > 1, and the covariates are of size (I,J,K,Q), edge-specific.

* ttv_mod.m
  <br> Computes the product of tensor with a (column) vector.





### DATA
-------------------------------------------------------------------------------

The following .mat files include the data used in the applications (Section 4) and a synthetic dataset:

* Data_financial.mat
  <br> Contains the data for the application to a financial network:
   - Xt: binary adjacency tensor of the multilayer network. 
      <br> Each entry Xt(i,j,k,t) = 1 represents a linkage between institutions i and j, on layer k, at time t. Layer k=1 represents the return network, layer k=2 the volatility network.
   - Zt: covariate tensor with Q=8 edge-specific covariates.
      <br> Each entry Zt(:,:,:,q,t) represents the value of the q-th covariate at time t:
          <br>  q=1 is a constant;
          <br>  q=2 is the lagged degree of layer 1;
          <br>  q=3 is the DVX;
          <br>  q=4 is the lagged DSTX;
          <br>  q=5 is the lagged CRS;
          <br>  q=6 is the lagged TRS;
          <br>  q=7 is the lagged MOM;
          <br>  q=8 is the RC.

* Data_email.mat
  <br> Contains the data for the application to an email network from EUcore (available at http://snap.stanford.edu/data/email-Eu-core-temporal.html). See also Ashwin Paranjape, Austin R. Benson, and Jure Leskovec. "Motifs in Temporal Networks.", In Proceedings of the Tenth ACM International Conference on Web Search and Data Mining, 2017.
 - Xt: binary adjacency tensor of the multilayer network
       <br> Each entry Xt(i,j,k,t) = 1 represents an email between researchers i and j, on layer k, at time t. Layer k=1 represents Department 3, layer k=2 Department 4.
 - Zt: covariate vector with Q=3 common covariates.
       <br> Each entry Zt(q,t) represents the value of the q-th covariate at time t:
         <br> q=1 is a constant;
         <br> q=2 is the lagged degree of layer 1;
         <br> q=3 is the lagged degree of layer 2.


* simdata_T100_I10_J10_K1_Q3_L2.mat
  <br> Synthetic dataset generated from the model given a specific coefficient tensor G as input.
