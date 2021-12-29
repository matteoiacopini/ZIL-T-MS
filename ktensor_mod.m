function t = ktensor_mod(u)
%% ktensor_mod Modified function KTENSOR from Tensor Toolbox. Creates a tensor from its PARAFAC marginals.
%
%   K = ktensor_mod(U) creates a tensor from its PARAFAC marginals.
%   Assumes U is a cell array containing matrix Um in cell m and assigns the weight of each factor to be one.
% 
% INPUTS
%   u   cell array that contains matrix Um in cell m.
%       Each matrix Um is of size (Im x R), where R is the PARAFAC rank.
%
% OUTPUTS
%   t   tensor reconstructed from the rank-R PARAFAC marginals in input.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% u = varargin{1};
t.lambda = ones(size(u{1},2),1);
t.u = u;
    
% % Check that each Um is indeed a matrix
% for i = 1 : length(t.u)
%     if ndims(t.u{i}) ~= 2
% 	error(['Matrix U' int2str(i) ' is not a matrix!']);
%     end
% end

% % Size error checking
% k = length(t.lambda);
% for i = 1 : length(t.u)            
%     if  size(t.u{i},2) ~= k
%        error(['Matrix U' int2str(i) ' does not have ' int2str(k) ' columns.']);
%     end
% end

% t = class(t, 'ktensor');
% return;

%% FULL Convert a ktensor to a (dense) tensor.
%
%   T = FULL(C) converts a ktensor to a (dense) tensor.
%
%   Examples
%   X = ktensor([3; 2], rand(4,2), rand(5,2), rand(3,2));
%   Y = full(A) %<-- equivalent dense tensor
%
%   See also KTENSOR, TENSOR, KTENSOR/DOUBLE.

% sz = size(t);
% data = t.lambda' * khatrirao(t.u,'r')';

% KHATRIRAO Khatri-Rao product of matrices.
% sz = cellfun(@(x) size(x, 1), t.u);
sz = zeros(1,length(t.u));   for i=1:length(t.u); sz(i)=size(t.u{i},1); end
A = t.u;
matorder = length(A):-1:1;
% ncols = cellfun(@(x) size(x, 2), A);
ncols = zeros(1,length(A));   for i=1:length(A); ncols(i)=size(A{i},2); end
N = ncols(1);
P = A{matorder(1)};
for i = matorder(2:end)
    P = bsxfun(@times, reshape(A{i},[],1,N),reshape(P,1,[],N));
end
khatrirao_t_u = reshape(P,[],N);

data = t.lambda' * khatrirao_t_u';
t = reshape(data,sz);
% t = tensor(data,sz);

end


%KTENSOR Class for Kruskal tensors (decomposed).
%
%KTENSOR Methods:
%   arrange      - Arranges the rank-1 components of a ktensor.
%   datadisp     - Special display of a ktensor.
%   disp         - Command window display for a ktensor.
%   display      - Command window display for a ktensor.
%   double       - Convert a ktensor to a double array.
%   end          - Last index of indexing expression for ktensor.
%   extract      - Creates a new ktensor with only the specified components.
%   fixsigns     - Fix sign ambiguity of a ktensor.
%   full         - Convert a ktensor to a (dense) tensor.
%   innerprod    - Efficient inner product with a ktensor.
%   isequal      - True if each datum of two ktensor's are numerically equal.
%   issymmetric  - Verify that a ktensor X is symmetric in all modes.
%   ktensor      - Tensor stored as a Kruskal operator (decomposed).
%   minus        - Binary subtraction for ktensor.  
%   mtimes       - Implement A*B (scalar multiply) for ktensor.
%   mttkrp       - Matricized tensor times Khatri-Rao product for ktensor.
%   ncomponents  - Number of components for a ktensor.
%   ndims        - Number of dimensions for a ktensor.
%   norm         - Frobenius norm of a ktensor.
%   normalize    - Normalizes the columns of the factor matrices.
%   nvecs        - Compute the leading mode-n vectors for a ktensor.
%   permute      - Permute dimensions of a ktensor.
%   plus         - Binary addition for ktensor.
%   redistribute - Distribute lambda values to a specified mode. 
%   score        - Checks if two ktensors match except for permutation.
%   size         - Size of ktensor.
%   subsasgn     - Subscripted assignement for ktensor.
%   subsref      - Subscripted reference for a ktensor.
%   symmetrize   - Symmetrize a ktensor X in all modes.
%   times        - Element-wise multiplication for ktensor.
%   tocell       - Convert X to a cell array.
%   ttm          - Tensor times matrix for ktensor.
%   ttv          - Tensor times vector for ktensor.
%   uminus       - Unary minus for ktensor. 
%   uplus        - Unary plus for a ktensor. 
%
% See also
%   TENSOR_TOOLBOX
