function c = ttv_mod(a,v,dims)
%% ttv_mod Modified function TTV from Tensor Toolbox. Computes Tensor times vector product for tensors of ORDER <= 5
%
%   C = TTV(A,V,DIMS) computes the product of tensor A with a (column) vector V.
%   The integer DIMS specifies the dimension in A along which V is multiplied.
%   If size(V) = [I,1], then A must have size(A,DIMS) = I.
%   Note that ndims(C) = ndims(A) - 1 because the DIMS-th dimension is removed.
%
%   C = TTV(A,{V1,V2,V3,...}) computes the product of tensor A with a
%   sequence of vectors in the cell array.  The products are computed
%   sequentially along all dimensions (or modes) of A. The cell array
%   contains ndims(A) vectors.
%
%   C = TTV(A,{V1,V2,V3,...},DIMS) computes the sequence of tensor-vector
%   products along the dimensions specified by DIMS.
%
% INPUTS
%   a           tensor
%   v      Ix1  vector
%   dims   1x1  integer 1 <= dims <= ndims(a) indicating the dimension in a along which v is multiplied.
% 
% OUTPUTS
%   c    tensor of order ndims(A) - 1, resulting from the multiplication
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% % Check the number of arguments
% if (nargin < 2)
%     error('TTV requires at least two arguments.');
% end

% % Check for 3rd argument
% if ~exist('dims','var')
%     dims = [];
% end

% Check that 2nd argument is cell array. If not, recall with v as a
% cell array with one element.
if ~iscell(v)
%     c = ttv(a,{v},dims);
%     return;
   vv = {v};
end

% Get sorted dims and index for multiplicands
% [dims,vidx] = tt_dimscheck(dims,ndims(a),numel(v));
P = length(dims);
% Reorder dims from smallest to largest (this matters for the vector multiplicand case, where the order affects the result)
[sdims,sidx] = sort(dims,'ascend');
% Check sizes to determine how to index multiplicands
if (P == numel(vv))
   % Case 1: Number of items in dims and number of multiplicands are equal;
   % therefore, index in order of how sdims was sorted.
   vidx = sidx;
else
   % Case 2: Number of multiplicands is equal to the number of
   % dimensions in the tensor; therefore, index multiplicands by dimensions specified in dims argument.
   vidx = sdims;  % index multiplicands by (sorted) dimension
end
dims = sdims;


% % Check that each multiplicand is the right size.
% for i = 1:numel(dims)
%     if ~isequal(size(v{vidx(i)}),[size(a,dims(i)) 1])
%         error('Multiplicand is wrong size');
%     end
% end

% if exist('tensor/ttv_single','file') == 3
%     c = a;
%     for i = numel(dims) : -1 : 1
%         c = ttv_single(c,v{vidx(i)},dims(i));
%     end
%     return;
% end

% Extract the MDA
% c = a.data;
c = a;

% Permute it so that the dimensions we're working with come last
remdims = setdiff(1:ndims(a),dims);
% if (ndims(a) > 1)
%     c = permute(c,[remdims, dims]);
% end

% Do each  multiply in sequence, doing the highest index first, which is important for vector multiplies.
n = ndims(a);
% sz = a.size([remdims dims]);
a_dims = size(a);
sz = a_dims([remdims, dims]);
for i = numel(dims) : -1 : 1
    c = reshape(c,prod(sz(1:n-1)),sz(n));
    c = c * vv{vidx(i)};
    n = n-1;
end

% If needed, convert the final result back to a tensor
% if (n > 0)
%     c = tensor(c,sz(1:n));
% end
if n == 2
   c = reshape(c,[sz(1),sz(2)]);
elseif n == 3
   c = reshape(c,[sz(1),sz(2),sz(3)]);
elseif n == 4
   c = reshape(c,[sz(1),sz(2),sz(3),sz(4)]);
end

end

