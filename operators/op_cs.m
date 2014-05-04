function op = op_cs(N1, N2, P)

% op_subsampling - create a compressed sensing operator
%
%   op = op_cs(N1, N2, P)
%
%   N1, N2 are the dimension of the image to analyse.
%   P is the compression level [0, N1 * N2]
%
%   Copyright (c) 2014 Charles Deledalle

% Degradation matrix
N = N1 * N2;
perm1 = randperm(N);
perm2 = [1 randperm(N-1)+1];

subs = @(x, I) x(I);
A  = @(f) ...
    [subs(subs(dct2(subs(f(:), perm1)), perm2), 1:P)];
AS = @(y) ...
    reshape(ups(idct2(ups(ups(y, 1:P, N), perm2, N)), ...
                perm1, N), N1, N2);
AAS_PseudoInv = @(y) y;
IdPtauAAS_Inv = @(x, tau) x / (1 + tau);

% Projection
Pi = @(x) AS(A(x));
ML = @(y) AS(y);

% Vectorize
op.A = vect(A, N1, N2);
op.AS = vect(AS, P, 1);
op.AAS_PseudoInv = vect(AAS_PseudoInv, P, 1);
op.Pi = vect(Pi, N1, N2);
op.ML = vect(ML, P, 1);
op.IdPtauAAS_Inv = IdPtauAAS_Inv;

global silent;
silent = ~isempty(silent) && sum(abs(silent)) > 0;
if ~silent
    disp(['Test compressed sensing operator']);
end
op = properties_tests(op, N1*N2);

function r = ups(v, I, N)

r = zeros(N, 1);
r(I) = v;
