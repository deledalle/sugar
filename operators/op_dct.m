function op = op_dct(N1, N2)

% op_subsampling - create a compressed sensing operator
%
%   op = op_cs(N1, N2, P)
%
%   N1, N2 are the dimension of the image to analyse.
%   P is the compression level [0, N1 * N2]
%
%   Copyright (c) 2014 Charles Deledalle

A = @(a) real(idct2(a));
AS = @(x) dct2(x)

IdPAAS_Inv = @(x) x/2;

op.A = vect(A, N1, N2);
op.AS = vect(AS, N1, N2);
op.A_PseudoInv = op.AS;
op.IdPAAS_Inv = vect(IdPAAS_Inv, N1, N2);

global silent;
silent = ~isempty(silent) && sum(abs(silent)) > 0;
if ~silent
    disp(['Test dct transform']);
end
op = properties_tests(op, N1 * N2);
