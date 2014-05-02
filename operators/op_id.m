function op = op_id(N1, N2)

% op_conv - create the identity operator
%
%   op = op_conv(N1, N2)
%
%   N1, N2 are the dimension of the image to analyse.
%
%   Copyright (c) 2014 Charles Deledalle

    op.A = @(a) a;
    op.AS  = @(x) x;
    op.A_PseudoInv  = @(x) x;
    op.IdPAAS_Inv = @(x) x / 2;
    op.ML = @(y) y;
    op.Pi = @(x) x;
    op.AAS_PseudoInv = @(x) x;

    global silent;
    silent = ~isempty(silent) && sum(abs(silent)) > 0;
    if ~silent
        disp(['Test identity operator']);
    end
    op = properties_tests(op, N1*N2, 1);
