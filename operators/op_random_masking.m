function op = op_random_masking(n1, n2, P)

% op_subsampling - create a vertical subsampling operator
%
%   op = op_subsampling(N1, N2, P)
%
%   N1, N2 are the dimension of the image to analyse.
%   P is the number of non zero elements.
%
%   Copyright (c) 2014 Charles Deledalle

    I = randperm(n1*n2); I = I(1:P); I = I(:);
    A  = @(x) x(I);
    AS = @(y) reshape( accumarray(I, y, [n1*n2 1], @sum), [n1 n2]);
    AAS_PseudoInv = @(y) y;
    Pi = @(x) AS(A(x));
    ML = @(x) AS(x);

    op.A = vect(A, n1, n2);
    op.AS = vect(AS, P, 1);
    op.AAS_PseudoInv = vect(AAS_PseudoInv, P, 1);
    op.Pi = vect(Pi, n1, n2);
    op.ML = vect(ML, P, 1);

    global silent;
    silent = ~isempty(silent) && sum(abs(silent)) > 0;
    if ~silent
        disp(['Test random masking']);
    end
    op = properties_tests(op, n1 * n2);
