function opv = vect(op, N1, N2, N3, L)

% vect - create a vectorial operator from a matrix valued one
%
%   opv = vect(op, N1, N2, N3, L)
%
%   op is an operator.
%   N1, N2, N3 are the dimensions of the argument of the operator.
%   L should be 1.
%
%   Copyright (c) 2014 Charles Deledalle

    if nargin < 4
        N3 = 1;
    end
    if nargin < 5
        L = 1;
    end
    N = N1 * N2 * N3;
    [Q1, Q2, Q3] = size(op(ones(N1, N2, N3)));
    Q = Q1 * Q2 * Q3 / L;

    function y = op_wrapper(x)
        if size(x, 1) ~= N
            error('Vect: first dimension size inconsistency');
        end
        K = size(x, 2);
        if K == 1
            y = reshape(op(reshape(x, N1, N2, N3)), Q, L);
        else
            if L ~= 1
                error('Vect: not implemented yet');
            end
            y = zeros(Q, K);
            for k = 1:K
                y(:,k) = ...
                    y(:,k) + ...
                    reshape(op(reshape(x(:,k), N1, N2, N3)), Q, 1);
            end
        end
    end
    opv = @(x) op_wrapper(x);

end