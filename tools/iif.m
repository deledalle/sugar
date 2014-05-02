function res = iif(x, y, z)

% iif - conditional statement
%
%   res = iif(cond, value1, value2)
%
%   cond, value1 and value2 are three matrices of the same size.
%   res contains entries of value1 where cond is true, value2 otherwise.
%
%   Copyright (c) 2014 Charles Deledalle

    if length(x(:)) > 1
        res = y .* ones(size(x));
        res(~x) = z(~x);
    else
        if x
            res = y;
        else
            res = z;
        end
    end
