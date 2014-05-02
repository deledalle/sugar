function res = mycellfunc(f, c)

% mycellfunc - apply function to each cell of a cell array
%
%   Copyright (c) 2014 Charles Deledalle

    for k = 1:length(c)
        res{k} = f(c{k});
    end
