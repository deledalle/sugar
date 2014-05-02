function [L,e] = compute_operator_trace(A,n)

% compute_operator_trace - compute operator trace
%
%   [L,e] = compute_operator_trace(A,n);
%
%   Copyright (c) 2014 Charles Deledalle

state = deterministic('on');

if length(n)==1
    u = randn(n,30);
else
    u = n;
end
L = 0;
for i = 1:size(n, 2)
    L = L + u(:, i)' * A(u(:, i));
end
L = L / size(n, 2);

deterministic('off', state);

