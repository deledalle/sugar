function varargout = call(f, varargin)

% call - call a function with supplied arguments
%
%   f is the function handle
%   varargin are the arguments of f
%   varargout are the returns of f
%
%   Copyright (c) 2014 Charles Deledalle

    varargout{:} = f(varargin{:});