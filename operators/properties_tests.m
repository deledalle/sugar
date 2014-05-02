function op = properties_tests(op, N1, N2, N3)

% properties_test - test the properties of an operator and adds
%                   accordingly some fields into it
%
%   op = properties_tests(op, N1, N2, N3)
%
%   op is the operator to test and refine
%   N1, N2, N3 are the dimension of the object that op applies to
%   N2 and N3 are optionals.
%
%   Copyright (c) 2014 Charles Deledalle

    state = deterministic('on');

    global silent;
    silent = ~isempty(silent) && sum(abs(silent)) > 0;
    if ~exist('N2')
        N2 = 1;
    end
    if ~exist('N3')
        N3 = 1;
    end
    if isfield(op, 'AS')
        [P1 P2 P3] = size(op.A(randn(N1, N2, N3)));
        x = randn(N1, N2, N3);
        y = randn(P1, P2, P3);
        a = sum(sum(sum(op.AS(y).*x)));
        b = sum(sum(sum(y.*conj(op.A(x)))));
        if ~silent
            s = sprintf('  Test adjoint?\t\t<A x, y> = <x, AS y>?\t%s\t(error %g)', ...
                         iif(sum(sum(abs(a - b).^2)) < 10e-10, 'OK', 'Fail'), ...
                         sum(sum(abs(a - b).^2)));
            disp(s);
        end
        op.AAS_norm = compute_operator_norm(@(x) op.A(op.AS(x)), ...
                                            randn(P1, P2, P3));
        op.ASA_norm = compute_operator_norm(@(x) op.AS(op.A(x)), ...
                                            randn(N1, N2, N3));
        op.AAS_trace = compute_operator_trace(@(x) op.A(op.AS(x)), ...
                                              randn(P1, P2, P3));
        op.ASA_trace = compute_operator_trace(@(x) op.AS(op.A(x)), ...
                                              randn(N1, N2, N3));
    end
    if isfield(op, 'A_PseudoInv')
        x = randn(N1, N2);
        a = op.A(op.A_PseudoInv(op.A(x)));
        b = op.A(x);
        if ~silent
            s = sprintf(['  Test A^+?\t\ttest int.\t\t%s\t(error %g)'], ...
                         iif(sum(sum((a - b).^2)) < 10e-10, 'OK', 'Fail'), ...
                         sum(sum((a - b).^2)));
            disp(s);
        end
        x = randn(P1, P2, P3);
        a = op.A_PseudoInv(op.A(op.A_PseudoInv(x)));
        b = op.A_PseudoInv(x);
        if ~silent
            s = sprintf(['  Test A^+?\t\ttest ext.\t\t%s\t(error %g)'], ...
                         iif(sum(sum((a - b).^2)) < 10e-10, 'OK', 'Fail'), ...
                         sum(sum((a - b).^2)));
            disp(s);
        end
    end
    if isfield(op, 'AS')
        x1 = randn(P1, P2, P3);
        x2 = op.A(op.AS(x1));
        if ~silent
            s = sprintf('  Norm. tight frame?\tA AS x = x?\t\t%s\t(error %g)', ...
                         iif(sum(sum((x1 - x2).^2)) < 10e-10, 'Yes', 'No'), ...
                         sum(sum((x1 - x2).^2)));
            disp(s);
        end
        a1 = randn(N1, N2, N3);
        a2 = op.AS(op.A(a1));
        op.normalized_tight_frame = sum(sum((x1 - x2).^2)) < 10e-10;
        if ~silent
            s = sprintf('  Unitary?\t\tand AS A x = x?\t\t%s\t(error %g)', ...
                         iif(op.normalized_tight_frame && sum(sum((a1 - a2).^2)) < 10e-10, 'Yes', 'No'), ...
                         sum(sum((a1 - a2).^2)));
            disp(s);
        end
        op.unitary = op.normalized_tight_frame && sum(sum((x1 - x2).^2)) < 10e-10;
    end
    if isfield(op, 'AS') && isfield(op, 'AAS_PseudoInv')
        x = randn(P1, P2, P3);
        a = op.A(op.AS(op.AAS_PseudoInv(op.A(op.AS(x)))));
        b = op.A(op.AS(x));
        if ~silent
            s = sprintf(['  Test (A AS)^+?\ttest int.\t\t%s\t(error %g)'], ...
                         iif(sum(sum((a - b).^2)) < 10e-10, 'OK', 'Fail'), ...
                         sum(sum((a - b).^2)));
            disp(s);
        end
        a = op.AAS_PseudoInv(op.A(op.AS(op.AAS_PseudoInv(x))));
        b = op.AAS_PseudoInv(x);
        if ~silent
            s = sprintf(['  Test (A AS)^+?\ttest ext.\t\t%s\t(error %g)'], ...
                         iif(sum(sum((a - b).^2)) < 10e-10, 'OK', 'Fail'), ...
                         sum(sum((a - b).^2)));
            disp(s);
        end

        op.ASA_PseudoInv_norm = ...
            compute_operator_norm(@(x) op.AAS_PseudoInv(x), ...
                                  randn(P1, P2, P3));
        op.AAS_PseudoInv_trace = ...
            compute_operator_trace(@(x) op.AAS_PseudoInv(x), ...
                                   randn(P1, P2, P3));
    end
    if isfield(op, 'IdPAAS_Inv')
        x1 = randn(P1, P2, P3);
        x2 = op.IdPAAS_Inv(x1 + op.A(op.AS(x1)));
        if ~silent
            s = sprintf(['  Test (Id + A AS)^-1?\ttest right\t\t%s\t(error %g)'], ...
                         iif(sum(sum((x1 - x2).^2)) < 10e-10, 'OK', 'Fail'), ...
                         sum(sum((x1 - x2).^2)));
            disp(s);
        end
        x1 = randn(P1, P2, P3);
        x2 = op.IdPAAS_Inv(x1) + op.A(op.AS(op.IdPAAS_Inv(x1)));
        if ~silent
            s = sprintf(['  Test (Id + A AS)^-1?\ttest left\t\t%s\t(error %g)'], ...
                         iif(sum(sum((x1 - x2).^2)) < 10e-10, 'OK', 'Fail'), ...
                         sum(sum((x1 - x2).^2)));
            disp(s);
        end
    end
    if isfield(op, 'IdPtauAAS_Inv')
        x1 = randn(P1, P2, P3);
        tau = 1 + randn;
        x2 = op.IdPtauAAS_Inv(x1 + tau * op.A(op.AS(x1)), tau);
        if ~silent
            s = sprintf(['  Test (Id + A AS)^-1?\ttest right\t\t%s\t(error %g)'], ...
                         iif(sum(sum((x1 - x2).^2)) < 10e-10, 'OK', 'Fail'), ...
                         sum(sum((x1 - x2).^2)));
            disp(s);
        end
        x1 = randn(P1, P2, P3);
        x2 = op.IdPtauAAS_Inv(x1, tau) + tau * ...
             op.A(op.AS(op.IdPtauAAS_Inv(x1, tau)));
        if ~silent
            s = sprintf(['  Test (Id + A AS)^-1?\ttest left\t\t%s\t(error %g)'], ...
                         iif(sum(sum((x1 - x2).^2)) < 10e-10, 'OK', 'Fail'), ...
                         sum(sum((x1 - x2).^2)));
            disp(s);
        end
    end

    deterministic('off', state);