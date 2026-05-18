function alpha = lasso_ista(A, y, lambda, maxIter, tol)
% LASSO_ISTA
% Solve:
%   min_a 0.5 * ||y - A a||_2^2 + lambda * ||a||_1
%
% Inputs:
%   A       : m x n dictionary
%   y       : m x 1 signal
%   lambda  : L1 penalty
%   maxIter : maximum iterations
%   tol     : stopping tolerance
%
% Output:
%   alpha   : n x 1 sparse coefficient vector

    if nargin < 4 || isempty(maxIter)
        maxIter = 200;
    end
    if nargin < 5 || isempty(tol)
        tol = 1e-6;
    end

    y = y(:);
    [~, n] = size(A);
    alpha = zeros(n,1);

    % Lipschitz constant L = ||A||_2^2
    L = estimate_lipschitz(A);
    step = 1 / L;

    for k = 1:maxIter
        alpha_old = alpha;

        grad = A' * (A * alpha - y);
        alpha = soft_threshold(alpha - step * grad, lambda * step);

        if norm(alpha - alpha_old) / max(1e-12, norm(alpha_old) + 1e-12) < tol
            break;
        end
    end
end

function L = estimate_lipschitz(A)
    v = randn(size(A,2),1);
    v = v / max(norm(v), 1e-12);

    for k = 1:20
        v = A' * (A * v);
        nv = norm(v);
        if nv < 1e-12
            L = 1;
            return;
        end
        v = v / nv;
    end

    L = v' * (A' * (A * v));
    L = max(L, 1e-8);
end