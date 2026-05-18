function xhat = omp(A, y, K, tol)
% OMP: Orthogonal Matching Pursuit
%
% Solve approximately:
%   min ||y - A x||_2  subject to ||x||_0 <= K
%
% Inputs:
%   A   : m x n dictionary
%   y   : m x 1 signal
%   K   : sparsity level
%   tol : residual tolerance (optional)
%
% Output:
%   xhat : n x 1 sparse coefficient vector

    if nargin < 4 || isempty(tol)
        tol = 1e-6;
    end

    [~, n] = size(A);
    y = y(:);

    xhat = zeros(n,1);
    r = y;
    support = [];

    for k = 1:K
        corr = abs(A' * r);
        [~, idx] = max(corr);

        if ismember(idx, support)
            break;
        end

        support = [support; idx];

        As = A(:, support);
        alpha_s = As \ y;
        r = y - As * alpha_s;

        if norm(r) < tol
            break;
        end
    end

    if ~isempty(support)
        xhat(support) = alpha_s;
    end
end