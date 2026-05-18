function [resid, commonPart, model] = remove_common_factors(rets, numFactors)
mu = mean(rets, 1);
X = rets - mu;
[U, S, V] = svd(X, 'econ');
k = min([numFactors, size(U,2), size(V,2)]);
commonPart = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';
resid = X - commonPart;
model.mu = mu; model.U = U(:,1:k); model.S = S(1:k,1:k); model.V = V(:,1:k);
end
