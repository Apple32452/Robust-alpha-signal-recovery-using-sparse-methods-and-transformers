function retsCorrupt = inject_sparse_outliers(rets, frac, scale)
% INJECT_SPARSE_OUTLIERS Add sparse shocks to random entries of returns matrix

retsCorrupt = rets;
[T, N] = size(rets);

K = round(frac * T * N);
if K <= 0
    return;
end

idx = randperm(T * N, K);
idx = idx(:);

sigma = std(rets(:), 'omitnan');
if isnan(sigma) || sigma < 1e-12
    sigma = 1;
end

shocks = scale * sigma .* sign(randn(K,1));
retsCorrupt(idx) = retsCorrupt(idx) + shocks;
end