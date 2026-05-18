function bt = cross_sectional_long_short_rebalance(signal, realizedReturns, dates, longQ, shortQ, tcBps, signalLag, annualization, rebalanceEvery)

[T, N] = size(signal);
weights = zeros(T, N);
currentW = zeros(1, N);

for t = 1:T
    sIdx = t - signalLag;
    if sIdx < 1
        weights(t,:) = currentW;
        continue;
    end

    % Rebalance only every k days
    if mod(t-1, rebalanceEvery) ~= 0
        weights(t,:) = currentW;
        continue;
    end

    s = signal(sIdx, :);
    valid = ~isnan(s);

    if nnz(valid) < max(3, ceil((longQ + shortQ) * N))
        weights(t,:) = currentW;
        continue;
    end

    sv = s(valid);
    qLong = quantile(sv, 1 - longQ);
    qShort = quantile(sv, shortQ);

    w = zeros(1, N);
    longMask = valid & (s >= qLong);
    shortMask = valid & (s <= qShort);

    nLong = nnz(longMask);
    nShort = nnz(shortMask);

    if nLong > 0
        w(longMask) =  1 / nLong;
    end
    if nShort > 0
        w(shortMask) = -1 / nShort;
    end

    currentW = w;
    weights(t,:) = currentW;
end

grossReturns = zeros(T,1);
turnover = zeros(T,1);

for t = 2:T
    grossReturns(t) = weights(t-1,:) * realizedReturns(t,:)';
    turnover(t) = sum(abs(weights(t,:) - weights(t-1,:)));
end

costRate = tcBps * 1e-4;
netReturns = grossReturns - costRate * turnover;

bt.dates = dates;
bt.weights = weights;
bt.grossReturns = grossReturns;
bt.netReturns = netReturns;
bt.turnover = turnover;
bt.annualization = annualization;
end