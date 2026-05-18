%% run_alpha_project.m
clear; clc; close all; rng(0);

% ---------------- Configuration ----------------
cfg.dataFolder = fullfile(pwd, 'data_small');
cfg.startDate = datetime(2010,1,1);

cfg.maxMissingFrac = 0.05;
cfg.winsorZ = 5.0;
cfg.numFactors = 3;

% Baseline mean-reversion signal
cfg.lookback = 20;

% DCT + Thresholding
cfg.window = 60;
cfg.lambda = 0.015;
cfg.numDctKeep = [];
cfg.standardizeWithinWindow = true;
cfg.reverseThreshSignal = true;

% DCT + OMP
cfg.ompK = 6;
cfg.reverseOmpSignal = true;

% DCT + LASSO
cfg.lassoLambda = 0.01;
cfg.reverseLassoSignal = true;

% Backtest
cfg.signalLag = 0;
cfg.longQuantile = 0.10;
cfg.shortQuantile = 0.10;
cfg.transactionCostBps = 5;
cfg.annualization = 252;
cfg.rebalanceEvery = 5;   % weekly-ish

% Outlier experiment
cfg.runOutlierExperiment = true;
cfg.outlierFrac = 0.01;
cfg.outlierScale = 8.0;

% Clean universe
wanted = ["aapl.us","adbe.us","amat.us","amd.us","amzn.us", ...
          "googl.us","intc.us","meta.us","msft.us","mu.us"];

% ---------------- Load data ----------------
if ~isfolder(cfg.dataFolder)
    error('Data folder not found: %s', cfg.dataFolder);
end

[dates, prices, symbols] = load_price_panel(cfg.dataFolder);
fprintf('Loaded %d assets with %d dates.\n', numel(symbols), numel(dates));

keepWanted = ismember(lower(symbols), wanted);
prices = prices(:, keepWanted);
symbols = symbols(keepWanted);

fprintf('After symbol filter: %d assets remain.\n', numel(symbols));
disp(symbols)

keepDates = dates >= cfg.startDate;
dates = dates(keepDates);
prices = prices(keepDates, :);

fprintf('After date filter (%s): %d dates remain.\n', datestr(cfg.startDate), numel(dates));

% ---------------- Returns ----------------
rets = compute_log_returns(prices);
retDates = dates(2:end);

missingFrac = mean(isnan(rets), 1);
keepAssets = missingFrac <= cfg.maxMissingFrac;
rets = rets(:, keepAssets);
symbols = symbols(keepAssets);

for j = 1:size(rets,2)
    x = rets(:,j);

    if all(isnan(x))
        continue;
    end

    mu = mean(x(~isnan(x)));
    x(isnan(x)) = mu;

    sigma = std(x);
    if sigma > 0
        x = max(min(x, cfg.winsorZ * sigma), -cfg.winsorZ * sigma);
    end

    rets(:,j) = x;
end

fprintf('Using %d assets after filtering.\n', numel(symbols));
disp(symbols)

% ---------------- Remove common factors ----------------
[resid, ~, ~] = remove_common_factors(rets, cfg.numFactors);

% ---------------- Baseline signal ----------------
baselineSignal = nan(size(resid));

for t = cfg.lookback:size(resid,1)
    win = resid(t-cfg.lookback+1:t, :);
    mu = mean(win, 1);
    sig = std(win, 0, 1);
    sig(sig < 1e-8) = 1;

    z = (resid(t,:) - mu) ./ sig;

    % mean-reversion baseline
    baselineSignal(t,:) = -z;
end

% ---------------- DCT + Thresholding ----------------
[threshSignal, ~] = rolling_dct_alpha(resid, cfg.window, cfg.lambda, ...
    cfg.numDctKeep, cfg.standardizeWithinWindow);

if cfg.reverseThreshSignal
    threshSignal = -threshSignal;
end

% ---------------- DCT + OMP ----------------
[ompSignal, ~] = rolling_dct_alpha_omp(resid, cfg.window, cfg.ompK, ...
    cfg.standardizeWithinWindow);

if cfg.reverseOmpSignal
    ompSignal = -ompSignal;
end

% ---------------- DCT + LASSO ----------------
[lassoSignal, ~] = rolling_dct_alpha_lasso(resid, cfg.window, cfg.lassoLambda, ...
    cfg.standardizeWithinWindow);

if cfg.reverseLassoSignal
    lassoSignal = -lassoSignal;
end

% ---------------- Cross-sectional z-scoring ----------------
baselineSignal = normalize_cross_section(baselineSignal);
threshSignal   = normalize_cross_section(threshSignal);
ompSignal      = normalize_cross_section(ompSignal);
lassoSignal    = normalize_cross_section(lassoSignal);

% ---------------- Backtests ----------------
btBase = cross_sectional_long_short_rebalance(baselineSignal, rets, retDates, ...
    cfg.longQuantile, cfg.shortQuantile, cfg.transactionCostBps, ...
    cfg.signalLag, cfg.annualization, cfg.rebalanceEvery);

btThresh = cross_sectional_long_short_rebalance(threshSignal, rets, retDates, ...
    cfg.longQuantile, cfg.shortQuantile, cfg.transactionCostBps, ...
    cfg.signalLag, cfg.annualization, cfg.rebalanceEvery);

btOMP = cross_sectional_long_short_rebalance(ompSignal, rets, retDates, ...
    cfg.longQuantile, cfg.shortQuantile, cfg.transactionCostBps, ...
    cfg.signalLag, cfg.annualization, cfg.rebalanceEvery);

btLasso = cross_sectional_long_short_rebalance(lassoSignal, rets, retDates, ...
    cfg.longQuantile, cfg.shortQuantile, cfg.transactionCostBps, ...
    cfg.signalLag, cfg.annualization, cfg.rebalanceEvery);

fprintf('\n=== Baseline residual reversal stats ===\n');
disp(performance_stats(btBase.netReturns, cfg.annualization));

fprintf('\n=== DCT + Thresholding stats ===\n');
disp(performance_stats(btThresh.netReturns, cfg.annualization));

fprintf('\n=== DCT + OMP stats ===\n');
disp(performance_stats(btOMP.netReturns, cfg.annualization));

fprintf('\n=== DCT + LASSO stats ===\n');
disp(performance_stats(btLasso.netReturns, cfg.annualization));

% ---------------- Main comparison plot ----------------
figure;
plot(btBase.dates, cumsum(btBase.netReturns), 'LineWidth', 1.2); hold on;
plot(btThresh.dates, cumsum(btThresh.netReturns), 'LineWidth', 1.2);
plot(btOMP.dates, cumsum(btOMP.netReturns), 'LineWidth', 1.2);
plot(btLasso.dates, cumsum(btLasso.netReturns), 'LineWidth', 1.2);
grid on;
xlabel('Date');
ylabel('Cumulative return');
title('Baseline vs DCT+Thresholding vs DCT+OMP vs DCT+LASSO');
legend('Baseline reversal', 'DCT + Thresholding', 'DCT + OMP', 'DCT + LASSO', 'Location', 'best');

% ---------------- Thresholding gross/net ----------------
figure;
subplot(2,1,1);
plot(btThresh.dates, btThresh.grossReturns, 'LineWidth', 1.0);
grid on;
ylabel('Gross daily return');
title('DCT + Thresholding gross and net returns');

subplot(2,1,2);
plot(btThresh.dates, btThresh.netReturns, 'LineWidth', 1.0);
grid on;
ylabel('Net daily return');
xlabel('Date');

% ---------------- OMP gross/net ----------------
figure;
subplot(2,1,1);
plot(btOMP.dates, btOMP.grossReturns, 'LineWidth', 1.0);
grid on;
ylabel('Gross daily return');
title('DCT + OMP gross and net returns');

subplot(2,1,2);
plot(btOMP.dates, btOMP.netReturns, 'LineWidth', 1.0);
grid on;
ylabel('Net daily return');
xlabel('Date');

% ---------------- LASSO gross/net ----------------
figure;
subplot(2,1,1);
plot(btLasso.dates, btLasso.grossReturns, 'LineWidth', 1.0);
grid on;
ylabel('Gross daily return');
title('DCT + LASSO gross and net returns');

subplot(2,1,2);
plot(btLasso.dates, btLasso.netReturns, 'LineWidth', 1.0);
grid on;
ylabel('Net daily return');
xlabel('Date');

% ---------------- Signal matrices ----------------
figure;
imagesc(threshSignal');
colorbar;
xlabel('Time index');
ylabel('Asset index');
title('DCT + Thresholding Signal Matrix');

figure;
imagesc(ompSignal');
colorbar;
xlabel('Time index');
ylabel('Asset index');
title('DCT + OMP Signal Matrix');

figure;
imagesc(lassoSignal');
colorbar;
xlabel('Time index');
ylabel('Asset index');
title('DCT + LASSO Signal Matrix');

% ---------------- Outlier experiment ----------------
if cfg.runOutlierExperiment
    fprintf('\nRunning outlier experiment...\n');

    retsCorrupt = inject_sparse_outliers(rets, cfg.outlierFrac, cfg.outlierScale);
    [residCorrupt, ~, ~] = remove_common_factors(retsCorrupt, cfg.numFactors);

    baselineSignalCorrupt = nan(size(residCorrupt));
    for t = cfg.lookback:size(residCorrupt,1)
        win = residCorrupt(t-cfg.lookback+1:t, :);
        mu = mean(win, 1);
        sig = std(win, 0, 1);
        sig(sig < 1e-8) = 1;
        z = (residCorrupt(t,:) - mu) ./ sig;
        baselineSignalCorrupt(t,:) = -z;
    end

    [threshSignalCorrupt, ~] = rolling_dct_alpha(residCorrupt, cfg.window, cfg.lambda, ...
        cfg.numDctKeep, cfg.standardizeWithinWindow);
    if cfg.reverseThreshSignal
        threshSignalCorrupt = -threshSignalCorrupt;
    end

    [ompSignalCorrupt, ~] = rolling_dct_alpha_omp(residCorrupt, cfg.window, cfg.ompK, ...
        cfg.standardizeWithinWindow);
    if cfg.reverseOmpSignal
        ompSignalCorrupt = -ompSignalCorrupt;
    end

    [lassoSignalCorrupt, ~] = rolling_dct_alpha_lasso(residCorrupt, cfg.window, cfg.lassoLambda, ...
        cfg.standardizeWithinWindow);
    if cfg.reverseLassoSignal
        lassoSignalCorrupt = -lassoSignalCorrupt;
    end

    baselineSignalCorrupt = normalize_cross_section(baselineSignalCorrupt);
    threshSignalCorrupt   = normalize_cross_section(threshSignalCorrupt);
    ompSignalCorrupt      = normalize_cross_section(ompSignalCorrupt);
    lassoSignalCorrupt    = normalize_cross_section(lassoSignalCorrupt);

    btBaseCorrupt = cross_sectional_long_short_rebalance(baselineSignalCorrupt, retsCorrupt, retDates, ...
        cfg.longQuantile, cfg.shortQuantile, cfg.transactionCostBps, ...
        cfg.signalLag, cfg.annualization, cfg.rebalanceEvery);

    btThreshCorrupt = cross_sectional_long_short_rebalance(threshSignalCorrupt, retsCorrupt, retDates, ...
        cfg.longQuantile, cfg.shortQuantile, cfg.transactionCostBps, ...
        cfg.signalLag, cfg.annualization, cfg.rebalanceEvery);

    btOMPCorrupt = cross_sectional_long_short_rebalance(ompSignalCorrupt, retsCorrupt, retDates, ...
        cfg.longQuantile, cfg.shortQuantile, cfg.transactionCostBps, ...
        cfg.signalLag, cfg.annualization, cfg.rebalanceEvery);

    btLassoCorrupt = cross_sectional_long_short_rebalance(lassoSignalCorrupt, retsCorrupt, retDates, ...
        cfg.longQuantile, cfg.shortQuantile, cfg.transactionCostBps, ...
        cfg.signalLag, cfg.annualization, cfg.rebalanceEvery);

    fprintf('\n=== Outlier experiment: Baseline stats ===\n');
    disp(performance_stats(btBaseCorrupt.netReturns, cfg.annualization));

    fprintf('\n=== Outlier experiment: DCT + Thresholding stats ===\n');
    disp(performance_stats(btThreshCorrupt.netReturns, cfg.annualization));

    fprintf('\n=== Outlier experiment: DCT + OMP stats ===\n');
    disp(performance_stats(btOMPCorrupt.netReturns, cfg.annualization));

    fprintf('\n=== Outlier experiment: DCT + LASSO stats ===\n');
    disp(performance_stats(btLassoCorrupt.netReturns, cfg.annualization));

    figure;
    plot(btBaseCorrupt.dates, cumsum(btBaseCorrupt.netReturns), 'LineWidth', 1.1); hold on;
    plot(btThreshCorrupt.dates, cumsum(btThreshCorrupt.netReturns), 'LineWidth', 1.1);
    plot(btOMPCorrupt.dates, cumsum(btOMPCorrupt.netReturns), 'LineWidth', 1.1);
    plot(btLassoCorrupt.dates, cumsum(btLassoCorrupt.netReturns), 'LineWidth', 1.1);
    grid on;
    xlabel('Date');
    ylabel('Cumulative return');
    title('Outlier Experiment: Baseline vs DCT+Thresholding vs DCT+OMP vs DCT+LASSO');
    legend('Baseline', 'DCT + Thresholding', 'DCT + OMP', 'DCT + LASSO', 'Location', 'best');
end

%% ---------------- Local helper functions ----------------
function S = normalize_cross_section(S)
    for t = 1:size(S,1)
        x = S(t,:);
        mu = mean(x, 'omitnan');
        sig = std(x, 0, 'omitnan');
        if ~isnan(sig) && sig > 1e-8
            S(t,:) = (x - mu) / sig;
        end
    end
end