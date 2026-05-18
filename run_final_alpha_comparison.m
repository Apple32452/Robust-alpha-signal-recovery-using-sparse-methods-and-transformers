%% run_final_alpha_comparison.m
clear; clc; close all; rng(0);

% ===================== PATHS =====================
projectFolder = '/Users/taewoonchoi/Compressed Sensing and Sparse Recovery/Project-Transformers';
dataFolder    = projectFolder;
signalFile    = fullfile(projectFolder, 'transformer_signal_cross_sectional.csv');

% ===================== CONFIG =====================
startDate = datetime(2010,1,1);

wanted = ["aapl.us","adbe.us","amat.us","amd.us","amzn.us", ...
          "googl.us","intc.us","meta.us","msft.us","mu.us"];

maxMissingFrac = 0.05;
winsorZ = 5.0;
numFactors = 3;

% Baseline
lookback = 20;

% DCT sparse settings
window = 60;
lambdaThresh = 0.015;
ompK = 6;
lassoLambda = 0.01;
standardizeWithinWindow = true;

% Backtest settings
longQuantile = 0.10;
shortQuantile = 0.10;
transactionCostBps = 5;
annualization = 252;
signalLag = 0;
rebalanceEvery = 5;

% ===================== LOAD PRICE DATA =====================
[dates, prices, symbols] = load_price_panel(dataFolder);

keepWanted = ismember(lower(symbols), wanted);
prices = prices(:, keepWanted);
symbols = symbols(keepWanted);

keepDates = dates >= startDate;
dates = dates(keepDates);
prices = prices(keepDates, :);

rets = compute_log_returns(prices);
retDates = dates(2:end);

missingFrac = mean(isnan(rets), 1);
keepAssets = missingFrac <= maxMissingFrac;
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
        x = max(min(x, winsorZ * sigma), -winsorZ * sigma);
    end
    rets(:,j) = x;
end

fprintf('Using %d assets in final comparison.\n', numel(symbols));
disp(symbols)

% ===================== BUILD RESIDUALS =====================
[resid, ~, ~] = remove_common_factors(rets, numFactors);

% ===================== 1) BASELINE SIGNAL =====================
baselineSignal = nan(size(resid));
for t = lookback:size(resid,1)
    win = resid(t-lookback+1:t, :);
    mu = mean(win, 1);
    sig = std(win, 0, 1);
    sig(sig < 1e-8) = 1;
    z = (resid(t,:) - mu) ./ sig;
    baselineSignal(t,:) = -z;
end

% ===================== 2) DCT + THRESHOLDING =====================
[threshSignal, ~] = rolling_dct_alpha(resid, window, lambdaThresh, [], standardizeWithinWindow);
threshSignal = -threshSignal;

% ===================== 3) DCT + OMP =====================
[ompSignal, ~] = rolling_dct_alpha_omp(resid, window, ompK, standardizeWithinWindow);
ompSignal = -ompSignal;

% ===================== 4) DCT + LASSO =====================
[lassoSignal, ~] = rolling_dct_alpha_lasso(resid, window, lassoLambda, standardizeWithinWindow);
lassoSignal = -lassoSignal;

% ===================== 5) TRANSFORMER SIGNAL =====================
Tsig = readtable(signalFile, 'VariableNamingRule', 'preserve');

if isdatetime(Tsig{1,1})
    signalDates = Tsig{:,1};
else
    signalDates = datetime(Tsig{:,1});
end

signalSymbols = string(Tsig.Properties.VariableNames(2:end));
signalMat = Tsig{:,2:end};

% symbol alignment
symbolsNorm = lower(strrep(symbols, '_', '.'));
signalSymbolsNorm = lower(strrep(signalSymbols, '_', '.'));

[tfSym, locSym] = ismember(symbolsNorm, signalSymbolsNorm);
if ~all(tfSym)
    disp("Backtest symbols:")
    disp(symbols)
    disp("Transformer CSV symbols:")
    disp(signalSymbols)
    error('Some backtest symbols are missing from transformer_signal.csv');
end
signalMat = signalMat(:, locSym);

% date alignment
[tfDate, locDate] = ismember(retDates, signalDates);
transformerSignal = nan(numel(retDates), numel(symbols));
transformerSignal(tfDate, :) = signalMat(locDate(tfDate), :);

% ===================== NORMALIZE CROSS-SECTIONALLY =====================
baselineSignal    = normalize_cross_section(baselineSignal);
threshSignal      = normalize_cross_section(threshSignal);
ompSignal         = normalize_cross_section(ompSignal);
lassoSignal       = normalize_cross_section(lassoSignal);
transformerSignal = normalize_cross_section(transformerSignal);

% ===================== BACKTEST ALL =====================
btBase = cross_sectional_long_short_rebalance( ...
    baselineSignal, rets, retDates, longQuantile, shortQuantile, ...
    transactionCostBps, signalLag, annualization, rebalanceEvery);

btThresh = cross_sectional_long_short_rebalance( ...
    threshSignal, rets, retDates, longQuantile, shortQuantile, ...
    transactionCostBps, signalLag, annualization, rebalanceEvery);

btOMP = cross_sectional_long_short_rebalance( ...
    ompSignal, rets, retDates, longQuantile, shortQuantile, ...
    transactionCostBps, signalLag, annualization, rebalanceEvery);

btLasso = cross_sectional_long_short_rebalance( ...
    lassoSignal, rets, retDates, longQuantile, shortQuantile, ...
    transactionCostBps, signalLag, annualization, rebalanceEvery);

btTransformer = cross_sectional_long_short_rebalance( ...
    transformerSignal, rets, retDates, longQuantile, shortQuantile, ...
    transactionCostBps, signalLag, annualization, rebalanceEvery);

% ===================== STATS =====================
statsBase = performance_stats(btBase.netReturns, annualization);
statsThresh = performance_stats(btThresh.netReturns, annualization);
statsOMP = performance_stats(btOMP.netReturns, annualization);
statsLasso = performance_stats(btLasso.netReturns, annualization);
statsTransformer = performance_stats(btTransformer.netReturns, annualization);

fprintf('\n=== Final Combined Comparison Stats ===\n');

disp('Baseline residual reversal:')
disp(statsBase)

disp('DCT + Thresholding:')
disp(statsThresh)

disp('DCT + OMP:')
disp(statsOMP)

disp('DCT + LASSO:')
disp(statsLasso)

disp('Transformer:')
disp(statsTransformer)

% ===================== COMBINED CUMULATIVE RETURN PLOT =====================
figure;
plot(btBase.dates, cumsum(btBase.netReturns), 'LineWidth', 1.3); hold on;
plot(btThresh.dates, cumsum(btThresh.netReturns), 'LineWidth', 1.3);
plot(btOMP.dates, cumsum(btOMP.netReturns), 'LineWidth', 1.3);
plot(btLasso.dates, cumsum(btLasso.netReturns), 'LineWidth', 1.3);
plot(btTransformer.dates, cumsum(btTransformer.netReturns), 'LineWidth', 1.6);
grid on;
xlabel('Date');
ylabel('Cumulative return');
title('Final Alpha Comparison: Baseline vs DCT Sparse Methods vs Transformer');
legend('Baseline reversal', 'DCT + Thresholding', 'DCT + OMP', 'DCT + LASSO', 'Transformer', ...
    'Location', 'best');

% ===================== SHARPE BAR CHART =====================
methodNames = categorical({'Baseline','DCT+Thresh','DCT+OMP','DCT+LASSO','Transformer'});
methodNames = reordercats(methodNames, {'Baseline','DCT+Thresh','DCT+OMP','DCT+LASSO','Transformer'});

sharpes = [
    statsBase.Sharpe
    statsThresh.Sharpe
    statsOMP.Sharpe
    statsLasso.Sharpe
    statsTransformer.Sharpe
];

cagrs = [
    statsBase.CAGR
    statsThresh.CAGR
    statsOMP.CAGR
    statsLasso.CAGR
    statsTransformer.CAGR
];

figure;
bar(methodNames, sharpes);
grid on;
ylabel('Sharpe');
title('Sharpe Comparison Across Alpha Methods');

figure;
bar(methodNames, cagrs);
grid on;
ylabel('CAGR');
title('CAGR Comparison Across Alpha Methods');

% ===================== MAX DRAWDOWN BAR CHART =====================
maxDDs = [
    statsBase.MaxDrawdown
    statsThresh.MaxDrawdown
    statsOMP.MaxDrawdown
    statsLasso.MaxDrawdown
    statsTransformer.MaxDrawdown
];

figure;
bar(methodNames, maxDDs);
grid on;
ylabel('Max Drawdown');
title('Max Drawdown Comparison Across Alpha Methods');

% ===================== OPTIONAL: TRANSFORMER VS BEST SPARSE =====================
figure;
plot(btOMP.dates, cumsum(btOMP.netReturns), 'LineWidth', 1.4); hold on;
plot(btTransformer.dates, cumsum(btTransformer.netReturns), 'LineWidth', 1.6);
grid on;
xlabel('Date');
ylabel('Cumulative return');
title('Best Sparse Method vs Transformer');
legend('DCT + OMP', 'Transformer', 'Location', 'best');

% ===================== LOCAL HELPER =====================
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