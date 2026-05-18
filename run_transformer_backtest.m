%% run_transformer_backtest.m
clear; clc; close all;

% ---------------- Paths ----------------
projectFolder = '/Users/taewoonchoi/Compressed Sensing and Sparse Recovery/Project-Transformers';
signalFile = fullfile(projectFolder, 'transformer_signal_cross_sectional.csv');

% Original price data folder
dataFolder = projectFolder;

% ---------------- Config ----------------
startDate = datetime(2010,1,1);
wanted = ["aapl.us","adbe.us","amat.us","amd.us","amzn.us", ...
          "googl.us","intc.us","meta.us","msft.us","mu.us"];

maxMissingFrac = 0.05;
winsorZ = 5.0;

longQuantile = 0.10;
shortQuantile = 0.10;
transactionCostBps = 5;
annualization = 252;
signalLag = 0;
rebalanceEvery = 5;

% ---------------- Load prices ----------------
[dates, prices, symbols] = load_price_panel(dataFolder);

keepWanted = ismember(lower(symbols), wanted);
dates = dates(:);
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

fprintf('Using %d assets for transformer backtest.\n', numel(symbols));
disp(symbols)

% ---------------- Load transformer signal ----------------
Tsig = readtable(signalFile, 'VariableNamingRule', 'preserve');

% Convert date column
if isdatetime(Tsig{1,1})
    signalDates = Tsig{:,1};
else
    signalDates = datetime(Tsig{:,1});
end

signalSymbols = string(Tsig.Properties.VariableNames(2:end));
signalMat = Tsig{:,2:end};

% ---------------- Align symbols ----------------
% Normalize symbol strings for safe matching
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

% ---------------- Align dates ----------------
[tfDate, locDate] = ismember(retDates, signalDates);
if ~all(tfDate)
    warning('Some retDates not found in transformer signal; unmatched rows will remain NaN');
end

alignedSignal = nan(numel(retDates), numel(symbols));
alignedSignal(tfDate, :) = signalMat(locDate(tfDate), :);

% ---------------- Cross-sectional z-score ----------------
alignedSignal = normalize_cross_section(alignedSignal);

% ---------------- Backtest ----------------
btTransformer = cross_sectional_long_short_rebalance( ...
    alignedSignal, rets, retDates, ...
    longQuantile, shortQuantile, ...
    transactionCostBps, signalLag, annualization, rebalanceEvery);

statsTransformer = performance_stats(btTransformer.netReturns, annualization);

fprintf('\n=== Transformer signal stats ===\n');
disp(statsTransformer);
% Simple regime filter from average market return
marketRet = mean(rets, 2, 'omitnan');
ma100 = movmean(marketRet, [99 0], 'omitnan');
tradeMask = ma100 > 0;

% ---------------- Plots ----------------
figure;
plot(btTransformer.dates, cumsum(btTransformer.netReturns), 'LineWidth', 1.2);
grid on;
xlabel('Date');
ylabel('Cumulative return');
title('Transformer Signal Cumulative Return');

figure;
subplot(2,1,1);
plot(btTransformer.dates, btTransformer.grossReturns, 'LineWidth', 1.0);
grid on;
ylabel('Gross daily return');
title('Transformer gross and net returns');

subplot(2,1,2);
plot(btTransformer.dates, btTransformer.netReturns, 'LineWidth', 1.0);
grid on;
ylabel('Net daily return');
xlabel('Date');

figure;
imagesc(alignedSignal');
colorbar;
xlabel('Time index');
ylabel('Asset index');
title('Transformer Signal Matrix');

%% -------- Local helper --------
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