# Robust Alpha Signal Recovery Using Sparse Methods and Transformers

This project studies whether ideas from **compressed sensing**, **sparse recovery**, and **transformer-based sequence modeling** can improve financial alpha extraction from noisy stock return data.

Instead of trading raw returns directly, the pipeline first removes common market structure, then attempts to recover a cleaner **residual alpha signal** using:
- **DCT + Soft Thresholding**
- **DCT + Orthogonal Matching Pursuit (OMP)**
- **DCT + LASSO**
- **Transformer-based alpha model**

The main goal is to compare classical sparse recovery methods with a modern transformer model in a unified **cross-sectional long-short trading framework**.

---

## Motivation

Financial returns are noisy, weakly predictable, and strongly affected by broad market factors. A simple reversal strategy applied directly to raw returns or raw residuals often produces unstable performance and large drawdowns.

This project is motivated by a compressed sensing idea:

> Even if a signal looks noisy in the original domain, it may become sparse or compressible in the right transform domain.

Here, I apply that idea to financial residual return windows using the **Discrete Cosine Transform (DCT)**. The hypothesis is that the useful predictive component is more recoverable in the DCT domain than in the raw time domain.

I then compare this sparse-recovery approach against a transformer model that learns cross-sectional return structure directly.

---

## Data

The project uses **daily U.S. equity price files** and converts them into **daily log returns**.

### Final filtered stock universe
- AAPL
- ADBE
- AMAT
- AMD
- AMZN
- GOOGL
- INTC
- META
- MSFT
- MU

These are liquid, large-cap, technology-oriented U.S. equities.

---

## Method Overview

### 1. Preprocessing
- Load daily stock price data
- Convert prices to daily log returns
- Filter assets with too many missing values
- Winsorize returns to reduce extreme outliers

### 2. Factor Removal
Let \( R \in \mathbb{R}^{T \times N} \) be the return matrix.  
Remove common factors via:
\[
R = FB^\top + E
\]
where:
- \(F\): common factors
- \(B\): factor loadings
- \(E\): residual returns

The residual matrix \(E\) is the main signal source for alpha construction.

### 3. Baseline Signal
A rolling residual-reversal score is computed from the standardized residual return:
\[
s^{\mathrm{base}}_{t,i} = -\frac{e_{t,i}-\mu_{t,i}}{\sigma_{t,i}+\varepsilon}
\]

### 4. Sparse Recovery in the DCT Domain
For a rolling residual window \(x\), apply DCT:
\[
c = Dx
\]
Then recover a cleaner signal using:
- **Soft Thresholding**
- **OMP**
- **LASSO**

### 5. Transformer Signal
A transformer model is trained to predict **cross-sectional return structure** rather than a single raw next-day return.

### 6. Portfolio Construction
Signals are converted into demeaned, normalized long-short portfolio weights:
\[
w_{t,i} = \frac{\tilde s_{t,i}}{\sum_{j=1}^{N} |\tilde s_{t,j}|}
\]
and evaluated using next-period returns with transaction costs.

---

## Sparse Methods

### DCT + Thresholding
Small transform coefficients are shrunk toward zero:
\[
\hat c_j = \mathrm{sign}(c_j)\max(|c_j|-\lambda,0)
\]
This is the simplest sparsity-based denoising method.

### DCT + OMP
OMP greedily selects the most important transform atoms one at a time and reconstructs the signal using a sparse support set.

### DCT + LASSO
LASSO solves:
\[
\hat c = \arg\min_c \frac{1}{2}\|x-D^\top c\|_2^2 + \lambda\|c\|_1
\]
This is a convex sparsity-regularized recovery method.

---

## Transformer Model

The transformer is used as a nonlinear sequence model that predicts cross-sectional alpha structure from rolling return features.

Compared with sparse methods:
- sparse methods impose **explicit transform-domain structure**
- the transformer **learns nonlinear structure directly from data**

---

## Performance Metrics

The following metrics are used:
- Mean Daily Return
- Daily Return Standard Deviation
- Annualized Sharpe Ratio
- CAGR
- Maximum Drawdown

---

## Main Results

### Clean Data Results

| Method | MeanDailyRet | StdDailyRet | Sharpe | CAGR | MaxDrawdown |
|---|---:|---:|---:|---:|---:|
| Baseline residual reversal | -0.00086643 | 0.021783 | -0.63141 | -0.24322 | -0.99053 |
| DCT + Thresholding | -0.00030586 | 0.021969 | -0.22101 | -0.12907 | -0.91745 |
| DCT + OMP | -0.00020703 | 0.022229 | -0.14785 | -0.10841 | -0.85745 |
| DCT + LASSO | -0.00027837 | 0.021966 | -0.20117 | -0.12300 | -0.90762 |
| Transformer | **0.00010208** | 0.026423 | **0.06133** | -0.060618 | -0.93633 |

### Key Takeaways
- All sparse methods outperform the naive baseline
- **DCT + OMP** is the best sparse method in clean data
- The **transformer performs best overall**
- The transformer is the **only method with positive mean daily return and positive Sharpe ratio**

---

## Outlier Robustness Results

| Method | MeanDailyRet | StdDailyRet | Sharpe | CAGR | MaxDrawdown |
|---|---:|---:|---:|---:|---:|
| Baseline | -0.0016987 | 0.033480 | -0.80543 | -0.43651 | -0.99993 |
| DCT + Thresholding | -0.00089877 | 0.032251 | -0.44239 | -0.30355 | -0.99804 |
| DCT + OMP | -0.0015439 | 0.033091 | -0.74064 | -0.41299 | -0.99988 |
| DCT + LASSO | -0.00082233 | 0.031972 | -0.40830 | -0.28816 | -0.99806 |

### Outlier Robustness Takeaways
- All methods deteriorate under sparse outlier corruption
- Sparse methods still outperform the baseline
- **Thresholding and LASSO are more robust than OMP** under outlier noise

---

## Project Insights

This project shows that compressed sensing ideas can be meaningfully applied outside classical signal recovery.

Main insights:
1. Financial residual return series are not pure noise; they contain recoverable structure
2. Transform-domain sparsity improves alpha extraction
3. OMP works best among sparse methods in clean data
4. Thresholding and LASSO are more stable under outlier contamination
5. A transformer can outperform sparse methods when trained to predict cross-sectional structure directly

---

## Repository Structure

```text
.
├── data/                      # raw stock text files
├── figures/                   # generated plots
├── load_price_panel.m
├── compute_log_returns.m
├── remove_common_factors.m
├── rolling_dct_alpha.m
├── cross_sectional_long_short.m
├── performance_stats.m
├── inject_sparse_outliers.m
├── run_alpha_project.m
├── run_transformer_backtest.m
├── transformer_signal.csv
├── final_alpha_comparison.m
└── README.md
