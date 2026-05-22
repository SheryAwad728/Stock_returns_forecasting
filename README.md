# Stock Returns as Leading Indicators of Economic Growth

## Overview
This project tests whether S&P 500 returns hold explanatory and out-of-sample 
forecasting power over industrial production growth using monthly data from FRED 
(1990-2025).

## Methods
- In-sample AR(2) benchmark vs AR(2) + S&P 500 model comparison
- Out-of-sample forecasting using expanding and rolling window estimation
- Clark-West test for forecast accuracy
- VAR model and Granger causality tests
- GDP growth robustness check

## Data Sources
All data pulled from the Federal Reserve Economic Data (FRED) database:
- Industrial Production Index (INDPRO)
- Real GDP (GDPC1)
- S&P 500 Total Return Index (SPASTT01USM661N)

## Requirements
R packages: fredr, tidyverse, forecast, vars, ggplot2, stargazer

## Key Findings
Lagged S&P 500 returns significantly predict Industrial Production growth in-sample,
nearly tripling the benchmark R² from 6.6% to 18.7%, consistent with Fama (1990).
Out-of-sample, the expanding window shows a statistically significant 11.3% reduction
in MSFE (Clark-West p = 0.026). Rolling window results show large but time-varying
gains concentrated around post-recession periods, consistent with a behavioral shift
from speculation toward fundamentals during economic downturns. GDP robustness
checks are consistent in sign but do not reach statistical significance, suggesting S&P
500 returns are a stronger indicator of cyclical IP growth than of broader output.

## Author
Shery Awad & Juan Perez| MA Economics Candidates, California State University Long Beach
