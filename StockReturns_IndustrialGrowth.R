################################################################################
# Stock Returns as Leading Indicators of Economic Growth
# Author: Shery Awad
# Date: 2026
#
# This script tests whether S&P 500 returns hold explanatory and out-of-sample
# forecasting power over industrial production growth using data from FRED.
#
# Analysis includes:
#   - In-sample AR(2) benchmark vs AR(2) + S&P 500 model comparison
#   - Out-of-sample forecasting using expanding and rolling window estimation
#   - Clark-West test for forecast accuracy
#   - VAR model and Granger causality tests
#   - GDP growth robustness check
#
# Data Sources: Federal Reserve Economic Data (FRED)
#   - Industrial Production Index (INDPRO)
#   - Real GDP (GDPC1)
#   - S&P 500 Total Return Index (SPASTT01USM661N)
#
# Required packages: fredr, tidyverse, forecast, vars, ggplot2, stargazer
################################################################################

#install.packages(c("fredr", "tidyverse", "forecast", "vars", "ggplot2"))

# Load libraries
library(fredr)
library(tidyverse)
library(forecast)
library(ggplot2)
library(dplyr)


detach("package:MASS", unload = TRUE)


# Set your FRED API key
fredr_set_key("YOUR_API_KEY_HERE")

# Pull data from FRED
indpro <- fredr(series_id = "INDPRO",
                observation_start = as.Date("1990-01-01"),
                observation_end = as.Date("2025-01-01"))

gdp <- fredr(series_id = "GDPC1",
             observation_start = as.Date("1990-01-01"),
             observation_end = as.Date("2025-01-01"))

sp500 <- fredr(series_id = "SP500",
               observation_start = as.Date("1990-01-01"),
               observation_end = as.Date("2025-01-01"))

# Check the data pulled correctly
head(indpro)
head(gdp)
head(sp500)

# Keep only date and value, rename value column
indpro_clean <- indpro %>% select(date, value) %>% rename(indpro = value)
gdp_clean <- gdp %>% select(date, value) %>% rename(gdp = value)

# SP500 is daily - convert to monthly (last trading day of each month)
sp500_monthly <- sp500 %>%
  select(date, value) %>%
  rename(sp500 = value) %>%
  mutate(year = year(date), month = month(date)) %>%
  group_by(year, month) %>%
  slice_tail(n = 1) %>%  # take last observation of each month
  ungroup() %>%
  select(date, sp500)

# Calculate log growth rates
indpro_clean <- indpro_clean %>%
  mutate(ip_growth = log(indpro) - lag(log(indpro))) %>%
  drop_na()

gdp_clean <- gdp_clean %>%
  mutate(gdp_growth = log(gdp) - lag(log(gdp))) %>%
  drop_na()

sp500_monthly <- sp500_monthly %>%
  mutate(sp500_return = log(sp500) - lag(log(sp500))) %>%
  drop_na()

# Check
head(indpro_clean)
head(gdp_clean)
head(sp500_monthly)

sp500 <- fredr(series_id = "SPASTT01USM661N",
               observation_start = as.Date("1990-01-01"),
               observation_end = as.Date("2025-01-01"))

head(sp500)
sp500_monthly <- data.frame(date = sp500$date, sp500 = sp500$value) %>%
  mutate(sp500_return = log(sp500) - lag(log(sp500))) %>%
  drop_na()

head(sp500_monthly)


# Merge IP and SP500 by date
monthly_data <- merge(indpro_clean, sp500_monthly, by = "date")

# Check
head(monthly_data)
nrow(monthly_data)

#The data used in this study are drawn from the Federal Reserve Economic Data 
#(FRED) database. The sample covers January 1990 through January 2025. Industrial
#production growth serves as the primary measure of real economic activity and 
#is constructed as the log difference of the Industrial Production Index (INDPRO), 
#which is available at a monthly frequency. Real GDP growth serves as a secondary 
#measure and is used as a robustness check; it is constructed as the log difference 
#of real GDP (GDPC1), which is reported quarterly. Stock market returns are measured 
#using the S&P 500 index (SPASTT01USM661N) and are also constructed as monthly 
#log differences to match the frequency of industrial production. After merging
#the industrial production and stock return series, the final monthly dataset 
#contains 420 observations. The quarterly dataset, used for the GDP robustness 
#check, is kept separate due to the frequency mismatch. Lag length for the 
#autoregressive models is selected using AIC and BIC criteria, with a maximum 
#of 12 lags considered given the monthly frequency of the data.

###############################################################################

# AIC/BIC 

# Use auto.arima to select lag length via AIC/BIC
library(forecast)

ar_aic <- auto.arima(monthly_data$ip_growth, 
                     max.p = 12,  # max 12 lags for monthly data
                     d = 0, 
                     max.q = 0,   # pure AR, no MA terms
                     ic = "aic",
                     seasonal = FALSE)

ar_bic <- auto.arima(monthly_data$ip_growth, 
                     max.p = 12, 
                     d = 0, 
                     max.q = 0, 
                     ic = "bic",
                     seasonal = FALSE)

summary(ar_aic) #ARIMA(2,0,0) with non-zero mean 
summary(ar_bic) #ARIMA(2,0,0) with zero mean 


### Both models agree on the AR(2) model. 
## Choosing the BIC model of excluding the mean 

### Have to explain why!!!

################################################################################


# AR(2) benchmark model
ar_model <- lm(ip_growth ~ lag(ip_growth, 1) + lag(ip_growth, 2), 
               data = monthly_data)

# AR(2) + stock returns model
ar_sp500_model <- lm(ip_growth ~ lag(ip_growth, 1) + lag(ip_growth, 2) + 
                       lag(sp500_return, 1), 
                     data = monthly_data)

summary(ar_model)
summary(ar_sp500_model)

# R-squared jumps from 0.066 to 0.187 — adding stock returns nearly 
#triples the explanatory power

#Note we use lagged SP500 returns — lag(sp500_return, 1) — because we're asking 
#whether last month's stock returns help predict this month's IP growth. 

#strong in-sample evidence that stock returns contain useful information about 
#future industrial production growth, consistent with Fama (1990).

monthly_data$ip_growth_lag1 <- c(NA, head(monthly_data$ip_growth, -1))
monthly_data$ip_growth_lag2 <- c(NA, NA, head(monthly_data$ip_growth, -2))
monthly_data$sp500_return_lag1 <- c(NA, head(monthly_data$sp500_return, -1))

ar_model2 <- lm(ip_growth ~ ip_growth_lag1 + ip_growth_lag2, 
                data = monthly_data)

ar_sp500_model2 <- lm(ip_growth ~ ip_growth_lag1 + ip_growth_lag2 + 
                        sp500_return_lag1, 
                      data = monthly_data)

insample_msfe_ar <- mean(residuals(ar_model2)^2, na.rm = TRUE)
insample_msfe_ar_sp500 <- mean(residuals(ar_sp500_model2)^2, na.rm = TRUE)
insample_msfe_ratio <- insample_msfe_ar_sp500 / insample_msfe_ar

cat("In-sample MSFE AR benchmark:", insample_msfe_ar, "\n")
cat("In-sample MSFE AR + SP500:", insample_msfe_ar_sp500, "\n")
cat("In-sample MSFE Ratio:", insample_msfe_ratio, "\n")

insample_table <- data.frame(
  Model = c("AR(2) Benchmark", "AR(2) + S\\&P 500"),
  MSFE = c(insample_msfe_ar, insample_msfe_ar_sp500),
  MSFE_Ratio = c(NA, insample_msfe_ratio)
)
library(stargazer)

stargazer(insample_table,
          title = "In-Sample Fit Evaluation",
          summary = FALSE,
          rownames = FALSE,
          digits = 7,
          type = "latex",
          out = "table_insample_msfe.tex")

###############################################################################

## Out of sample estimations (60 months - 5 years)

# Start fresh - remerge and build clean dataset
monthly_data <- merge(indpro_clean, sp500_monthly, by = "date")

# Create lagged variables explicitly
monthly_data$ip_growth_lag1 <- c(NA, head(monthly_data$ip_growth, -1))
monthly_data$ip_growth_lag2 <- c(NA, NA, head(monthly_data$ip_growth, -2))
monthly_data$sp500_return_lag1 <- c(NA, head(monthly_data$sp500_return, -1))

# Drop NAs
monthly_data <- na.omit(monthly_data)

# Check
nrow(monthly_data)
head(monthly_data)



# Reset
n <- nrow(monthly_data)
initial_window <- 60
forecast_ar <- rep(NA, n)
forecast_ar_sp500 <- rep(NA, n)

# Expanding window loop
for(i in initial_window:(n-1)){
  train <- monthly_data[1:i, ]
  next_obs <- monthly_data[i+1, ]
  
  # AR(2) benchmark
  ar_fit <- lm(ip_growth ~ ip_growth_lag1 + ip_growth_lag2, data = train)
  forecast_ar[i+1] <- predict(ar_fit, newdata = next_obs)
  
  # AR(2) + SP500
  ar_sp500_fit <- lm(ip_growth ~ ip_growth_lag1 + ip_growth_lag2 + 
                       sp500_return_lag1, data = train)
  forecast_ar_sp500[i+1] <- predict(ar_sp500_fit, newdata = next_obs)
}

# Add forecasts to data
monthly_data$forecast_ar <- forecast_ar
monthly_data$forecast_ar_sp500 <- forecast_ar_sp500

# Check
head(monthly_data[60:65, ])




################################################################################

### MSFE and clark west test 


# Calculate forecast errors
monthly_data$error_ar <- monthly_data$ip_growth - monthly_data$forecast_ar
monthly_data$error_ar_sp500 <- monthly_data$ip_growth - monthly_data$forecast_ar_sp500

# Drop NAs for evaluation
eval_data <- na.omit(monthly_data[, c("date", "ip_growth", "forecast_ar", 
                                      "forecast_ar_sp500", "error_ar", 
                                      "error_ar_sp500")])

# MSFE for each model
msfe_ar <- mean(eval_data$error_ar^2)
msfe_ar_sp500 <- mean(eval_data$error_ar_sp500^2)

# MSFE ratio (below 1 means AR+SP500 beats benchmark)
msfe_ratio <- msfe_ar_sp500 / msfe_ar

cat("MSFE AR benchmark:", msfe_ar, "\n")
cat("MSFE AR + SP500:", msfe_ar_sp500, "\n")
cat("MSFE Ratio:", msfe_ratio, "\n")



# Clark-West test
# Step 1: Calculate the adjusted forecast error
eval_data$cw_adj <- eval_data$error_ar^2 - 
  eval_data$error_ar_sp500^2 + 
  (eval_data$forecast_ar - eval_data$forecast_ar_sp500)^2

# Step 2: Regress on a constant and test if mean is positive
cw_test <- lm(cw_adj ~ 1, data = eval_data)
summary(cw_test)

# Step 3: One-sided t-statistic
cw_tstat <- coef(summary(cw_test))["(Intercept)", "t value"]
cw_pvalue <- pt(cw_tstat, df = nrow(eval_data) - 1, lower.tail = FALSE)

cat("Clark-West t-statistic:", cw_tstat, "\n")
cat("Clark-West p-value (one-sided):", cw_pvalue, "\n")

#The in-sample results provide strong evidence that stock returns contain useful
# information about future industrial production growth. The AR(2) benchmark model, 
#which includes only two lags of industrial production growth, yields an R-squared 
#of 0.066, suggesting that past industrial production growth alone explains little 
#of its future variation. Adding one lag of S&P 500 returns to the benchmark model 
#increases the R-squared to 0.187, nearly tripling the explanatory power of the model. 
#The coefficient on lagged stock returns is positive and highly statistically
#significant, indicating that higher stock returns today are associated with 
#stronger industrial production growth next month. Turning to the out-of-sample 
#results, the expanding window forecasts show that the AR(2) + stock returns model 
#produces a mean squared forecast error that is 11.3 percent lower than the AR(2) 
#benchmark, yielding an MSFE ratio of 0.887. The Clark-West test, which accounts 
#for the additional parameter uncertainty in the larger model, yields a 
#t-statistic of 1.953 and a one-sided p-value of 0.026, indicating that this 
#improvement in forecast accuracy is statistically significant at the 5 percent 
#level. Together, these results suggest that stock returns contain meaningful 
#predictive information for future industrial production growth beyond what is 
#captured by its own lags, consistent with the findings of Fama (1990) but now 
#confirmed in an out-of-sample forecasting framework using more recent data.



################################################################################

# Rolling window loop
forecast_ar_roll <- rep(NA, n)
forecast_ar_sp500_roll <- rep(NA, n)

for(i in initial_window:(n-1)){
  # Fixed 60-month rolling window
  train <- monthly_data[max(1, i - initial_window + 1):i, ]
  next_obs <- monthly_data[i+1, ]
  
  # AR(2) benchmark
  ar_fit <- lm(ip_growth ~ ip_growth_lag1 + ip_growth_lag2, data = train)
  forecast_ar_roll[i+1] <- predict(ar_fit, newdata = next_obs)
  
  # AR(2) + SP500
  ar_sp500_fit <- lm(ip_growth ~ ip_growth_lag1 + ip_growth_lag2 + 
                       sp500_return_lag1, data = train)
  forecast_ar_sp500_roll[i+1] <- predict(ar_sp500_fit, newdata = next_obs)
}

# Add to data
monthly_data$forecast_ar_roll <- forecast_ar_roll
monthly_data$forecast_ar_sp500_roll <- forecast_ar_sp500_roll

# Check
head(monthly_data[60:65, c("date", "forecast_ar_roll", "forecast_ar_sp500_roll")])

################################################################################

# Rolling window forecast errors
monthly_data$error_ar_roll <- monthly_data$ip_growth - monthly_data$forecast_ar_roll
monthly_data$error_ar_sp500_roll <- monthly_data$ip_growth - monthly_data$forecast_ar_sp500_roll

# Drop NAs
eval_roll <- na.omit(monthly_data[, c("date", "ip_growth", "forecast_ar_roll",
                                      "forecast_ar_sp500_roll", "error_ar_roll",
                                      "error_ar_sp500_roll")])

# MSFE
msfe_ar_roll <- mean(eval_roll$error_ar_roll^2)
msfe_ar_sp500_roll <- mean(eval_roll$error_ar_sp500_roll^2)
msfe_ratio_roll <- msfe_ar_sp500_roll / msfe_ar_roll

cat("Rolling MSFE AR benchmark:", msfe_ar_roll, "\n")
cat("Rolling MSFE AR + SP500:", msfe_ar_sp500_roll, "\n")
cat("Rolling MSFE Ratio:", msfe_ratio_roll, "\n")

# Clark-West test
eval_roll$cw_adj_roll <- eval_roll$error_ar_roll^2 - 
  eval_roll$error_ar_sp500_roll^2 + 
  (eval_roll$forecast_ar_roll - eval_roll$forecast_ar_sp500_roll)^2

cw_test_roll <- lm(cw_adj_roll ~ 1, data = eval_roll)
cw_tstat_roll <- coef(summary(cw_test_roll))["(Intercept)", "t value"]
cw_pvalue_roll <- pt(cw_tstat_roll, df = nrow(eval_roll) - 1, lower.tail = FALSE)

cat("Rolling Clark-West t-statistic:", cw_tstat_roll, "\n")
cat("Rolling Clark-West p-value (one-sided):", cw_pvalue_roll, "\n")

#The rolling window results provide additional nuance to the out-of-sample 
#forecasting evidence. Using a fixed 60-month estimation window, the AR(2) + 
#stock returns model produces an MSFE that is 41.6 percent lower than the AR(2) 
#benchmark, yielding an MSFE ratio of 0.584. However, the Clark-West test yields 
#a t-statistic of 1.278 and a one-sided p-value of 0.101, indicating that this 
#improvement is not statistically significant at conventional levels. Taken 
#together, the expanding and rolling window results paint a consistent but 
#nuanced picture. The expanding window results suggest that stock returns 
#contain meaningful predictive information for industrial production growth on 
#average over the full sample, while the rolling window results suggest that 
#this relationship is not stable across all subperiods. This time variation in 
#predictive power is consistent with the findings of Binswanger (2000), who 
#documents a weakening of the stock return-real activity relationship in more 
#recent data, and with Cochrane (2008), who argues that the predictive content 
#of financial variables tends to be unstable across samples. Together, these 
#results suggest that while stock returns improve forecasts of industrial 
#production growth on average, their predictive power varies over time and may 
#be concentrated in certain periods rather than constant throughout the sample.

#######################

# Calculate rolling MSFE ratio over time
window_size <- 60
rolling_msfe_ratio <- rep(NA, n)

for(i in (initial_window + window_size):(n)){
  window_data <- eval_roll[max(1, i - window_size + 1):i, ]
  msfe_ar_w <- mean(window_data$error_ar_roll^2, na.rm = TRUE)
  msfe_sp500_w <- mean(window_data$error_ar_sp500_roll^2, na.rm = TRUE)
  rolling_msfe_ratio[i] <- msfe_sp500_w / msfe_ar_w
}

monthly_data$rolling_msfe_ratio <- rolling_msfe_ratio

# Plot
plot_data <- na.omit(monthly_data[, c("date", "rolling_msfe_ratio")])

ggplot(plot_data, aes(x = date, y = rolling_msfe_ratio)) +
  geom_line(color = "steelblue", linewidth = 0.8) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "Rolling MSFE Ratio: AR + SP500 vs AR Benchmark",
       subtitle = "Values below 1 indicate AR + SP500 outperforms benchmark",
       x = "Date",
       y = "MSFE Ratio") +
  theme_minimal()

###############################################################################


library(vars)

# Set up VAR data
var_data <- monthly_data[, c("ip_growth", "sp500_return")]

# First select lag length
var_lag <- VARselect(var_data, lag.max = 12, type = "const")
var_lag$selection

# Estimate VAR with BIC selected lag
var_model <- VAR(var_data, p = var_lag$selection["SC(n)"], type = "const")

# Granger causality tests
# Does SP500 Granger-cause IP growth?
causality(var_model, cause = "sp500_return")

# Does IP growth Granger-cause SP500?
causality(var_model, cause = "ip_growth")


# Estimate VAR(1) based on BIC selection
var_model <- VAR(var_data, p = 1, type = "const")
summary(var_model)

# Impulse Response Functions
irf_result <- irf(var_model, 
                  impulse = "sp500_return", 
                  response = "ip_growth",
                  n.ahead = 12,  # 12 months ahead
                  boot = TRUE,   # bootstrap confidence intervals
                  ci = 0.95)    # 95% confidence interval

# Plot IRF
plot(irf_result)


# Extract IRF values
irf_plot <- irf(var_model, 
                impulse = "sp500_return", 
                response = "ip_growth",
                n.ahead = 12,
                boot = TRUE,
                ci = 0.95)

# Extract components
irf_mean <- irf_plot$irf$sp500_return
irf_lower <- irf_plot$Lower$sp500_return
irf_upper <- irf_plot$Upper$sp500_return
periods <- 0:12

# Plot manually with legend
plot(periods, irf_mean, type = "l", lwd = 2, col = "black",
     ylim = range(c(irf_lower, irf_upper)),
     xlab = "Months After Shock",
     ylab = "Response of IP Growth",
     main = "Impulse Response: SP500 Return → IP Growth")

# Add confidence bands
lines(periods, irf_upper, col = "red", lty = 2, lwd = 1.5)
lines(periods, irf_lower, col = "red", lty = 2, lwd = 1.5)

# Add zero line
abline(h = 0, col = "red", lwd = 1)

# Add legend
legend("topright", 
       legend = c("IRF Estimate", "95% Confidence Interval", "Zero Line"),
       col = c("black", "red", "red"),
       lty = c(1, 2, 1),
       lwd = c(2, 1.5, 1))



#As a robustness check, we estimate a bivariate vector autoregression (VAR) 
#model including industrial production growth and S&P 500 returns. Lag length 
#is selected using BIC, which selects one lag. The Granger causality tests 
#confirm the one-directional nature of the relationship — S&P 500 returns 
#strongly Granger-cause industrial production growth (F-statistic = 57.013, 
#p-value < 0.001), while industrial production growth does not Granger-cause 
#S&P 500 returns (F-statistic = 0.893, p-value = 0.345). This confirms that 
#endogeneity is not a major concern in the single equation models and that the 
#predictive relationship runs from financial markets to real economic activity, 
#not the other way around. The impulse response function, shown in Figure 2, 
#illustrates that a positive shock to S&P 500 returns leads to a statistically 
#significant increase in industrial production growth that peaks one month after 
#the shock and dissipates within approximately six months. Together, the VAR 
#results are consistent with the single equation findings and support the 
#conclusion that stock returns contain short-lived but meaningful predictive 
#information about future real economic activity.


################################################################################


# Aggregate SP500 returns to quarterly by compounding monthly returns
sp500_quarterly <- sp500_monthly %>%
  mutate(year = format(date, "%Y"),
         quarter = quarters(date)) %>%
  group_by(year, quarter) %>%
  summarise(sp500_return_q = sum(sp500_return, na.rm = TRUE),
            date = min(date)) %>%
  ungroup() %>%
  dplyr::select(date, sp500_return_q)

# Merge with GDP
quarterly_data <- merge(gdp_clean, sp500_quarterly, by = "date")

# Create lagged variables
quarterly_data$gdp_growth_lag1 <- c(NA, head(quarterly_data$gdp_growth, -1))
quarterly_data$gdp_growth_lag2 <- c(NA, NA, head(quarterly_data$gdp_growth, -2))
quarterly_data$sp500_return_lag1 <- c(NA, head(quarterly_data$sp500_return_q, -1))

# Drop NAs
quarterly_data <- na.omit(quarterly_data)

# Check
nrow(quarterly_data)
head(quarterly_data)


# Lag selection for GDP
ar_aic_gdp <- auto.arima(quarterly_data$gdp_growth,
                         max.p = 8,  # max 8 lags for quarterly data
                         d = 0,
                         max.q = 0,
                         ic = "aic",
                         seasonal = FALSE)

ar_bic_gdp <- auto.arima(quarterly_data$gdp_growth,
                         max.p = 8,
                         d = 0,
                         max.q = 0,
                         ic = "bic",
                         seasonal = FALSE)

cat("AIC selects:", arimaorder(ar_aic_gdp)[1], "lags\n")
cat("BIC selects:", arimaorder(ar_bic_gdp)[1], "lags\n")

# AR(1) benchmark model for GDP
ar_gdp <- lm(gdp_growth ~ gdp_growth_lag1, data = quarterly_data)

# AR(1) + SP500 model
ar_sp500_gdp <- lm(gdp_growth ~ gdp_growth_lag1 + sp500_return_lag1, 
                   data = quarterly_data)

summary(ar_gdp)
summary(ar_sp500_gdp)

# GDP out of sample forecasts
n_q <- nrow(quarterly_data)
initial_window_q <- 20
forecast_ar_gdp <- rep(NA, n_q)
forecast_ar_sp500_gdp <- rep(NA, n_q)

# Expanding window
for(i in initial_window_q:(n_q-1)){
  train <- quarterly_data[1:i, ]
  next_obs <- quarterly_data[i+1, ]
  
  ar_fit <- lm(gdp_growth ~ gdp_growth_lag1, data = train)
  forecast_ar_gdp[i+1] <- predict(ar_fit, newdata = next_obs)
  
  ar_sp500_fit <- lm(gdp_growth ~ gdp_growth_lag1 + sp500_return_lag1, data = train)
  forecast_ar_sp500_gdp[i+1] <- predict(ar_sp500_fit, newdata = next_obs)
}

quarterly_data$forecast_ar_gdp <- forecast_ar_gdp
quarterly_data$forecast_ar_sp500_gdp <- forecast_ar_sp500_gdp

# MSFE
quarterly_data$error_ar_gdp <- quarterly_data$gdp_growth - quarterly_data$forecast_ar_gdp
quarterly_data$error_ar_sp500_gdp <- quarterly_data$gdp_growth - quarterly_data$forecast_ar_sp500_gdp

eval_gdp <- na.omit(quarterly_data[, c("date", "gdp_growth", "forecast_ar_gdp",
                                       "forecast_ar_sp500_gdp", "error_ar_gdp",
                                       "error_ar_sp500_gdp")])

msfe_ar_gdp <- mean(eval_gdp$error_ar_gdp^2)
msfe_ar_sp500_gdp <- mean(eval_gdp$error_ar_sp500_gdp^2)
msfe_ratio_gdp <- msfe_ar_sp500_gdp / msfe_ar_gdp

cat("GDP MSFE AR benchmark:", msfe_ar_gdp, "\n")
cat("GDP MSFE AR + SP500:", msfe_ar_sp500_gdp, "\n")
cat("GDP MSFE Ratio:", msfe_ratio_gdp, "\n")

# Clark-West test
eval_gdp$cw_adj_gdp <- eval_gdp$error_ar_gdp^2 - 
  eval_gdp$error_ar_sp500_gdp^2 +
  (eval_gdp$forecast_ar_gdp - eval_gdp$forecast_ar_sp500_gdp)^2

cw_test_gdp <- lm(cw_adj_gdp ~ 1, data = eval_gdp)
cw_tstat_gdp <- coef(summary(cw_test_gdp))["(Intercept)", "t value"]
cw_pvalue_gdp <- pt(cw_tstat_gdp, df = nrow(eval_gdp) - 1, lower.tail = FALSE)

cat("GDP Clark-West t-statistic:", cw_tstat_gdp, "\n")
cat("GDP Clark-West p-value (one-sided):", cw_pvalue_gdp, "\n")



library(stargazer)


stargazer(ar_model, ar_sp500_model,
          title = "In-Sample Regression Results: Industrial Production Growth",
          dep.var.labels = "IP Growth",
          covariate.labels = c("IP Growth (t-1)", "IP Growth (t-2)", 
                               "SP500 Return (t-1)"),
          column.labels = c("AR(2) Benchmark", "AR(2) + SP500"),
          omit.stat = c("ser", "f"),
          digits = 4,
          type = "text")

stargazer(ar_model, ar_sp500_model,
          title = "In-Sample Regression Results: Industrial Production Growth",
          dep.var.labels = "IP Growth",
          covariate.labels = c("IP Growth (t-1)", "IP Growth (t-2)", 
                               "SP500 Return (t-1)"),
          column.labels = c("AR(2) Benchmark", "AR(2) + SP500"),
          omit.stat = c("ser", "f"),
          digits = 4,
          type = "latex",
          out = "table1_insample.tex")

# Create out-of-sample results table
oos_table <- data.frame(
  Method = c("IP Expanding Window", "IP Rolling Window", "GDP Expanding Window"),
  MSFE_Benchmark = c(msfe_ar, msfe_ar_roll, msfe_ar_gdp),
  MSFE_SP500 = c(msfe_ar_sp500, msfe_ar_sp500_roll, msfe_ar_sp500_gdp),
  MSFE_Ratio = c(msfe_ratio, msfe_ratio_roll, msfe_ratio_gdp),
  CW_Stat = c(cw_tstat, cw_tstat_roll, cw_tstat_gdp),
  CW_Pvalue = c(cw_pvalue, cw_pvalue_roll, cw_pvalue_gdp)
)

stargazer(oos_table,
          title = "Out-of-Sample Forecast Evaluation",
          summary = FALSE,
          rownames = FALSE,
          digits = 6,
          type = "latex",
          out = "table2_oos.tex")



# Extract VAR equations as individual lm-like objects for stargazer
var_eq1 <- var_model$varresult$ip_growth
var_eq2 <- var_model$varresult$sp500_return

stargazer(var_eq1, var_eq2,
          title = "VAR(1) Estimation Results",
          dep.var.labels = c("IP Growth", "SP500 Return"),
          covariate.labels = c("IP Growth (t-1)", "SP500 Return (t-1)", "Constant"),
          column.labels = c("Equation 1", "Equation 2"),
          omit.stat = c("ser", "f"),
          digits = 4,
          type = "text",
          out = "table3_var.tex")


granger_table <- data.frame(
  Direction = c("SP500 Return -> IP Growth", "IP Growth -> SP500 Return"),
  F_Statistic = c(57.013, 0.893),
  P_Value = c(0.001, 0.345),
  Conclusion = c("Reject H0", "Fail to Reject H0")
)

stargazer(granger_table,
          title = "Granger Causality Tests",
          summary = FALSE,
          rownames = FALSE,
          digits = 3,
          type = "text",
          out = "table4_granger.tex")



pdf("figure3_timeseries.pdf", width = 10, height = 6)

par(mfrow = c(2,1), mar = c(3, 4, 2, 1))

# IP Growth
plot(monthly_data$date, monthly_data$ip_growth, 
     type = "l", col = "steelblue", lwd = 1,
     main = "Industrial Production Growth",
     xlab = "", ylab = "IP Growth",
     xaxt = "n")
axis.Date(1, monthly_data$date, format = "%Y")
abline(h = 0, col = "red", lty = 2)

# SP500 Returns
plot(monthly_data$date, monthly_data$sp500_return,
     type = "l", col = "darkgreen", lwd = 1,
     main = "S&P 500 Returns",
     xlab = "Date", ylab = "SP500 Return",
     xaxt = "n")
axis.Date(1, monthly_data$date, format = "%Y")
abline(h = 0, col = "red", lty = 2)

dev.off()


stargazer(ar_gdp, ar_sp500_gdp,
          title = "In-Sample Regression Results: GDP Growth (Robustness Check)",
          dep.var.labels = "GDP Growth",
          covariate.labels = c("GDP Growth (t-1)", "SP500 Return (t-1)"),
          column.labels = c("AR(1) Benchmark", "AR(1) + SP500"),
          omit.stat = c("ser", "f"),
          digits = 4,
          type = "text",
          out = "table5_gdp.tex")


# Monthly summary statistics
monthly_stats <- data.frame(
  ip_growth = monthly_data$ip_growth,
  sp500_return = monthly_data$sp500_return
)

# Quarterly summary statistics
quarterly_stats <- data.frame(
  gdp_growth = quarterly_data$gdp_growth,
  sp500_return_q = quarterly_data$sp500_return_q
)

stargazer(monthly_stats, quarterly_stats,
          title = "Summary Statistics",
          covariate.labels = c("IP Growth", "SP500 Return (Monthly)",
                               "GDP Growth", "SP500 Return (Quarterly)"),
          digits = 4,
          type = "text",
          out = "table0_summary.tex")



summary(ar_model)$adj.r.squared
summary(ar_sp500_model)$adj.r.squared

################################################################################

# GDP Rolling window
forecast_ar_gdp_roll <- rep(NA, n_q)
forecast_ar_sp500_gdp_roll <- rep(NA, n_q)

for(i in initial_window_q:(n_q-1)){
  train <- quarterly_data[max(1, i - initial_window_q + 1):i, ]
  next_obs <- quarterly_data[i+1, ]
  
  ar_fit <- lm(gdp_growth ~ gdp_growth_lag1, data = train)
  forecast_ar_gdp_roll[i+1] <- predict(ar_fit, newdata = next_obs)
  
  ar_sp500_fit <- lm(gdp_growth ~ gdp_growth_lag1 + sp500_return_lag1, data = train)
  forecast_ar_sp500_gdp_roll[i+1] <- predict(ar_sp500_fit, newdata = next_obs)
}

quarterly_data$error_ar_gdp_roll <- quarterly_data$gdp_growth - forecast_ar_gdp_roll
quarterly_data$error_ar_sp500_gdp_roll <- quarterly_data$gdp_growth - forecast_ar_sp500_gdp_roll

eval_gdp_roll <- na.omit(quarterly_data[, c("date", "gdp_growth", 
                                            "error_ar_gdp_roll", "error_ar_sp500_gdp_roll")])

msfe_ar_gdp_roll <- mean(eval_gdp_roll$error_ar_gdp_roll^2)
msfe_ar_sp500_gdp_roll <- mean(eval_gdp_roll$error_ar_sp500_gdp_roll^2)
msfe_ratio_gdp_roll <- msfe_ar_sp500_gdp_roll / msfe_ar_gdp_roll

eval_gdp_roll$cw_adj <- eval_gdp_roll$error_ar_gdp_roll^2 - 
  eval_gdp_roll$error_ar_sp500_gdp_roll^2 +
  (forecast_ar_gdp_roll[!is.na(forecast_ar_gdp_roll)] - 
     forecast_ar_sp500_gdp_roll[!is.na(forecast_ar_sp500_gdp_roll)])^2

cw_test_gdp_roll <- lm(cw_adj ~ 1, data = eval_gdp_roll)
cw_tstat_gdp_roll <- coef(summary(cw_test_gdp_roll))["(Intercept)", "t value"]
cw_pvalue_gdp_roll <- pt(cw_tstat_gdp_roll, df = nrow(eval_gdp_roll) - 1, lower.tail = FALSE)

cat("GDP Rolling MSFE AR benchmark:", msfe_ar_gdp_roll, "\n")
cat("GDP Rolling MSFE AR + SP500:", msfe_ar_sp500_gdp_roll, "\n")
cat("GDP Rolling MSFE Ratio:", msfe_ratio_gdp_roll, "\n")
cat("GDP Rolling CW t-statistic:", cw_tstat_gdp_roll, "\n")
cat("GDP Rolling CW p-value:", cw_pvalue_gdp_roll, "\n")


oos_table_updated <- data.frame(
  Method = c("IP Expanding Window", "IP Rolling Window", 
             "GDP Expanding Window", "GDP Rolling Window"),
  MSFE_Benchmark = c(msfe_ar, msfe_ar_roll, msfe_ar_gdp, msfe_ar_gdp_roll),
  MSFE_SP500 = c(msfe_ar_sp500, msfe_ar_sp500_roll, msfe_ar_sp500_gdp, msfe_ar_sp500_gdp_roll),
  MSFE_Ratio = c(msfe_ratio, msfe_ratio_roll, msfe_ratio_gdp, msfe_ratio_gdp_roll),
  CW_Stat = c(cw_tstat, cw_tstat_roll, cw_tstat_gdp, cw_tstat_gdp_roll),
  CW_Pvalue = c(cw_pvalue, cw_pvalue_roll, cw_pvalue_gdp, cw_pvalue_gdp_roll)
)

stargazer(oos_table_updated,
          title = "Out-of-Sample Forecast Evaluation",
          summary = FALSE,
          rownames = FALSE,
          digits = 6,
          type = "latex",
          out = "table2_oos_updated.tex")








