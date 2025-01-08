data = read.csv("../data/data.csv")

head(data, n=10)


acf(data$Portfolio.Daily.Return, ci=0.999999999999, main=expression(ACF(DR[t])))


library(tseries)
Box.test(data$Portfolio.Daily.Return, lag = 10, type = "Ljung-Box")

library(lmtest)
dwtest(lm(data$Portfolio.Daily.Return ~ 1))

# Load necessary libraries
library(dplyr)
library(rstan)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Prepare the data
response <- data[, ncol(data)]  # Extract the response variable (last column)
predictors <- data[2:4025]  # Extract the predictors (excluding date and response)

cat("Number of predictors:", ncol(predictors), "\n")
cat("Number of observations:", nrow(predictors))

# Prepare the data list for Stan
stan_data <- list(
  N = nrow(predictors),       # Number of observations
  P = ncol(predictors),       # Number of predictors
  X = predictors,             # Predictor matrix
  y = response                # Response variable
)

saveRDS(fit, file = "hs_fit_results.rds")

sprintf("First %s and last columns %s:", colnames(predictors[0:1]), colnames(predictors[ncol(predictors)]))

# Compile the Stan model
stan_model_code <- readLines("horseshoe_model.stan")
stan_model <- stan_model(model_code = stan_model_code)

# Fit the model
fit <- sampling(
  stan_model,
  data = stan_data, 
  iter = 2000,              # Number of iterations
  chains = 4,               # Number of chains
  warmup = 500,             # Number of warmup (burn-in) samples
  thin = 1,                 # Thinning interval
  seed = 123,               # Random seed for reproducibility
  control = list(adapt_delta = 0.95, max_treedepth=15)

)

# Compile the Stan model
spike_and_slab_code <- readLines("spike-and-slab_model.stan")
spike_and_slab <- stan_model(model_code = spike_and_slab_code)

# Fit the model
ss_fit <- sampling(
  spike_and_slab,
  data = stan_data,
  iter = 2000,             # Number of iterations
  chains = 4,              # Number of chains
  warmup = 500,            # Number of warmup (burn-in) samples
  thin = 1,                # Thinning interval
  seed = 123               # Random seed for reproducibility
)

saveRDS(ss_fit, file = "ss_fit_results.rds")

library(cmdstanr)
set_cmdstan_path("/Users/otto/.cmdstan/cmdstan-2.35.0")

file <- ("/Users/otto/Thesis/code/horseshoe_model.stan")
horseshoe <- cmdstan_model(file, cpp_options = list(stan_opencl = TRUE))

# Fit the model
cmdstanr_hs_fit <- horseshoe$sample(
  data = stan_data,
  seed = 123,
  chains = 1,
  parallel_chains = 1,
  refresh = 10,
  opencl_ids = c(0, 0)
)



cmdstanr_hs_fit$output(4)
