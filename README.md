# Bayesian Regression Techniques for High-Dimensional Financial Time Series Data

## Overview

This repository contains the implementation and analysis of **Bayesian Regression Techniques for High-Dimensional Financial Time Series Data**, focusing on the S&P 500 index. The study explores **shrinkage priors**—specifically, **horseshoe** and **spike-and-slab**—to identify sparsity in portfolios and enhance model explainability while maintaining predictive power.

![Main project visualization](images/main_project_visual.png)

## Objectives

- Identify the most impactful stocks in the S&P 500 portfolio.
- Apply Bayesian regression techniques with shrinkage priors to uncover sparsity in high-dimensional data.
- Evaluate the performance of **horseshoe** and **spike-and-slab** priors on financial data.
- Suggest optimal portfolio weights based on Bayesian shrinkage models.

## Methodology

### High-Dimensionality in Financial Data
Financial datasets often exhibit high-dimensionality with \( p \gg n \), leading to issues such as ill-posedness, multicollinearity, and overfitting. Dimensionality reduction using **Bayesian shrinkage priors** helps address these challenges.

### Bayesian Regression Model

The Bayesian regression model is expressed as:
\[
y = X\beta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
\]

Shrinkage priors applied:
1. **Horseshoe Prior**:
    $\beta_j \mid \lambda_j, \tau  \sim \mathcal{N}(0, \lambda_j^2 \tau^2), \\
    \lambda_j \sim C^{+}(0, 1)$
   
Defined in Stan as
```R
parameters {
  real alpha;                      // Intercept
  vector[P] beta;                  // Coefficients for predictors
  real<lower=0> sigma;             // Standard deviation of residuals
  real<lower=0> tau;               // Global shrinkage parameter
  vector<lower=0>[P] lambda;       // Local shrinkage parameters
  real<lower=0> c;                 // Hyperparameter for the horseshoe
}

transformed parameters {
  vector[P] beta_shrunk;           // Shrinkage applied coefficients
  beta_shrunk = beta .* (tau * lambda);
}

model {
  // Priors
  alpha ~ normal(0, 5);
  lambda ~ cauchy(0, 1);
  tau ~ cauchy(0, 1);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);

  // Likelihood
  y ~ normal(X * beta_shrunk + alpha, sigma);
}
```


3. **Spike-and-Slab Prior**:
    $\beta_j \mid \lambda_j, c, \epsilon \sim \lambda_j \cdot \mathcal{N}\left(0, c^2\right) + (1 - \lambda_j) \cdot \mathcal{N}\left(0, \epsilon^2\right), \\
    \lambda_j \mid \pi \sim \text{Bernoulli}(\pi), \quad j = 1, \dots, p$

Stan code
```R
parameters {
  real alpha;                      // Intercept
  vector[P] beta;                  // Coefficients for predictors
  real<lower=0> sigma;             // Standard deviation of residuals
  real<lower=0> slab_scale;        // Slab standard deviation
  real<lower=0> spike_scale;       // Spike standard deviation
  vector<lower=0, upper=1>[P] theta;  // Mixing variable between spike and slab (Bernoulli)
}

transformed parameters {
  vector[P] beta_shrunk;           // Shrinkage applied coefficients
  beta_shrunk = beta .* (1 - theta) * spike_scale + beta .* theta * slab_scale;
}

model {
  // Priors
  alpha ~ normal(0, 5);
  spike_scale ~ normal(0, 0.1);    // Spike has small variance
  slab_scale ~ normal(0, 1);       // Slab has larger variance
  theta ~ beta(1, 1);              // Prior on mixing variable
  beta ~ normal(0, 1);             // Coefficients prior
  sigma ~ normal(0, 1);            // Residual standard deviation

  // Likelihood
  y ~ normal(X * beta_shrunk + alpha, sigma);
}

generated quantities {
  vector[N] y_pred;
  y_pred = X * beta_shrunk + alpha;  // Predictions for each data point
}
```


### Data

The dataset comprises daily observations from the S&P 500 between **2018–2022** for training and **2023** for validation. Features include adjusted closing prices, daily returns, exponential moving averages, and more.

#### Feature Engineering
- Normalize features:
  $\text{Normalized Feature} = \frac{x - \mu}{\sigma}$
  
- Calculate daily returns:
  $DR_{t} = \frac{AC_t - AC_{t-1}}{AC_{t-1}}$
  
  

![Dataset visualization](images/dataset.png)

### Model Training
The models were trained using the **RStan** package with **MCMC sampling**:
- **Horseshoe Prior**: Looser shrinkage to aid convergence.
- **Spike-and-Slab Prior**: Binary indicator to determine variable importance.
- Model training as:

```R
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
```

### Results
- **Predictive Performance**: Spike-and-slab achieved superior results with lower Mean Squared Error (MSE).
- **Goodness of Fit**:
  $\bar{R}^{2} = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}$

### Visualization
- Comparison of adjusted \( R^2 \) values for both priors:
  ![Adjusted R2 comparison](images/adjusted_r2.png)
- Spike-and-slab parameter estimates by credible intervals:
  ![Credible intervals](images/credible_intervals.png)

## Key Findings

- **Sparsity Uncovered**: Approximately **180 regressors** were found sufficient for maintaining an adjusted $\bar{R}^2 \geq 0.95$
- **Predictive Accuracy**: Spike-and-slab outperformed horseshoe in predictive tasks.

## Future Work

- Explore forecasting using time and opening prices only.
- Investigate stricter shrinkage priors for enhanced explainability.


## References

- [Bayesian Data Analysis](https://users.aalto.fi/~ave/BDA3.pdf)
- [RStan Documentation](https://mc-stan.org/rstan/)

## Acknowledgments

This project was developed as part of a Bachelor's Thesis at Aalto University under the guidance of **Ersin Yılmaz**.
