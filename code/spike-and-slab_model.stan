data {
  int<lower=0> N;           // Number of observations
  int<lower=0> P;           // Number of predictors
  matrix[N, P] X;           // Predictor matrix
  vector[N] y;              // Response variable
}

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
