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