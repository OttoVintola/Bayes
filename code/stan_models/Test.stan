data {
  int<lower=0> N;           // Number of observations
  int<lower=0> P;           // Number of predictors
  matrix[N, P] X;           // Predictor matrix
  vector[N] y;              // Response variable
}

parameters {
  vector[P] beta;           // Regression coefficients
  real alpha;               // Intercept
  real<lower=0> sigma;      // Error standard deviation
}

model {
  // Priors
  beta ~ normal(0, 1);            // Prior for coefficients
  alpha ~ normal(0, 1);           // Prior for intercept
  sigma ~ cauchy(0, 2.5);         // Prior for error standard deviation
  
  // Likelihood
  y ~ normal(X * beta + alpha, sigma);
}

generated quantities {
  vector[N] y_pred;               // Predicted values for each observation
  
  for (n in 1:N) {
    y_pred[n] = normal_rng(X[n] * beta + alpha, sigma);
  }
}
