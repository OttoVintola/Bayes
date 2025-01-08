library(ggplot2)
library(MASS)  # For the Half-Cauchy distribution

# Set the parameter estimate beta with a narrower range
beta <- seq(-5, 5, length.out = 1000)

# Compute the densities for each distribution centered around beta
gaussian_density <- dnorm(beta, mean = 0, sd = 1)         # Gaussian distribution
student_t_density <- dt(beta, df = 3)                     # Student-t distribution with 3 degrees of freedom

# Define the Half-Cauchy density function manually
half_cauchy_density <- function(x, tau) {
  ifelse(x < 0, 0, (2 / (pi * tau)) * (1 / (1 + (x / tau)^2)))
}

# Compute the Horseshoe densities for different values of tau
horseshoe_density_tau1 <- half_cauchy_density(abs(beta), tau = 1)
horseshoe_density_tau08 <- half_cauchy_density(abs(beta), tau = 0.8)
horseshoe_density_tau05 <- half_cauchy_density(abs(beta), tau = 0.5)

# Combine the densities into a data frame for plotting
data <- data.frame(
  beta = rep(beta, 5),
  density = c(gaussian_density, student_t_density, horseshoe_density_tau1, horseshoe_density_tau08, horseshoe_density_tau05),
  distribution = rep(c("Gaussian", "Student-t (df=3)", "Horseshoe (τ=1)", "Horseshoe (τ=0.8)", "Horseshoe (τ=0.5)"), each = length(beta))
)

# Plot the distributions using ggplot2
ggplot(data, aes(x = beta, y = density, color = distribution)) +
  geom_line(size = 1) +
  labs(title = "Comparison of Distributions with Different τ Values",
       x = expression(beta),  # Use beta as x-axis label
       y = expression(p(beta))) +
  theme_minimal() +
  theme(legend.title = element_blank()) +
  scale_color_manual(values = c("lightblue", "lightpink", "coral", "lightgreen", "deeppink3"))






n <- 100  # number of observations
sigma <- 1  # variance of model
tau <- 0.3  # global shrinkage 
s_j <- 1  # variance of the pred
lambda_j <- seq(0, 1, length.out = 100)  # local shrinkage 

kappa_j <- 1 / (1 + n * (sigma^-2) * (tau^2) * (s_j^2) * (lambda_j^2))

a_j <- tau * sigma^-1 * sqrt(n) * s_j 
density_kappa <- (1 / pi) * (a_j / ((a_j^2 - 1) * kappa_j + 1)) * (1 / sqrt(kappa_j * (1 - kappa_j)))

ggplot(data = data.frame(kappa_j, density_kappa), aes(x = kappa_j, y = density_kappa)) +
  geom_line(color = "blue", size = 1) +
  geom_area(alpha = 0.2, fill = "blue") +
  labs(
    title = "Density of Shrinkage Factor kappa_j for Horseshoe Prior",
    x = expression(kappa[j] ~ (Shrinkage ~ Factor)),
    y = "Density"
  ) +
  theme_minimal()




# Define the parameters for the spike and slab components
spike_mean <- 0
spike_sd <- 0.1   # Narrow spike
slab_mean <- 0
slab_sd <- 1      # Wider slab
pi <- 0.3         # Probability of slab (variable inclusion)

# Define a sequence of beta values for plotting
beta_values <- seq(-2, 2, length.out = 1000)

spike_density <- dnorm(beta_values, mean = spike_mean, sd = spike_sd)
slab_density <- dnorm(beta_values, mean = slab_mean, sd = slab_sd)
mixture_density <- (1 - pi) * spike_density + pi * slab_density
student_t_density <- dt(beta_values, df = 3)
normal_density <- dnorm(beta_values, mean = 0, sd = 0.5)

# Create a data frame for plotting
data <- data.frame(
  beta = beta_values,
  Spike = spike_density,
  Slab = slab_density,
  Mixture = mixture_density,
  t_distribution = student_t_density,
  Normal = normal_density
)

# Reshape data for easier plotting using ggplot2
library(reshape2)
data_melted <- melt(data, id.vars = 'beta', variable.name = 'Distribution', value.name = 'Density')

#"lightblue", "lightpink", "coral", "lightgreen", "deeppink3"

# Plot the spike, slab, mixture, Student's t, and normal densities with a legend on the right
ggplot(data_melted, aes(x = beta, y = Density, color = Distribution)) +
  geom_line(size = 1) +
  labs(title = "Spike-and-Slab Prior with Student's t and Normal Distributions",
       x = expression(beta),
       y = expression(p(beta))) +
  scale_color_manual(values = c("Spike" = 'red', "Slab" = 'blue', "Mixture" = 'black', 
                                "t_distribution" = 'deeppink3', "Normal" = 'lightblue')) +
  theme_minimal() +
  theme(legend.position = "right",  # Position the legend on the right side
        legend.title = element_blank(),  # Remove legend title
        legend.text = element_text(size = 10))  # Adjust text size for readability


# Set the value of pi for both density functions
pi1 <- 0.7  # First density function
pi2 <- 0.2  # Second density function

# Define kappa values from 0 to 1 with step size of 0.25
kappa_values <- seq(0, 1, by = 0.25)

# Calculate probabilities for both pi values
probabilities <- data.frame(
  kappa = rep(kappa_values, 2),  # Repeat kappa values for both densities
  probability = c(
    c(1 - pi1, 0, 0, 0, pi1),  # Density for pi = 0.7
    c(1 - pi2, 0, 0, 0, pi2)   # Density for pi = 0.2
  ),
  pi_value = rep(c("π = 0.7", "π = 0.2"), each = length(kappa_values))  # Indicate pi value for each density
)

# Create a bar plot with ggplot
ggplot(probabilities, aes(x = factor(kappa), y = probability, fill = pi_value)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.2) +  # Adjust width for visibility
  scale_fill_manual(values = c("lightblue", "lightgreen")) +  # Different colors for different pi values
  labs(
    title = expression("Probability Density of Shrinkage Factor " * kappa[j]),
    x = expression(kappa[j]),
    y = expression(p(kappa[j]))
  ) +
  ylim(0, 1) +
  theme_minimal() +
  theme(legend.title = element_blank())  # Remove legend title


# Creating the Sequence
gfg = seq(0, 1, by = 0.1)
 
# Plotting the beta density
plot(gfg, dbeta(gfg, 1,1), xlab="X",
     ylab = "Beta Density", type = "l",
     col = "Red")
