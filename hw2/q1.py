import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters for the prior Beta distribution
alpha_prior = 1/5
beta_prior = 1/5

# Observed data: 3 heads (successes), 0 tails (failures)
successes = 3
failures = 0

# Updated parameters for the posterior Beta distribution
alpha_posterior = alpha_prior + successes
beta_posterior = beta_prior + failures

# Create a range of theta values between 0 and 1
theta = np.linspace(0, 1, 1000)

# Compute the posterior density
posterior_density = beta.pdf(theta, alpha_posterior, beta_posterior)

# Compute the posterior mean
posterior_mean = beta.mean(alpha_posterior, beta_posterior)

# Plotting the posterior density
plt.figure(figsize=(10, 6))
plt.plot(theta, posterior_density, label=f'Posterior Beta({alpha_posterior:.2f}, {beta_posterior:.2f})', color='blue')
plt.axvline(posterior_mean, color='red', linestyle='--', label=f'Posterior Mean ≈ {posterior_mean:.3f}')
plt.title('Posterior Density of θ Given HHH with Beta(1/5, 1/5) Prior')
plt.xlabel('θ (Probability of Heads)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
