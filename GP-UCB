# GP-UCB modeling using reward function base algorithm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ---------------------------------------------
# 1. SETUP: SIMULATION DOMAIN & TRUE FUNCTION
# ---------------------------------------------

n_arms = 25  # Number of discrete options (think: positions of the "arm" in a bandit)
X_arms = np.linspace(0, 10, n_arms).reshape(-1, 1)  # These are the testable input positions

# Normalize input to [0, 1] for better GP numerical stability
X_arms_norm = (X_arms - X_arms.min()) / (X_arms.max() - X_arms.min())

def true_reward(x):
    """Define the true underlying reward function to optimize (can be anything!)"""
    # Here: A 'bell' curve + some periodic bump
    return 2.5 * np.exp(-((x - 0.5)**2) / 0.18) + 0.2 * np.sin(6*x)

true_means = true_reward(X_arms_norm).flatten()  # True expected reward at each input

# ---------------------------------------------
# 2. GP-UCB ALGORITHM PARAMETERS
# ---------------------------------------------

noise = 0.2           # Standard deviation of measurement noise (simulates real-world uncertainty)
beta = 2.0            # UCB "exploration" parameter; higher = more exploration
n_rounds = 100         # Number of UCB rounds (iterations)
np.random.seed(42)    # For reproducibility

X_hist = []           # Store all input locations sampled so far
y_hist = []           # Store corresponding observed noisy rewards
regret = []           # Store instantaneous regret at each round
eigval_records = []   # Store kernel matrix eigenvalues at each round

# ---------------------------------------------
# 3. GP-UCB ACTIVE LEARNING LOOP
# ---------------------------------------------

for t in range(n_rounds):
    if len(X_hist) > 0:
        # 3a. Define the kernel (Matern kernel is common for physical/smooth data)
        kernel = Matern(length_scale=0.3, length_scale_bounds=(0.1, 2.0), nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=noise**2, normalize_y=True, n_restarts_optimizer=3
        )
        gp.fit(np.array(X_hist), np.array(y_hist))  # Fit GP to observed data
        mu, sigma = gp.predict(X_arms_norm, return_std=True)  # Predict mean and uncertainty
        
        ucb = mu + np.sqrt(beta) * sigma  # Upper Confidence Bound (UCB acquisition function)

        # 3b. Kernel matrix diagnostics: calculate eigenvalues
        K = kernel(np.array(X_hist))
        eigvals = np.linalg.eigvalsh(K)
        eigval_records.append(eigvals)
    else:
        # If no data, force exploration
        ucb = np.ones(n_arms) * np.inf
        mu = np.zeros(n_arms)
        sigma = np.ones(n_arms)
        eigval_records.append(np.array([1.0]))  # Trivial kernel at first sample

    # 3c. Choose next point: arm with max UCB (most promising for max reward)
    if np.any(np.isinf(ucb)):
        chosen_arm = np.random.choice(n_arms)  # First round: random sample
    else:
        chosen_arm = np.argmax(ucb)            # Later: use UCB to balance exploration/exploitation

    X_hist.append(X_arms_norm[chosen_arm])     # Save chosen input
    reward = true_means[chosen_arm] + np.random.normal(0, noise)
    y_hist.append(reward)                      # Save observed reward (with noise)

    # 3d. Regret: how much worse was our chosen arm vs. the *best* possible?
    best_mean = np.max(true_means)
    regret.append(best_mean - true_means[chosen_arm])

    # 3e. Plot at final round
    if t == n_rounds - 1:
        plt.figure(figsize=(12,5))
        # --------- Plot 1: GP fit vs. ground truth ---------
        plt.subplot(121)
        plt.title("Final GP Estimate of Reward Function")
        plt.plot(X_arms_norm.flatten(), true_means, 'k--', label="True Reward")
        plt.plot(X_arms_norm.flatten(), mu, 'b-', label="GP Mean")
        plt.fill_between(X_arms_norm.flatten(), mu-2*sigma, mu+2*sigma, color='b', alpha=0.2, label="95% Conf")
        plt.scatter(np.array(X_hist).flatten(), np.array(y_hist), c='k', s=40, alpha=0.7, label="Samples")
        plt.xlabel("Arm (Normalized Position)")
        plt.ylabel("Reward")
        plt.legend()
        # This plot matches your screenshot (left): shows how GP fits the reward (with uncertainty)
        # "Samples" = where UCB chose to explore

        # --------- Plot 2: Cumulative regret over time ---------
        plt.subplot(122)
        plt.title("Cumulative Regret Over Time (GP-UCB)")
        plt.plot(np.cumsum(regret), 'r-', lw=2)
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative Regret")
        # This matches your screenshot (top right): if curve plateaus, UCB is learning the optimal!

        plt.tight_layout()
        plt.show()

# ---------------------------------------------
# 4. (Optional) Plot Minimum Kernel Matrix Eigenvalue Over Time
# ---------------------------------------------
min_eigs = [np.min(e) for e in eigval_records if len(e) > 0]
plt.figure(figsize=(6,4))
plt.plot(min_eigs, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Minimum Eigenvalue')
plt.title('Minimum Kernel Matrix Eigenvalue over Time')
plt.grid(True)
plt.show()
# This plot (bottom) tells you about GP numerical stability and how "redundant" your samples are.

