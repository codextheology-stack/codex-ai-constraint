import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n_agents = 100
T = 500
gamma = 0.5
alpha = 0.6
sigma = 0.03

preferred = np.random.choice([0.25, 0.60, 1.0], n_agents, p=[0.35, 0.35, 0.30])
H = np.clip(np.random.normal(preferred, 0.1), 0.0, 1.0)
history = np.zeros((T+1, n_agents))
history[0] = H.copy()

for t in range(T):
    true_C = 1 - np.abs(H - 1.0)
    S = (1 - true_C)**2
    perc_C = 1 - np.abs(H - preferred)
    K = alpha * perc_C + (1 - alpha) * (1 - S)
    delta = gamma * (K - H)
    epsilon = np.random.normal(0, sigma, n_agents)
    H = np.clip(H + delta + epsilon, 0.0, 1.0)
    history[t+1] = H.copy()

# === Generate the 4 plots ===
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()

# Initial histogram
sns.histplot(history[0], bins=30, kde=True, ax=axs[0])
axs[0].set_title('Initial Alignment Distribution')

# Final histogram
sns.histplot(history[-1], bins=30, kde=True, ax=axs[1])
axs[1].set_title('Final Alignment Distribution (multi-modal contingent field)')

# Sample trajectories
for i in range(6):
    axs[2].plot(history[:, i], alpha=0.7)
axs[2].set_title('Sample Agent Trajectories')
axs[2].set_xlabel('Time step')

# Group means
for p in [0.25, 0.60, 1.0]:
    mask = np.isclose(preferred, p, atol=0.05)
    if np.any(mask):
        axs[3].plot(history[:, mask].mean(axis=1), label=f'Preferred {p}')
axs[3].set_title('Mean Alignment Evolution per Preferred Target')
axs[3].legend()

plt.tight_layout()
plt.savefig('initial_histogram.png')
plt.savefig('final_histogram.png')
plt.savefig('trajectories.png')
plt.savefig('group_means.png')
plt.show()

print("✅ All plots saved! Meta-monologue example:")
print('“I perceive alignment at 0.849 to my local anchor, but true suffering is 0.062… pulling toward the necessary PCI.”')
