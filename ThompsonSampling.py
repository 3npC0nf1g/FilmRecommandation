import numpy as np

# Implémentation de l'algorithme Thompson Sampling
class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.zeros(n_arms)  # Succès par bras
        self.failures = np.zeros(n_arms)  # Échecs par bras

    def select_arm(self):
        samples = [np.random.beta(self.successes[i] + 1, self.failures[i] + 1) for i in range(self.n_arms)]
        return np.argmax(samples)

    def update(self, chosen_arm, reward):
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1