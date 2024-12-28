import numpy as np

# Classe pour la simulation des feedbacks utilisateurs

class UserFeedbackSimulator:
    def __init__(self, n_movies, seed=42):
        np.random.seed(seed)  # Fixer la graine pour reproductibilité
        self.true_rewards = np.random.beta(2, 5, size=n_movies)  # Probabilités réelles des récompenses

    def simulate_feedback(self, movie_index):
        return np.random.binomial(1, self.true_rewards[movie_index])