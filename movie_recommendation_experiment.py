import numpy as np
import matplotlib.pyplot as plt
from UserFeedbackSimulator import UserFeedbackSimulator
from ThompsonSampling import ThompsonSampling


# Classe principale pour orchestrer l'expérience
class MovieRecommendationExperiment:
    def __init__(self, movie_list, n_simulations=1000):
        self.movie_list = movie_list
        self.n_simulations = n_simulations
        self.n_movies = len(movie_list)
        self.simulator = UserFeedbackSimulator(self.n_movies)
        self.ts = ThompsonSampling(self.n_movies)

    def run_experiment(self):
        rewards = np.zeros(self.n_simulations)
        chosen_arms = np.zeros(self.n_simulations)

        for t in range(self.n_simulations):
            chosen_arm = self.ts.select_arm()
            feedback = self.simulator.simulate_feedback(chosen_arm)
            self.ts.update(chosen_arm, feedback)
            rewards[t] = feedback
            chosen_arms[t] = chosen_arm

        return rewards, chosen_arms

    def plot_results(self, rewards):
        cumulative_rewards = np.cumsum(rewards)
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_rewards, label='Récompenses cumulées')
        plt.xlabel('Essais')
        plt.ylabel('Récompenses cumulées')
        plt.title('Progression des récompenses cumulées')
        plt.legend()
        plt.grid(True)
        plt.show()

    def display_most_popular_movie(self, chosen_arms):
        most_chosen_arm = np.bincount(chosen_arms.astype(int)).argmax()
        most_chosen_movie = self.movie_list.iloc[most_chosen_arm]['title']
        print(f'Film le plus populaire : {most_chosen_movie}')
