import numpy as np
import matplotlib.pyplot as plt
from user_feedback_simulator import UserFeedbackSimulator
from thompson_sampling import ThompsonSampling


# Classe principale pour orchestrer l'expérience
class MovieRecommendationExperiment:
    def __init__(self, movie_list, n_simulations=10): # Ici nous avons le nombre d'itération au bout desquelles l'expérience va s'arrêter
        self.movie_list = movie_list
        self.n_simulations = n_simulations
        self.n_movies = len(movie_list)
        self.simulator = UserFeedbackSimulator(self.n_movies)
        self.ts = ThompsonSampling(self.n_movies)

    def run_experiment(self, dynamic_mode=False):
        rewards = np.zeros(self.n_simulations)
        chosen_arms = np.zeros(self.n_simulations)
        regrets = np.zeros(self.n_simulations)
        optimal_reward = max(self.simulator.true_rewards)

        for t in range(self.n_simulations):
            chosen_arm = self.ts.select_arm()
            if dynamic_mode:
                print(f"Film proposé : {self.movie_list.iloc[chosen_arm]['title']}")
                feedback = self.simulator.dynamic_feedback()
            else:
                feedback = self.simulator.simulate_feedback(chosen_arm)
            self.ts.update(chosen_arm, feedback)
            rewards[t] = feedback
            chosen_arms[t] = chosen_arm

            # Calculer le regret cumulatif
            regrets[t] = optimal_reward - self.simulator.true_rewards[chosen_arm]

        return rewards, chosen_arms, regrets

    def plot_results(self, rewards, regrets):
        cumulative_rewards = np.cumsum(rewards)
        cumulative_regrets = np.cumsum(regrets)

        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_rewards, label='Récompenses cumulées')
        plt.plot(cumulative_regrets, label='Regret cumulatif')
        plt.xlabel('Essais')
        plt.ylabel('Valeurs cumulatives')
        plt.title('Progression des récompenses et regrets cumulés')
        plt.legend()
        plt.grid(True)
        plt.show()

    def display_most_popular_movie(self, chosen_arms):
        most_chosen_arm = np.bincount(chosen_arms.astype(int)).argmax()
        most_chosen_movie = self.movie_list.iloc[most_chosen_arm]['title']
        print(f'Film le plus populaire : {most_chosen_movie}')
