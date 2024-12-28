import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from user_feedback_simulator import UserFeedbackSimulator
from thompson_sampling import ThompsonSampling


class MovieRecommendationExperiment:
    """
    Classe principale pour l'expérimentation de recommandation de films.

    Cette classe gère l'expérience de recommandation en utilisant l'algorithme
    Thompson Sampling pour sélectionner les films et collecter les retours
    utilisateurs.

    Attributes:
        movie_list (pd.DataFrame): Liste des films disponibles
        n_simulations (int): Nombre d'itérations de l'expérience
        n_movies (int): Nombre total de films
        ratings_history (list): Historique des notes données
    """

    STATS_DISPLAY_INTERVAL = 5  # Afficher les stats tous les 5 tours
    MAX_RATING = 5  # Note maximale possible
    MIN_RATING = 1  # Note minimale possible

    def __init__(self, movie_list, n_simulations=1000):
        """
        Initialise l'expérience de recommandation.

        Args:
            movie_list (pd.DataFrame): Liste des films disponibles
            n_simulations (int): Nombre d'itérations à effectuer

        Raises:
            ValueError: Si la liste de films est vide ou si n_simulations <= 0
        """
        if len(movie_list) == 0:
            raise ValueError("La liste de films ne peut pas être vide")
        if n_simulations <= 0:
            raise ValueError("Le nombre de simulations doit être positif")

        self.movie_list = movie_list
        self.n_simulations = n_simulations
        self.n_movies = len(movie_list)
        self.simulator = UserFeedbackSimulator(self.n_movies)
        self.ts = ThompsonSampling(self.n_movies)
        self.ratings_history = []
        self.start_time = datetime.utcnow()

    def run_experiment(self, dynamic_mode=False):
        """
        Exécute l'expérience de recommandation.

        Args:
            dynamic_mode (bool): Si True, utilise l'interaction utilisateur

        Returns:
            tuple: (rewards, chosen_arms, regrets)
        """
        rewards = np.zeros(self.n_simulations)
        chosen_arms = np.zeros(self.n_simulations, dtype=int)
        regrets = np.zeros(self.n_simulations)

        for t in range(self.n_simulations):
            chosen_arm = self.ts.select_arm()
            chosen_movie = self.movie_list.iloc[chosen_arm]['title']

            if dynamic_mode:
                print(f"\nItération {t + 1}/{self.n_simulations}")
                print(f"Film recommandé: {chosen_movie}")
                rating = self.simulator.dynamic_feedback(chosen_movie)
            else:
                rating = self.simulator.simulate_feedback(chosen_arm)
                # Ajouter cette ligne pour sauvegarder aussi en mode simulation
                self._update_ratings_history(t, chosen_movie, rating)

            self.ts.update(chosen_arm, rating)

            normalized_rating = (rating - self.MIN_RATING) / (self.MAX_RATING - self.MIN_RATING)
            rewards[t] = normalized_rating
            chosen_arms[t] = chosen_arm
            regrets[t] = 1.0 - normalized_rating

            if dynamic_mode and (t + 1) % self.STATS_DISPLAY_INTERVAL == 0:
                self.show_interim_stats()

        return rewards, chosen_arms, regrets

    def _update_ratings_history(self, iteration, movie, rating):
        """
        Met à jour l'historique des notes.

        Args:
            iteration (int): Numéro de l'itération
            movie (str): Titre du film
            rating (int): Note donnée
        """
        self.ratings_history.append({
            'iteration': iteration + 1,
            'movie': movie,
            'rating': rating,
            'timestamp': datetime.utcnow().isoformat()
        })

    def show_interim_stats(self):
        """Affiche les statistiques intermédiaires de l'expérience."""
        print("\n=== Statistiques intermédiaires ===")
        if self.ratings_history:
            ratings = [r['rating'] for r in self.ratings_history]
            avg_rating = np.mean(ratings)
            print(f"Note moyenne: {avg_rating:.2f} ⭐")
            print("\nDernières notes:")
            for rating in self.ratings_history[-self.STATS_DISPLAY_INTERVAL:]:
                stars = "★" * rating['rating'] + "☆" * (self.MAX_RATING - rating['rating'])
                print(f"{rating['movie']}: {stars}")
        print("================================\n")

    def plot_results(self, rewards, regrets):
        """
        Affiche les graphiques des résultats.

        Args:
            rewards (np.array): Tableau des récompenses
            regrets (np.array): Tableau des regrets
        """
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        cumulative_rewards = np.cumsum(rewards)
        line1 = ax1.plot(cumulative_rewards, 'b-', label='Récompenses cumulées')
        ax1.set_xlabel('Itérations')
        ax1.set_ylabel('Récompenses cumulées', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        cumulative_regrets = np.cumsum(regrets)
        line2 = ax2.plot(cumulative_regrets, 'y-', label='Regret cumulé')
        ax2.set_ylabel('Regret cumulé', color='y')
        ax2.tick_params(axis='y', labelcolor='y')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.title('Évolution des récompenses et du regret')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _calculate_movie_stats(self, chosen_arms):
        movie_stats = {}
        total_selections = len(chosen_arms)

        for i, movie in enumerate(self.movie_list['title']):
            selections = np.sum(chosen_arms == i)
            if selections > 0:
                movie_ratings = [r['rating'] for r in self.ratings_history
                                 if r['movie'] == movie]
                avg_rating = np.mean(movie_ratings) if movie_ratings else 0
                std_rating = np.std(movie_ratings) if len(movie_ratings) > 1 else 0

                movie_stats[movie] = {
                    'selections': selections,
                    'selection_ratio': selections / total_selections,
                    'avg_rating': avg_rating,
                    'std_rating': std_rating,
                    'n_ratings': len(movie_ratings),
                    'popularity_score': (avg_rating * selections) / total_selections
                }
        return movie_stats

    def display_most_popular_movie(self, chosen_arms):
        """
        Affiche les films les plus populaires et leurs statistiques.

        Args:
            chosen_arms (np.array): Tableau des bras choisis
        """
        movie_stats = self._calculate_movie_stats(chosen_arms)

        if movie_stats:
            most_popular_movie = max(movie_stats.items(),
                                     key=lambda x: x[1]['popularity_score'])

            print("\n=== Film le plus populaire ===")
            print(f"Titre : {most_popular_movie[0]}")
            print(f"Nombre de sélections : {most_popular_movie[1]['selections']}")
            if most_popular_movie[1]['avg_rating'] > 0:
                stars = "★" * int(round(most_popular_movie[1]['avg_rating'])) + \
                        "☆" * (self.MAX_RATING - int(round(most_popular_movie[1]['avg_rating'])))
                print(f"Note moyenne : {most_popular_movie[1]['avg_rating']:.2f}/5 {stars}")
                print(f"Score de popularité : {most_popular_movie[1]['popularity_score']:.3f}")

            print("\n=== Top 3 des films ===")
            sorted_movies = sorted(movie_stats.items(),
                                   key=lambda x: x[1]['popularity_score'],
                                   reverse=True)[:3]
            for i, (movie, stats) in enumerate(sorted_movies, 1):
                stars = "★" * int(round(stats['avg_rating'])) + \
                        "☆" * (self.MAX_RATING - int(round(stats['avg_rating'])))
                print(f"{i}. {movie}")
                print(f"   Notes: {stats['avg_rating']:.2f}/5 {stars}")
                print(f"   Sélections: {stats['selections']}")
                print(f"   Score: {stats['popularity_score']:.3f}")

    def save_results(self, filename):
        """
        Sauvegarde les résultats de l'expérience dans un fichier JSON.

        Args:
            filename (str): Nom du fichier de sauvegarde
        """
        import json
        from datetime import datetime

        results = {
            'experiment_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.utcnow().isoformat(),
                'n_movies': self.n_movies,
                'n_simulations': self.n_simulations
            },
            'ratings_history': self.ratings_history
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)