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

    def __init__(self, movie_list, n_simulations=100):
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
        if n_simulations > len(movie_list):
            n_simulations = len(movie_list)
            print(f"Le nombre de simulations a été ajusté à {n_simulations} (nombre total de films)")

        self.movie_list = movie_list
        self.n_simulations = n_simulations
        self.n_movies = len(movie_list)
        self.simulator = UserFeedbackSimulator(self.n_movies)
        self.ratings_history = []
        self.start_time = datetime.utcnow()
        self.ts = ThompsonSampling(self.n_movies)
        self.best_movie_history = []  # Pour suivre l'évolution du meilleur film

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

            # Vérifier si tous les films ont été évalués
            if chosen_arm is None:
                print("\nTous les films ont été évalués!")
                return rewards[:t], chosen_arms[:t], regrets[:t]

            chosen_movie = self.movie_list.iloc[chosen_arm]['title']

            if dynamic_mode:
                print(f"\nItération {t + 1}/{self.n_simulations}")
                print(f"Film recommandé: {chosen_movie}")
                rating = self.simulator.dynamic_feedback(chosen_movie)
            else:
                rating = self.simulator.simulate_feedback(chosen_arm)

            self.ts.update(chosen_arm, rating)
            self._update_ratings_history(t, chosen_movie, rating)

            normalized_rating = (rating - self.__class__.MIN_RATING) / (
                        self.__class__.MAX_RATING - self.__class__.MIN_RATING)
            rewards[t] = normalized_rating
            chosen_arms[t] = chosen_arm
            regrets[t] = 1.0 - normalized_rating

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

    def plot_results(self, rewards, regrets):
        """
        Affiche les résultats avec plus de détails statistiques.
        """
        plt.figure(figsize=(15, 10))

        # Subplot 1: Récompenses et regrets
        plt.subplot(2, 1, 1)
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        cumulative_rewards = np.cumsum(rewards)
        avg_rewards = cumulative_rewards / np.arange(1, len(rewards) + 1)
        ax1.plot(avg_rewards, 'b-', label='Récompenses moyennes')

        cumulative_regrets = np.cumsum(regrets)
        ax2.plot(cumulative_regrets, 'r-', label='Regret cumulé')

        ax1.set_xlabel('Itérations')
        ax1.set_ylabel('Récompenses moyennes', color='b')
        ax2.set_ylabel('Regret cumulé', color='r')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Subplot 2: Intervalles de confiance
        plt.subplot(2, 1, 2)
        self.plot_confidence_intervals()

        plt.tight_layout()
        plt.show()

    def plot_confidence_intervals(self):
        """
        Affiche les intervalles de confiance pour chaque film.
        """
        lower_bounds, upper_bounds = self.ts.get_confidence_bounds()
        mean_ratings = self.ts.get_mean_ratings()

        movies = np.arange(self.n_movies)

        # Calculer les erreurs de manière à ce qu'elles soient toujours positives
        yerr_lower = mean_ratings - lower_bounds
        yerr_upper = upper_bounds - mean_ratings

        # S'assurer que les erreurs sont positives
        yerr_lower = np.maximum(0, yerr_lower)
        yerr_upper = np.maximum(0, yerr_upper)

        plt.errorbar(movies, mean_ratings,
                     yerr=[yerr_lower, yerr_upper],
                     fmt='o', capsize=5, capthick=1, elinewidth=1)

        # Ajouter les titres des films en rotation pour une meilleure lisibilité
        plt.xticks(movies, self.movie_list['title'], rotation=45, ha='right')

        plt.xlabel('Films')
        plt.ylabel('Note estimée (avec intervalle de confiance)')
        plt.title('Estimation de la qualité des films')
        plt.grid(True)

        # Ajuster la mise en page pour éviter que les titres soient coupés
        plt.tight_layout()

    def _calculate_movie_stats(self, chosen_arms):
        """
        Calcule les statistiques détaillées pour chaque film.

        Args:
            chosen_arms (np.array): Tableau des bras choisis pendant l'expérience

        Returns:
            dict: Dictionnaire contenant les statistiques pour chaque film
        """
        movie_stats = {}
        total_selections = len(chosen_arms)

        for i, movie in enumerate(self.movie_list['title']):
            selections = np.sum(chosen_arms == i)
            if selections > 0:
                movie_ratings = [r['rating'] for r in self.ratings_history
                                 if r['movie'] == movie]
                avg_rating = np.mean(movie_ratings) if movie_ratings else 0
                std_rating = np.std(movie_ratings) if len(movie_ratings) > 1 else 0

                # Utiliser les statistiques de Thompson Sampling
                mean_ratings = self.ts.get_mean_ratings()
                lower_bounds, upper_bounds = self.ts.get_confidence_bounds()

                movie_stats[movie] = {
                    'selections': selections,
                    'selection_ratio': selections / total_selections,
                    'avg_rating': avg_rating,
                    'std_rating': std_rating,
                    'n_ratings': len(movie_ratings),
                    'ts_mean_rating': mean_ratings[i],
                    'confidence_lower': lower_bounds[i],
                    'confidence_upper': upper_bounds[i],
                    'popularity_score': (avg_rating * selections) / total_selections
                }
        return movie_stats
    def display_most_popular_movie(self, chosen_arms):
        """
        Affiche les statistiques détaillées des films.
        """
        movie_stats = self._calculate_movie_stats(chosen_arms)
        best_arm, best_rating = self.ts.get_best_arm()

        print("\n=== Statistiques des films ===")
        sorted_movies = sorted(movie_stats.items(),
                               key=lambda x: x[1]['avg_rating'],
                               reverse=True)

        for i, (movie, stats) in enumerate(sorted_movies, 1):
            stars = "★" * int(round(stats['avg_rating'])) + \
                    "☆" * (self.MAX_RATING - int(round(stats['avg_rating'])))

            print(f"\n{i}. {movie}")
            print(f"   Note moyenne: {stats['avg_rating']:.2f}/5 {stars}")
            print(f"   Nombre d'évaluations: {stats['n_ratings']}")

            # Ajouter les intervalles de confiance
            lower, upper = self.ts.get_confidence_bounds()
            movie_idx = self.movie_list[self.movie_list['title'] == movie].index[0]
            print(f"   Intervalle de confiance: [{lower[movie_idx]:.2f}, {upper[movie_idx]:.2f}]")
    def save_results_csv(self, filename):
        """
        Sauvegarde les résultats de l'expérience dans un fichier CSV.

        Args:
            filename (str): Nom du fichier de sauvegarde (ex: 'resultats_experience.csv')
        """
        import csv
        from datetime import datetime

        # Ouvrir le fichier en mode écriture
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Écrire l'en-tête
            writer.writerow(['Iteration', 'Movie', 'Rating', 'Timestamp'])

            # Écrire les données des évaluations
            for record in self.ratings_history:
                writer.writerow([
                    record['iteration'],
                    record['movie'],
                    record['rating'],
                    record['timestamp']
                ])

        print(f"Résultats sauvegardés dans {filename}")
