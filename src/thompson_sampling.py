import numpy as np


class ThompsonSampling:
    """
    Implementation de Thompson Sampling pour la recommandation de films.
    Utilise une distribution Beta pour modéliser l'incertitude sur la qualité de chaque film.
    """

    def __init__(self, n_arms):
        """
        Initialise le Thompson Sampling.

        Args:
            n_arms (int): Nombre de films disponibles
        """
        self.n_arms = n_arms
        # Paramètres de la distribution Beta pour chaque film
        self.alpha = np.ones(n_arms)  # Succès (somme des notes normalisées)
        self.beta = np.ones(n_arms)  # "Échecs" (complement des notes)
        self.rated_arms = set()  # Films déjà évalués
        self.total_pulls = 0  # Nombre total d'évaluations
        self.ratings_sum = np.zeros(n_arms)  # Somme des notes pour chaque film
        self.ratings_count = np.zeros(n_arms)  # Nombre d'évaluations par film

    def get_mean_ratings(self):
        """
        Calcule la note moyenne pour chaque film.

        Returns:
            np.array: Notes moyennes pour chaque film
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.divide(self.ratings_sum, self.ratings_count)
            means[np.isnan(means)] = 0  # Remplace NaN par 0 pour les films non notés
        return means

    def select_arm(self):
        """
        Sélectionne le prochain film à recommander en utilisant Thompson Sampling.

        Returns:
            int or None: Index du film sélectionné, None si tous les films ont été évalués
        """
        # Vérifier si tous les films ont été évalués
        if len(self.rated_arms) == self.n_arms:
            return None

        # Filtrer les films non évalués
        available_arms = np.array([i for i in range(self.n_arms) if i not in self.rated_arms])

        # Échantillonner à partir des distributions Beta
        samples = np.random.beta(
            self.alpha[available_arms],
            self.beta[available_arms]
        )

        # Sélectionner le film avec l'échantillon le plus élevé
        selected_idx = np.argmax(samples)
        return available_arms[selected_idx]

    def update(self, chosen_arm, rating):
        """
        Met à jour les paramètres après avoir reçu une note.

        Args:
            chosen_arm (int): Index du film évalué
            rating (float): Note donnée (1-5)
        """
        if chosen_arm in self.rated_arms:
            raise ValueError("Ce film a déjà été évalué!")

        self.rated_arms.add(chosen_arm)
        self.total_pulls += 1

        # Mettre à jour les statistiques de base
        self.ratings_sum[chosen_arm] += rating
        self.ratings_count[chosen_arm] += 1

        # Normaliser la note entre 0 et 1 pour la distribution Beta
        normalized_rating = (rating - 1) / 4

        # Mettre à jour les paramètres de la distribution Beta
        self.alpha[chosen_arm] += normalized_rating
        self.beta[chosen_arm] += (1 - normalized_rating)

    def get_best_arm(self):
        """
        Retourne le meilleur film basé sur les notes moyennes.

        Returns:
            int: Index du meilleur film
            float: Note moyenne du meilleur film
        """
        mean_ratings = self.get_mean_ratings()
        best_arm = np.argmax(mean_ratings)
        return best_arm, mean_ratings[best_arm]

    def get_confidence_bounds(self):
        """
        Calcule les intervalles de confiance pour chaque film.

        Returns:
            tuple: (lower_bounds, upper_bounds)
        """
        # Utiliser la distribution Beta pour calculer les intervalles
        lower_bounds = np.zeros(self.n_arms)
        upper_bounds = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            if self.ratings_count[arm] > 0:
                samples = np.random.beta(self.alpha[arm], self.beta[arm], 1000)
                lower_bounds[arm] = np.percentile(samples, 2.5)
                upper_bounds[arm] = np.percentile(samples, 97.5)

        return lower_bounds, upper_bounds