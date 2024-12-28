import numpy as np

# Implémentation de l'algorithme Thompson Sampling
class ThompsonSampling:
    """
    Implémentation de l'algorithme Thompson Sampling avec support pour les notes de 1 à 5.
    """

    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Initialisation des paramètres de la distribution bêta pour chaque bras
        self.alpha = np.ones(n_arms)  # Succès
        self.beta = np.ones(n_arms)  # Échecs
        # Compteurs pour les statistiques
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        """
        Sélectionne un bras (film) en échantillonnant à partir des distributions bêta.
        """
        if np.any(self.counts == 0):
            # S'il reste des bras non testés, en choisir un au hasard
            return np.random.choice(np.where(self.counts == 0)[0])

        # Échantillonner une valeur de chaque distribution bêta
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, chosen_arm, rating):
        """
        Met à jour les paramètres de la distribution bêta pour le bras choisi.

        Args:
            chosen_arm (int): Index du bras choisi
            rating (float): Note reçue (1-5)
        """
        self.counts[chosen_arm] += 1

        # Normaliser la note entre 0 et 1
        normalized_rating = (rating - 1) / 4

        # Mise à jour des paramètres alpha et beta
        # Plus la note est élevée, plus on augmente alpha
        # Plus la note est basse, plus on augmente beta
        self.alpha[chosen_arm] += normalized_rating
        self.beta[chosen_arm] += (1 - normalized_rating)

        # Mise à jour de la valeur moyenne
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * rating