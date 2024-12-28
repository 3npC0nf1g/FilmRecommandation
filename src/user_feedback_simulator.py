import numpy as np

# Classe pour la simulation des feedbacks utilisateurs
class UserFeedbackSimulator:
    def __init__(self, n_movies, seed=42):
        np.random.seed(seed)
        # Générer des vraies qualités de films plus variées
        self.true_qualities = np.random.beta(2, 2, size=n_movies)  # Distribution plus uniforme

    def simulate_feedback(self, movie_index):
        quality = self.true_qualities[movie_index]
        # Ajouter un bruit gaussien pour introduire de l'incertitude
        noisy_quality = np.clip(quality + np.random.normal(0, 0.1), 0, 1)
        rating = int(np.round(1 + (noisy_quality * 4)))
        return max(1, min(5, rating))

    def dynamic_feedback(self, movie_title):
        while True:
            try:
                print(f"\nNotez le film '{movie_title}' de 1 à 5 étoiles:")
                print("1 - ★☆☆☆☆ (Très mauvais)")
                print("2 - ★★☆☆☆ (Mauvais)")
                print("3 - ★★★☆☆ (Moyen)")
                print("4 - ★★★★☆ (Bon)")
                print("5 - ★★★★★ (Excellent)")
                rating = int(input("\nVotre note: "))
                if 1 <= rating <= 5:
                    return rating
                print("⚠️ Erreur: Veuillez entrer une note entre 1 et 5.")
            except ValueError:
                print("⚠️ Erreur: Veuillez entrer un nombre valide.")