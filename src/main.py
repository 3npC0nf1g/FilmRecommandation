import pandas as pd
from movie_recommendation_experiment import MovieRecommendationExperiment
import os
# Charger les données des films
movies = pd.read_csv('dataset/movie.csv')

popular_movies = movies['movieId'][:10]
movie_list = movies[movies['movieId'].isin(popular_movies)]

# Afficher le nombre de films sélectionnés
print("=== Système de recommandation de films ===")
print(f"Nombre de films disponibles : {len(movie_list)}")

print("\nCatalogue des films :")
for i, movie in enumerate(movie_list['title'], 1):
    print(f"{i:2d}. {movie}")

# Lancer l'expérience
experiment = MovieRecommendationExperiment(movie_list)

print("\nModes disponibles:")
print("1. Mode automatique (simulation)")
print("2. Mode interactif (notation manuelle)")
while True:
    try:
        mode_choice = int(input("\nChoisissez le mode (1 ou 2): "))
        if mode_choice in [1, 2]:
            break
        print("⚠️ Erreur: Veuillez choisir 1 ou 2.")
    except ValueError:
        print("⚠️ Erreur: Veuillez entrer un nombre valide.")

dynamic_mode = mode_choice == 2

if dynamic_mode:
    print("\n=== Mode interactif activé ===")
    print("Vous allez noter des films sur une échelle de 1 à 5 étoiles.")
    print("Le système utilisera Thompson Sampling pour apprendre vos préférences.")
    n_iterations = int(input("Combien de films souhaitez-vous évaluer ? (recommandé: 5-10): "))
    if n_iterations > len(movie_list):
        n_iterations = len(movie_list)
        print(f"Nombre d'évaluations ajusté à {n_iterations} (nombre total de films)")
    experiment.n_simulations = n_iterations

rewards, chosen_arms, regrets = experiment.run_experiment(dynamic_mode=dynamic_mode)
experiment.plot_results(rewards, regrets)
experiment.display_most_popular_movie(chosen_arms)
os.makedirs('results', exist_ok=True)
filename = 'results/resultats_experience.csv'
experiment.save_results_csv(filename)
