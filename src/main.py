import pandas as pd
from movie_recommendation_experiment import MovieRecommendationExperiment

# Charger les données des films
movies = pd.read_csv('dataset/movie.csv')

# Filtrer les films
popular_movies = movies['movieId'][:50]
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
    n_iterations = int(input("Combien de films souhaitez-vous évaluer ? (recommandé: 10-20): "))
    experiment.n_simulations = n_iterations

rewards, chosen_arms, regrets = experiment.run_experiment(dynamic_mode=dynamic_mode)
experiment.plot_results(rewards, regrets)
experiment.display_most_popular_movie(chosen_arms)