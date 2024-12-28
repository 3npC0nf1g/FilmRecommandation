import pandas as pd
from movie_recommendation_experiment import MovieRecommendationExperiment


# Charger les données des films
movies = pd.read_csv('dataset/movie.csv')  # Données sur les films

# Filtrer les films
popular_movies = movies['movieId'][:50]  # Exemple : on prend les 50 premiers films (aucune évaluation initiale)
movie_list = movies[movies['movieId'].isin(popular_movies)]

# Afficher le nombre de films  sélectionnés
print(f"Nombre de films : {len(movie_list)}")

print("\nDétails des films :")
for i, movie in enumerate(movie_list['title']):
    print(f"{i+1}. {movie}")


# Lancer l'expérience
experiment = MovieRecommendationExperiment(movie_list)
mode = input("Voulez-vous utiliser le mode dynamique ? (y/n): ").strip().lower()
dynamic_mode = mode == 'y'
rewards, chosen_arms, regrets = experiment.run_experiment(dynamic_mode=dynamic_mode)
experiment.plot_results(rewards, regrets)
experiment.display_most_popular_movie(chosen_arms)
