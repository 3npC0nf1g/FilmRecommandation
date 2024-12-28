import pandas as pd
from movie_recommendation_experiment import MovieRecommendationExperiment


# Charger les données des films
movies = pd.read_csv('dataset/movie.csv')  # Données sur les films

# Filtrer les films
popular_movies = movies['movieId'][:200]  # Exemple : on prend les 200 premiers films (aucune évaluation initiale)
movie_list = movies[movies['movieId'].isin(popular_movies)]

# Afficher le nombre de films  sélectionnés
print(f"Nombre de films : {len(movie_list)}")


# Lancer l'expérience
experiment = MovieRecommendationExperiment(movie_list)
rewards, chosen_arms = experiment.run_experiment()
experiment.plot_results(rewards)
experiment.display_most_popular_movie(chosen_arms)
