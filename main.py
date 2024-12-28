import pandas as pd
from MovieRecommendationExperiment import MovieRecommendationExperiment
# Charger les données des films uniquement
movies = pd.read_csv('dataset/movie.csv')  # Données sur les films

# Filtrer les films ayant un seuil arbitraire (ex. > 20000) comme critère de popularité présumée
popular_movies = movies['movieId'][:50]  # Exemple : on prend les 50 premiers films (aucune évaluation initiale)
movie_list = movies[movies['movieId'].isin(popular_movies)]

# Afficher le nombre de films populaires sélectionnés
print(f"Nombre de films populaires : {len(movie_list)}")


# Lancer l'expérience
experiment = MovieRecommendationExperiment(movie_list)
rewards, chosen_arms = experiment.run_experiment()
experiment.plot_results(rewards)
experiment.display_most_popular_movie(chosen_arms)
