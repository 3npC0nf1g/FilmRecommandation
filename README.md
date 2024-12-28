# Movie Recommendation System with Thompson Sampling

## Overview

This project implements an intelligent movie recommendation system using Thompson Sampling algorithm with a 5-star rating system. It's designed to learn from user preferences and optimize movie recommendations over time.

## Features

- 🌟 5-star rating system (1-5 stars)
- 🔄 Two operating modes: Simulation and Interactive
- 📊 Real-time performance visualization
- 📈 Detailed statistics and movie rankings
- 🎯 Adaptive learning through Thompson Sampling

## Requirements

```bash
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/3npC0nf1g/movie-recommendation-system.git
cd movie-recommendation-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── README.md
├── src/
│   ├── main.py
│   ├── movie_recommendation_experiment.py
│   ├── thompson_sampling.py
│   └── user_feedback_simulator.py
├── dataset/
│   └── movie.csv
└── docs/
    └── results.md
```

## Usage

Run the main script:
```bash
python src/main.py
```

Choose your preferred mode:
1. **Simulation Mode**: Automated testing with simulated user ratings
2. **Interactive Mode**: Manual rating system where you can rate movies

## Example Output

```
=== Film le plus populaire ===
Titre : Movie Title
Note moyenne : 4.25/5 ★★★★☆
Nombre de sélections : 25
Score de popularité : 0.850

=== Top 3 des films ===
1. Movie A
   Notes: 4.50/5 ★★★★★
   Sélections: 20
   Score: 0.800
```

## Algorithm Details

The Thompson Sampling implementation uses Beta distributions to model movie ratings:

```python
def select_arm(self):
    if np.any(self.counts == 0):
        return np.random.choice(np.where(self.counts == 0)[0])
    samples = np.random.beta(self.alpha, self.beta)
    return np.argmax(samples)
```

## Performance Metrics

- **Cumulative Reward**: Sum of normalized ratings
- **Cumulative Regret**: Difference between optimal and received ratings
- **Average Rating**: Mean rating per movie
- **Selection Frequency**: Movie selection distribution

## Visualization

The system generates plots showing:
- Cumulative rewards (blue line)
- Cumulative regret (yellow line)
- Movie selection distribution


## References
- [MoviesLens Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?resource=download&select=tag.csv)
- [Multi-armed bandits for dynamic movie recommendations](https://blog.insightdatascience.com/multi-armed-bandits-for-dynamic-movie-recommendations-5eb8f325ed1d)
