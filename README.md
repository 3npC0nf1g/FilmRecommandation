# **Thompson Sampling for Movie Recommendation**

## **1. Project Overview**
This project implements a **movie recommendation system** using the **Thompson Sampling algorithm**. The goal is to identify the most popular movie with minimal user feedback, modeled as a **multi-armed bandit problem**.

The system supports two modes:
1. **Simulated Feedback Mode** - Generates synthetic user feedback based on predefined probabilities.
2. **Dynamic Feedback Mode** - Collects real-time user feedback through the command line.

It also tracks metrics like **cumulative regret** and **rewards** to evaluate performance.

---

## **2. How It Works**

### **2.1 Problem Definition**
- Each movie is treated as an **arm** in the bandit problem.
- Users provide binary feedback:
  - **1** = Liked the movie.
  - **0** = Disliked the movie.
- The objective is to identify the movie with the **highest reward probability** while minimizing the number of trials.

### **2.2 Thompson Sampling Algorithm**
- Uses Bayesian inference to balance **exploration** (trying less-known movies) and **exploitation** (selecting movies likely to be popular).
- Updates movie probabilities based on user feedback using **Beta distributions**.

### **2.3 Evaluation Metrics**
- **Cumulative Reward**: Tracks the total positive feedback received.
- **Cumulative Regret**: Measures how much reward is lost by not always choosing the optimal movie.

---

## **3. Setup Instructions**

### **3.1 Requirements**
Ensure Python is installed (version 3.7+). Install dependencies:
```bash
pip install numpy pandas matplotlib
```

### **3.2 Files Needed**
1. **Movie Data**: A CSV file (`movie.csv`) containing:
   - `movieId`: Unique identifier for each movie.
   - `title`: Title of the movie.
2. **Script**: The Python script implementing the recommendation system.

### **3.3 Running the Script**
```bash
python main.py
```

- **Dynamic Mode**: Choose this option to provide real-time feedback.
  ```
  Voulez-vous utiliser le mode dynamique ? (y/n): y
  Film proposé : Toy Story
  Avez-vous aimé le film ? (1 pour Oui, 0 pour Non): 1
  ```
- **Simulated Mode**: Choose this option for synthetic feedback without manual input.

---

## **4. Key Components**

### **4.1 User Feedback Simulator**
Simulates user preferences using predefined probabilities to generate binary feedback.

### **4.2 Thompson Sampling Class**
Implements Bayesian updates to improve recommendations over time.

### **4.3 Movie Recommendation Experiment**
Coordinates simulations, collects feedback, updates probabilities, and visualizes performance.

---

## **5. Results and Visualizations**

### **5.1 Metrics**
- **Cumulative Reward**: Shows the total positive feedback received during trials.
- **Cumulative Regret**: Displays lost opportunities when suboptimal movies are selected.

### **5.2 Visualization**
Generates a plot to track rewards and regrets over time:
```python
plt.plot(cumulative_rewards, label='Cumulative Rewards')
plt.plot(cumulative_regrets, label='Cumulative Regret')
plt.legend()
plt.show()
```

---

## **6. Future Extensions**

1. **Web-Based UI**:
   - Replace command-line interaction with a graphical interface.

---

## **7. Conclusion**
This project demonstrates the effectiveness of **Thompson Sampling** for movie recommendation, balancing exploration and exploitation. It also provides insights into regret minimization and user-centric recommendation strategies.

---

## **8. References**
- [MoviesLens Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?resource=download&select=tag.csv)
- [Multi-armed bandits for dynamic movie recommendations](https://blog.insightdatascience.com/multi-armed-bandits-for-dynamic-movie-recommendations-5eb8f325ed1d)

