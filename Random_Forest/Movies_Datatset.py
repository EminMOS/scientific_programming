import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

movies = pd.read_csv("C:/Users/emina/OneDrive/Desktop/test/Scientific Programming/Random_Forest/movie_dataset.csv")

print("Dataset Info:")
print(movies.info())
print("\nFirst 5 Rows:")
print(movies.head())

movies['genres'] = movies['genres'].fillna('Unknown')
movies['director'] = movies['director'].fillna('Unknown')

movies = pd.get_dummies(movies, columns=["genres", "original_language"])

features_revenue = ['budget', 'vote_average'] + [col for col in movies.columns if col.startswith('genres_') or col.startswith('original_language_')]
target_revenue = 'revenue'

X_revenue = movies[features_revenue]
y_revenue = movies[target_revenue]

X_train_rev, X_test_rev, y_train_rev, y_test_rev = train_test_split(
    X_revenue, y_revenue, test_size=0.2, random_state=42
)

# Task 1: 
revenue_model = RandomForestRegressor(random_state=42)
revenue_model.fit(X_train_rev, y_train_rev)

revenue_predictions = revenue_model.predict(X_test_rev)
print("\nRevenue Prediction RMSE:", np.sqrt(mean_squared_error(y_test_rev, revenue_predictions)))

# Task 2: 
features_genre = ['budget', 'vote_average', 'popularity'] + [col for col in movies.columns if col.startswith('original_language_')]
target_genre = 'genres_Unknown'  

X_genre = movies[features_genre]
y_genre = movies[target_genre]

X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(
    X_genre, y_genre, test_size=0.2, random_state=42
)

genre_model = RandomForestClassifier(random_state=42)
genre_model.fit(X_train_gen, y_train_gen)

genre_predictions = genre_model.predict(X_test_gen)
print("\nGenre Classification Report:")
print(classification_report(y_test_gen, genre_predictions, zero_division=0))

features_rating = ['budget', 'vote_average', 'popularity'] + [col for col in movies.columns if col.startswith('genres_') or col.startswith('original_language_')]
target_rating = 'vote_average'

X_rating = movies[features_rating]
y_rating = movies[target_rating]

scaler = StandardScaler()
X_rating_scaled = scaler.fit_transform(X_rating)

X_train_rat, X_test_rat, y_train_rat, y_test_rat = train_test_split(
    X_rating_scaled, y_rating, test_size=0.2, random_state=42
)

rating_model = RandomForestRegressor(random_state=42)
rating_model.fit(X_train_rat, y_train_rat)


rating_predictions = rating_model.predict(X_test_rat)
print("\nRating Prediction RMSE:", np.sqrt(mean_squared_error(y_test_rat, rating_predictions)))

numeric_movies = movies.select_dtypes(include=[np.number])

sns.heatmap(numeric_movies.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

importance = revenue_model.feature_importances_

plt.figure(figsize=(12, 6)) 
plt.bar(features_revenue, importance)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Importance", fontsize=12)
plt.title("Feature Importance for Revenue Prediction", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=10)  
plt.tight_layout() 
plt.show()

