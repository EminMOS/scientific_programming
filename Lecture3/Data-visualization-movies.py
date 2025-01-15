import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Daten laden
movies = pd.read_csv("movie_dataset.csv")
netflix = pd.read_csv("netflix_titles.csv")

# Debugging: Spalten der DataFrames anzeigen
print(movies.columns)
print(netflix.columns)

# Aufgabe 1: Verteilung der Filmbewertungen visualisieren
plt.figure(figsize=(8, 6))
sns.histplot(data=movies, x="vote_average", kde=True, bins=20, color="blue")
plt.title("Distribution of Movie Ratings")
plt.xlabel("Ratings")
plt.ylabel("Frequency")
plt.show()

# Aufgabe 2: Balkendiagramm für Top-Genres nach durchschnittlicher Bewertung
top_genres = movies.groupby("genres")["vote_average"].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette="viridis")
plt.title("Top Genres by Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("Genre")
plt.show()

# Aufgabe 3: Heatmap zur Darstellung von Korrelationen
plt.figure(figsize=(10, 8))
correlation_matrix = movies.select_dtypes(include=np.number).corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Movie Features")
plt.show()
