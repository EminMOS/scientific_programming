import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Daten laden
movies = pd.read_csv("movie_dataset.csv")

# Aufgabe 1: Tortendiagramm für die prozentuale Verteilung der Filme nach Genre
genre_counts = movies['genres'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("Percentage of Movies by Genre")
plt.show()

# Aufgabe 2: Scatterplot für Budget vs. Einnahmen mit Farbkodierung für Genres
plt.figure(figsize=(10, 6))
sns.scatterplot(data=movies, x="budget", y="revenue", hue="genres", palette="viridis", alpha=0.7)
plt.title("Budget vs Revenue by Genre")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Aufgabe 3: Word Cloud für Filmtitel im profitabelsten Genre
# Profit berechnen und profitabelstes Genre finden
movies["profit"] = movies["revenue"] - movies["budget"]
most_profitable_genre = movies.groupby("genres")["profit"].mean().idxmax()

# Filmtitel aus dem profitabelsten Genre sammeln
titles_in_profitable_genre = movies[movies["genres"] == most_profitable_genre]["original_title"]

# Word Cloud generieren
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(titles_in_profitable_genre))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title(f"Word Cloud for Movie Titles in the Most Profitable Genre: {most_profitable_genre}")
plt.show()
