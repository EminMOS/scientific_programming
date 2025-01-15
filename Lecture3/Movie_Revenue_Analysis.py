import pandas as pd
import numpy as np
import json

movies = pd.read_csv("movie_dataset.csv")
netflix = pd.read_csv("netflix_titles.csv")

pd. merge(movies, netflix, on="title", how="inner")

print(movies.columns)
print(netflix.columns)

print("----------------------------")

# Neue Spalte 'Profit' erstellen
movies["Profit"] = movies["revenue"] - movies["budget"]

print(movies[["title", "revenue", "budget", "Profit"]].head())

print("----------------------------")

# Nach Länder Profit Listen
by_country = movies.groupby("production_countries").agg({"Profit": "sum"}).sort_values("Profit")

print(by_country.head())

print("----------------------------")

# Filme nach Profit sortieren
movies_sorted = movies.sort_values(by="Profit", ascending=False)

print(movies_sorted[["title", "Profit"]].head())

print("----------------------------")

# Profitabelste Genres finden
genre_profit = movies.groupby("genres").agg({"Profit": "sum"}).sort_values(by="Profit", ascending=False)

print(genre_profit.head())