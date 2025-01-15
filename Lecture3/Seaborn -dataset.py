import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Seaborn Tips-Dataset laden
tips = sns.load_dataset("tips")

# Aufgabe 1: Violinplot zur Darstellung der Tip-Verteilung nach Tageszeit
plt.figure(figsize=(8, 6))
sns.violinplot(data=tips, x="time", y="tip", palette="muted")
plt.title("Tip Distribution by Time of Day (Lunch vs. Dinner)")
plt.xlabel("Time of Day")
plt.ylabel("Tip Amount")
plt.show()

# Aufgabe 2: Balkendiagramm für durchschnittliche Tipp-Prozentsätze pro Tag
# Berechnung des Tipp-Prozentsatzes
tips["tip_percentage"] = (tips["tip"] / tips["total_bill"]) * 100

# Durchschnittliche Tipp-Prozentsätze berechnen
avg_tip_percentage = tips.groupby("day")["tip_percentage"].mean().sort_values()

plt.figure(figsize=(8, 6))
sns.barplot(x=avg_tip_percentage.index, y=avg_tip_percentage.values, palette="coolwarm")
plt.title("Average Tip Percentages by Day")
plt.xlabel("Day")
plt.ylabel("Average Tip Percentage (%)")
plt.show()

# Aufgabe 3: Scatterplot für Total Bill vs. Tip mit Regressionslinie
plt.figure(figsize=(8, 6))
sns.regplot(data=tips, x="total_bill", y="tip", scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
plt.title("Total Bill vs. Tip Amount with Regression Line")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip Amount ($)")
plt.show()
