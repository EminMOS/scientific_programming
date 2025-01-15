import random
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/nileshely/SuperStore-Dataset-2019-2022/refs/heads/main/superstore_dataset.csv") #data set lesen lassen
print(df.columns)
print(df.head())

print("-------------------------")

#Task 1
df["mean_profit"] = df ["profit"]
by_region = df.groupby("region").agg({"sales": 'sum', "profit": 'sum', "mean_profit": "mean"}).rename(columns= {"profit": "Total profit"}) # was soll für die regionen angezeigt werden (wir erzeugen eine geordnete tabelle)
print(by_region)

print("-------------------------")

#Task 2
by_category = df.groupby("category").agg({"sales": 'sum', "profit": 'sum', "mean_profit": "mean"}).rename(columns= {"profit": "Total profit"}) #dasselbe wie für die regionen nur für die Kategorien
print(by_category)

print("-------------------------")

# Task 3

print(df["sales"].isnull()) # überprüfen ob die spalte sales fehlende daten hat -> false -> alles ist voll

for row in range(len(df)) : 
    
    if random.random()  > 0.10: # zahl zwischen 0 und 1 bei random und dann > 0,1 größer 10 %
        
     df.loc[row, "sales"] = None # daten werden hier gelöscht 
     
print(df.head)
print(df[df.isna()["sales"]])

df["sales"] = df["sales"].fillna(df["sales"].mean()) #die leere stelle in der spalte sales wird mit einem durchschnitts Wert gefüllt

print(df.head())
