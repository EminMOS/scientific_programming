import pandas as pd

# Ersetze den Pfad mit dem vollständigen Pfad zu deiner Datei
file_path = "C:/Users/emina/OneDrive/Desktop/test/Scientific Programming/LectureTasks/diabetes.csv"


# CSV-Datei laden
data = pd.read_csv(file_path)

# Die ersten 5 Zeilen anzeigen
print(data.head())

# Informationen über den Datensatz anzeigen
print(data.info())

print(data.describe())

print(data["Glucose"].value_counts())
