import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

weather = pd.read_csv("weather_data.csv")

location = "New York"

weather_at_location = weather[weather["Location"] == location]

weather["month"] = weather["Date_Time"].apply(lambda date_raw:date_raw[:7])

by_month = weather.groupby("month").agg({"Temperature_C": "mean", "Humidity_pct": "mean"})


print(weather_at_location)
print(by_month.head())


plt.plot(by_month.index, by_month["Temperature_C"])
plt.show()

plt.scatter(by_month.index, by_month["Temperature_C"], c="red")
plt.scatter(by_month.index, by_month["Humidity_pct"], c="blue")
plt.show()