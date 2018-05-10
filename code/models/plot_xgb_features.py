import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

gains = [
    1204987.0626159394,
    1659050.998686975,
    1867418.8048780488,
    2422821.0159762297,
    2783757.624978355,
    2930197.156500419,
    3683953.060306203,
    3932692.6545359883,
    4843658.562030093,
    5847987.939604147,
    7222484.991683179,
    7229023.399610549,
    8796664.757133266,
    9572283.127301527,
    13301258.481861072,
    13443352.6347884,
    30620816.348913576,
    54044650.71347379,
    75731296.32970577,
    180522663.75268525
]

names = [
    "distance", "trig_distance", "vicenty_distance", "pickup_hour", "intersections",
    "day_of_week", "day_of_year", "pickup_month", "dropoff_latitude", "dropoff_longitude",
    "bearing", "pickup_longitude", "precipit_mm", "pickup_day", "week_of_year",
    "turns", "pickup_latitude", "vendor_id", "manhattan_distance", "pickup_minute"
]

gains = list(reversed(gains))

df = pd.DataFrame()
df["Average information gain"] = gains
df["Feature"] = names

sns.barplot(y="Feature", x="Average information gain", data=df, orient="h")
plt.show()
