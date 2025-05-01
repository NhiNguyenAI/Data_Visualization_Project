import pandas as pd

df = pd.read_csv("../data/HomeC.csv")
df['Date'] = pd.to_datetime(df['Date'])