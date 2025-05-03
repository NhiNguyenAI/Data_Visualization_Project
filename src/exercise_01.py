import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
import numpy as np


df = pd.read_csv("data/HomeC.csv")
df.info()

# Idee
# User Categorical for Use Energie of the the devices to find out the most used devices
# Constants
DEVICE_COLUMNS = [
    'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
    'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]',
    'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]',
    'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',
    'Microwave [kW]', 'Living room [kW]', 'Solar [kW]'
]

WEATHER_COLUMNS = [
    'temperature', 'humidity', 'windSpeed', 
    'windBearing', 'pressure', 'apparentTemperature',
    'dewPoint', 'precipProbability'
]


# Construct absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data", "HomeC.csv")


# First try reading with automatic datetime parsing
df = pd.read_csv(
    data_path,
    dtype={col: 'float32' for col in DEVICE_COLUMNS + WEATHER_COLUMNS},
    low_memory=False  # Optional, based on your needs
)

# Ensure the 'time' column is in the correct datetime format
# Check the first few rows of the 'time' column
df['time'].head()
# If the format is known, specify it during conversion

# Check for missing values or malformed rows
df['time'].isnull().sum(), df['time'].unique()
df['datetime'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
# Drop the original 'time' column if you no longer need it
df = df.drop(columns=['time'])
df




