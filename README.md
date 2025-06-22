
# Smart Energy Dashboard

## Overview

The **Smart Energy Dashboard** is a web application built with **Streamlit**, **Pandas**, **Plotly**, and several other Python libraries to visualize energy consumption and generation for a smart home system. This dashboard allows users to track energy usage and predict future consumption, providing insights into energy efficiency and sustainability.

### Features
- **Overview Page**: Visualize daily/hourly energy consumption and generation, with a weekly heatmap of usage patterns.
- **Devices Page**: Analyze energy consumption by device, including distribution, trends, and correlations.
- **Forecasting Page**: Predict future energy usage and device consumption using **Prophet** and **ARIMA** models.
- **Weather Page**: Explore relationships between weather conditions (e.g., temperature, humidity) and energy usage.
- **Interactive Filters**: Customize date ranges, devices, and time granularities for tailored analysis.
- **Advanced Analytics**: SHAP feature importance, anomaly detection, and weather-energy correlation analysis.

---

## Dataset

The application uses the **HomeC.csv** dataset, available at Kaggle: [Smart Home Dataset with Weather Information](https://www.kaggle.com/datasets). It contains:

- **Energy Data**: Minute-level power usage (`use [kW]`, `gen [kW]`) and device-specific consumption (e.g., Dishwasher [kW], Fridge [kW]).
- **Weather Data**: Metrics like temperature, humidity, wind speed, and weather summaries.
- **Time Range**: Data spans from January 1, 2016, with 1-minute resolution.

**Note:** The dataset must be placed in a `data/` folder in the project root.

---

## Prerequisites

- **Python**: Version 3.8 (as specified in `environment.yml`).
- **Conda**: For environment management.
- **Dependencies**: Listed in `environment.yml`, including Streamlit, Pandas, Plotly, Prophet, XGBoost, SHAP, and more.

**Note:** The provided `environment.yml` lacks some dependencies (prophet, xgboost, shap, statsmodels). Update it with:
```yaml
name: data_visualization
channels:
  - anaconda
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pip
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - plotly
  - prophet
  - xgboost
  - shap
  - statsmodels
  - pip:
      - ipykernel==6.17.1
      - ipython==8.7.0
      - jupyter-client==7.4.7
      - jupyter-core==5.1.0
      - scipy==1.9.3
      - scikit-learn==1.2.0
```

---

## Prepare Data

1. Download **HomeC.csv** from Kaggle.
2. Create a `data/` folder in the project root and place **HomeC.csv** inside it.

---

## Run the Application

To start the Streamlit application, use the following command:
```bash
streamlit run smart_home.py
```

The app will open in your default web browser (e.g., http://localhost:8501).

---

## Usage

### Navigation:
Use the sidebar to switch between **Overview**, **Devices**, **Forecasting**, and **Weather** pages.

### Interactivity:
Select date ranges, devices, or weather variables to customize visualizations.

### Download Data:
Export sample data as CSV from the sidebar.

### Forecasting:
Choose forecast horizons (7, 14, or 30 days) and view predictions with confidence intervals.

### Weather Analysis:
Explore correlations, regression trends, and anomalies in energy usage based on weather conditions.

---

## Dependencies

Key libraries (see `environment.yml` for full list):
- **streamlit**: Web app framework
- **pandas, numpy**: Data manipulation
- **plotly, seaborn, matplotlib**: Visualization
- **prophet, statsmodels**: Time-series forecasting
- **xgboost, shap**: Machine learning and explainability
- **scikit-learn**: Anomaly detection and preprocessing

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- **Dataset**:  [Smart Home Dataset with Weather Information](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information) by Taranvee on Kaggle.
- Built with **Streamlit** and various open-source data science libraries.
