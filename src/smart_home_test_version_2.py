import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# Configuration
st.set_page_config(
    page_title="Smart Home Energy Analysis",
    layout="wide",
    page_icon="🏠"
)

@st.cache_data
def load_data():
    """Load and preprocess the energy data"""
    try:
        file_path = os.path.join("data", "HomeC.csv")
        data = pd.read_csv(file_path, low_memory=False)
        
        # Remove last row if it contains NaN
        data = data[:-1]
        
        # Convert time column to datetime
        if 'time' in data.columns:
            try:
                data['datetime'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
                
                # Fallback to direct conversion if needed
                if data['datetime'].isnull().any():
                    data['datetime'] = pd.to_datetime(data['time'], errors='coerce')
                
                # Set datetime as index
                data = data.set_index('datetime').sort_index()
                
            except Exception as e:
                st.error(f"Time conversion error: {str(e)}")
                # Create a default timeline if conversion fails
                data['datetime'] = pd.date_range(start='2016-01-01', periods=len(data), freq='min')
                data = data.set_index('datetime')
        
        # Keep only numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        return data[numeric_cols].dropna()
    
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return None

def calculate_daily_energy(df, power_col='use [kW]'):
    """Calculate daily energy consumption in kWh"""
    if power_col not in df.columns:
        return pd.DataFrame()
    return df[[power_col]].resample('D').sum() / 60  # Convert kW to kWh

def display_hourly_consumption(data, selected_date):
    """Display hourly consumption for a selected day"""
    daily_data = data[data.index.date == selected_date]
    
    if not daily_data.empty:
        fig = px.area(
            daily_data, 
            x=daily_data.index, 
            y='use [kW]',
            title=f"Power Consumption on {selected_date.strftime('%d/%m/%Y')}",
            labels={'use [kW]': 'Power (kW)', 'datetime': 'Time'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate daily total
        daily_kWh = daily_data['use [kW]'].sum() / 60
        st.metric("Total Daily Consumption", f"{daily_kWh:.2f} kWh")
    else:
        st.warning("No data available for selected date")

def display_daily_summary(daily_energy, start_date, end_date):
    """Display summary of daily energy consumption"""
    if not daily_energy.empty:
        fig = px.bar(
            daily_energy,
            x=daily_energy.index,
            y='use [kW]',
            title=f"Energy Consumption from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}",
            labels={'use [kW]': 'Energy (kWh)', 'datetime': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary metrics
        total = daily_energy['use [kW]'].sum()
        avg = daily_energy['use [kW]'].mean()
        
        cols = st.columns(3)
        cols[0].metric("Total Energy", f"{total:.2f} kWh")
        cols[1].metric("Daily Average", f"{avg:.2f} kWh")
        cols[2].metric("Days Analyzed", len(daily_energy))
    else:
        st.warning("No data available for selected period")

def main():
    st.title("🏠 Smart Home Energy Analysis Dashboard")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Check for required column
    if 'use [kW]' not in data.columns:
        st.error("Required column 'use [kW]' not found in data")
        st.write("Available numeric columns:", data.columns.tolist())
        return
    
    # Date range selection
    st.sidebar.header("Date Range Selection")
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date", 
            min_date, 
            min_value=min_date, 
            max_value=max_date
        )
    with col2:
        end_date = st.date_input(
            "End Date", 
            max_date, 
            min_value=min_date, 
            max_value=max_date
        )
    
    if start_date > end_date:
        st.error("End date must be after start date!")
        return
    
    try:
        # Filter data by date range
        filtered = data.loc[f"{start_date}":f"{end_date}"]
        filtered = filtered.select_dtypes(include=['number'])
        
        # Calculate daily energy
        daily_energy = calculate_daily_energy(filtered)
        
        # Create tabs
        tab1, tab2 = st.tabs(["📈 Daily Detail", "📅 Period Summary"])
        
        with tab1:
            # Get unique dates with data
            valid_dates = pd.Series(filtered.index.date).unique()
            
            if len(valid_dates) == 0:
                st.warning("No data available for selected period")
                return
                
            selected_date = st.selectbox(
                "Select a day to analyze",
                options=valid_dates,
                format_func=lambda x: x.strftime("%d/%m/%Y")
            )
            
            display_hourly_consumption(filtered, selected_date)
        
        with tab2:
            display_daily_summary(daily_energy, start_date, end_date)
            
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")

if __name__ == "__main__":
    main()