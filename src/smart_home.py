import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
import numpy as np

# Configuration
st.set_page_config(
    page_title="Smart Home Energy Dashboard",
    layout="wide",
    page_icon="🏠",
    initial_sidebar_state="expanded"
)

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

# Data Loading with Robust Error Handling
@st.cache_data(ttl=3600)
def load_data():
    """Load and validate the energy data with comprehensive error handling"""
    try:
        # Construct absolute path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "data", "HomeC.csv")

        # Try reading the CSV file with specified dtypes for efficiency
        df = pd.read_csv(
            data_path,
            dtype={col: 'float32' for col in DEVICE_COLUMNS + WEATHER_COLUMNS},
            low_memory=False
        )

        # Ensure the 'time' column is in the correct datetime format
        if 'time' not in df.columns:
            raise ValueError("The 'time' column is missing from the dataset.")

        # Convert 'time' column from Unix timestamp to datetime
        df['datetime'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
        
        # Drop the original 'time' column if you no longer need it
        df = df.drop(columns=['time'])
        
        # Handle large datasets by sampling the last 2000 rows if necessary
        if len(df) > 500000:
            sample_size = min(2000, len(df))  # Cap at 2000 rows
            df = df.tail(sample_size)  # Get the last 2000 rows
            st.warning(f"Large dataset detected. Using the last {sample_size} rows for performance.")

        # Extract additional date/time-related features
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day_name'] = df['datetime'].dt.day_name()
        df['weekend'] = df['day_of_week'].isin([5, 6])

        return df

    except Exception as e:
        st.error(f"Critical error loading data: {str(e)}")
        st.info(f"Attempted to load from: {data_path}")
        return None

# Data Processing Pipeline with Validation
@st.cache_data
def preprocess_data(_df):
    """Process and enhance the raw data with validation checks"""
    if _df is None:
        return None
        
    df = _df.copy()
    
    try:
        # Datetime features with validation
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            # Check for missing values or malformed rows
            df['time'].isnull().sum(), df['time'].unique()
            df['datetime'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
            df = df.drop(columns=['time'])
            
        dt = df['datetime']
        df['date'] = dt.dt.date
        df['hour'] = dt.dt.hour
        df['day_of_week'] = dt.dt.dayofweek
        df['month'] = dt.dt.month
        df['day_name'] = dt.dt.day_name()
        df['weekend'] = df['day_of_week'].isin([5, 6])
        
        # Energy calculations with validation
        energy_cols = ['use [kW]', 'gen [kW]']
        for col in energy_cols:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return None
                
            # Convert to numeric and handle non-numeric values
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                st.warning(f"Found non-numeric values in {col}. These will be filled with 0.")
                df[col] = df[col].fillna(0)
        
        df['net_energy'] = df['use [kW]'] - df['gen [kW]']
        df['energy_ratio'] = np.where(
            df['use [kW]'] > 0,
            df['gen [kW]'] / df['use [kW]'],
            0
        )
        
        return df
        
    except Exception as e:
        st.error(f"Error during data preprocessing: {str(e)}")
        return None

# UI Components
def render_date_filter(df, key):
    """Render consistent date filter with validation"""
    try:
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        if pd.isnull(min_date) or pd.isnull(max_date):
            st.error("Invalid date range in data")
            return None
            
        return st.date_input(
            "Select date range",
            [min_date, max_date],
            key=key,
            min_value=min_date,
            max_value=max_date
        )
    except Exception as e:
        st.error(f"Error rendering date filter: {str(e)}")
        return None

def filter_data(df, date_range):
    """Filter data based on date range with validation"""
    if df is None or date_range is None or len(date_range) != 2:
        return df
        
    try:
        mask = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
        return df[mask].copy()
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        return df

def render_energy_metrics(df):
    """Display key energy metrics with validation"""
    if df is None:
        return
        
    cols = st.columns(4)
    metrics = [
        ("Total Consumption", 'use [kW]', "sum", "Total energy used"),
        ("Total Production", 'gen [kW]', "sum", "Total energy generated"),
        ("Net Energy", 'net_energy', "sum", "Net energy (use - generation)"),
        ("Self-sufficiency", None, "ratio", "Percentage of usage covered by generation")
    ]
    
    for i, (label, col, mtype, help_text) in enumerate(metrics):
        with cols[i]:
            try:
                if mtype == "sum":
                    value = df[col].sum()
                    st.metric(label, f"{value:,.0f} kW", help=help_text)
                elif mtype == "ratio":
                    ratio = (df['gen [kW]'].sum() / df['use [kW]'].sum() * 100 
                           if df['use [kW]'].sum() > 0 else 0)
                    st.metric(label, f"{ratio:.1f}%", help=help_text)
            except Exception as e:
                st.error(f"Error calculating {label}: {str(e)}")

# Dashboard Pages
def overview_dashboard(df):
    st.header("🏠 Energy Overview")
    
    if df is None:
        st.warning("No data available")
        return
        
    date_range = render_date_filter(df, "overview_date")
    filtered_df = filter_data(df, date_range)
    
    render_energy_metrics(filtered_df)
    st.markdown("---")
    
    resample_freq = st.selectbox(
        "Aggregation level",
        ["Raw", "Hourly", "Daily", "Weekly", "Monthly"],
        index=1
    )
    
    freq_map = {
        "Raw": None,
        "Hourly": "H",
        "Daily": "D",
        "Weekly": "W-MON",
        "Monthly": "M"
    }
    
    try:
        if resample_freq != "Raw":
            # Get numeric columns only for resampling
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            datetime_col = 'datetime'
            
            # Ensure we keep datetime column
            if datetime_col not in numeric_cols:
                numeric_cols.append(datetime_col)
            
            resampled = filtered_df[numeric_cols].set_index('datetime').resample(freq_map[resample_freq]).mean().reset_index()
        else:
            resampled = filtered_df
    except Exception as e:
        st.error(f"Error resampling data: {str(e)}")
        return
        
    # Rest of your function remains the same...
    tab1, tab2, tab3 = st.tabs(["📈 Energy Trends", "🌡️ Weather Impact", "📅 Temporal Patterns"])
    
    with tab1:
        try:
            fig = px.line(
                resampled,
                x='datetime',
                y=['use [kW]', 'gen [kW]', 'House overall [kW]'],
                title='Energy Flow Over Time',
                labels={'value': 'Power (kW)', 'variable': 'Type'},
                color_discrete_sequence=['#FF5733', '#33FF57', '#3377FF']
            )
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering energy trends: {str(e)}")
    


def devices_dashboard(df):
    st.header("🔌 Device-Level Analysis")
    
    if df is None:
        st.warning("No data available")
        return
        
    date_range = render_date_filter(df, "devices_date")
    filtered_df = filter_data(df, date_range)
    
    selected_devices = st.multiselect(
        "Select devices to analyze",
        DEVICE_COLUMNS,
        default=DEVICE_COLUMNS[:3]
    )
    
    if not selected_devices:
        st.warning("Please select at least one device")
        return
    
    st.markdown("---")
    
    try:
        device_totals = filtered_df[selected_devices].sum().sort_values(ascending=False)
        cols = st.columns(len(selected_devices))
        for i, (device, total) in enumerate(device_totals.items()):
            with cols[i]:
                st.metric(
                    device.replace(" [kW]", ""),
                    f"{total:,.0f} kW",
                    help=f"Total consumption for {device}"
                )
    except Exception as e:
        st.error(f"Error calculating device totals: {str(e)}")
    
    tab1, tab2, tab3 = st.tabs(["📊 Consumption Breakdown", "⏱ Usage Patterns", "🔗 Correlations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            try:
                fig = px.pie(
                    device_totals,
                    values=device_totals.values,
                    names=device_totals.index.str.replace(" [kW]", ""),
                    title='Energy Share by Device'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering pie chart: {str(e)}")
        
        with col2:
            try:
                fig = px.bar(
                    device_totals.reset_index(),
                    x='index',
                    y=0,
                    title='Total Consumption by Device',
                    labels={'index': 'Device', '0': 'Energy (kW)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering bar chart: {str(e)}")
    
    with tab2:
        try:
            fig = px.line(
                filtered_df.set_index('datetime')[selected_devices].resample('D').mean().reset_index(),
                x='datetime',
                y=selected_devices,
                title='Daily Usage Patterns',
                labels={'value': 'Power (kW)', 'datetime': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering usage patterns: {str(e)}")
    
    with tab3:
        try:
            fig = px.imshow(
                filtered_df[selected_devices].corr(),
                text_auto=True,
                aspect="auto",
                title='Device Usage Correlations',
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering correlation matrix: {str(e)}")

def weather_dashboard(df):
    st.header("🌤️ Weather Impact Analysis")
    
    if df is None:
        st.warning("No data available")
        return
        
    date_range = render_date_filter(df, "weather_date")
    filtered_df = filter_data(df, date_range)
    
    cols = st.columns(4)
    weather_stats = [
        ('temperature', '🌡️ Avg Temp', '°C'),
        ('humidity', '💧 Avg Humidity', '%'),
        ('windSpeed', '🌬️ Avg Wind Speed', ' km/h'),
        ('pressure', '⏲️ Avg Pressure', ' hPa')
    ]
    
    for i, (col, label, unit) in enumerate(weather_stats):
        with cols[i]:
            try:
                avg_value = filtered_df[col].mean()
                st.metric(label, f"{avg_value:.1f}{unit}")
            except Exception as e:
                st.error(f"Error calculating {label}: {str(e)}")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🌦 Weather Trends", "⚡ Energy Relationships"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            try:
                fig = px.line(
                    filtered_df.set_index('datetime')[['temperature', 'apparentTemperature']].resample('D').mean().reset_index(),
                    x='datetime',
                    y=['temperature', 'apparentTemperature'],
                    title='Temperature Trends',
                    labels={'value': 'Temperature (°C)', 'datetime': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering temperature trends: {str(e)}")
        
        with col2:
            try:
                fig = px.line(
                    filtered_df.set_index('datetime')[['humidity', 'dewPoint']].resample('D').mean().reset_index(),
                    x='datetime',
                    y=['humidity', 'dewPoint'],
                    title='Humidity & Dew Point',
                    labels={'value': 'Value', 'datetime': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering humidity trends: {str(e)}")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            try:
                sample_df = filtered_df.sample(min(1000, len(filtered_df)))
                fig = px.scatter(
                    sample_df,
                    x='temperature',
                    y='use [kW]',
                    color='hour',
                    trendline="lowess",
                    title='Temperature vs Consumption',
                    labels={'temperature': 'Temperature (°C)', 'use [kW]': 'Power (kW)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering temperature scatter: {str(e)}")
        
        with col2:
            try:
                fig = px.scatter(
                    filtered_df.sample(min(1000, len(filtered_df))),
                    x='humidity',
                    y='use [kW]',
                    color='temperature',
                    trendline="lowess",
                    title='Humidity vs Consumption',
                    labels={'humidity': 'Humidity (%)', 'use [kW]': 'Power (kW)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering humidity scatter: {str(e)}")

# Main App Structure
def main():
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Preprocess data
    with st.spinner("Processing data..."):
        processed_df = preprocess_data(df)
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🏠 Navigation")
        
        st.image("https://via.placeholder.com/150x50?text=Energy+Dashboard", width=150)
        
        page = st.radio(
            "Select Page",
            ["🏠 Overview", "🔌 Devices", "🌤️ Weather"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("**Data Summary**")
        
        if processed_df is not None:
            try:
                st.metric("Total Records", f"{len(processed_df):,}")
                st.metric("Date Range", 
                         f"{processed_df['date'].min().strftime('%Y-%m-%d')} to "
                         f"{processed_df['date'].max().strftime('%Y-%m-%d')}")
            except Exception as e:
                st.error(f"Error displaying data summary: {str(e)}")
        else:
            st.warning("No data loaded")
        
        st.markdown("---")
        st.markdown("**Data Export**")
        
        if processed_df is not None and st.button("Generate Sample Data"):
            try:
                sample = processed_df.sample(min(1000, len(processed_df)))
                csv = sample.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="energy_sample.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error generating sample: {str(e)}")
    
    # Page routing
    if processed_df is not None:
        if page == "🏠 Overview":
            overview_dashboard(processed_df)
        elif page == "🔌 Devices":
            devices_dashboard(processed_df)
        elif page == "🌤️ Weather":
            weather_dashboard(processed_df)
    else:
        st.error("Failed to load data. Please check your data file and try again.")

if __name__ == "__main__":
    main()