import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from datetime import datetime

# ====================== APP CONFIGURATION ======================
st.set_page_config(
    page_title="Smart Energy Dashboard",
    layout="wide",
    page_icon="🏠",
    initial_sidebar_state="expanded"
)

# ====================== CONSTANTS ======================
DEVICES = [
    'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
    'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]',
    'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]',
    'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',
    'Microwave [kW]', 'Living room [kW]', 'Solar [kW]'
]

WEATHER = [
    'temperature', 'humidity', 'windSpeed', 
    'windBearing', 'pressure', 'apparentTemperature',
    'dewPoint', 'precipProbability'
]

# ====================== DATA PROCESSING ======================
@st.cache_data
def load_data():
    """
    Load and preprocess the energy consumption data from CSV file.
    
    Returns:
        pd.DataFrame: Processed dataframe with datetime index
        None: If error occurs during loading
    """
    try:
        # Path to data file (adjust as needed)
        file_path = os.path.join("data", "HomeC.csv")
        
        # Read data
        data = pd.read_csv(file_path, low_memory=False)
        
        # Data cleaning
        data = data[:-1]  # Remove last row if it contains NaN
        
        # Convert time column - handle errors if any
        if 'time' in data.columns:
            try:
                # Try converting from Unix timestamp
                data['datetime'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
                
                # If unsuccessful, try direct conversion
                if data['datetime'].isnull().any():
                    data['datetime'] = pd.to_datetime(data['time'], errors='coerce')
                
                # Set datetime as index
                data = data.set_index('datetime')
                data = data.sort_index()
                
            except Exception as e:
                st.error(f"Time conversion error: {str(e)}")
                # Create sample timeline if needed
                data['datetime'] = pd.date_range(start='2016-01-01', periods=len(data), freq='min')
                data = data.set_index('datetime')
        
        return data.dropna()
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_daily_for_use(df, power_col='use [kW]'):
    """
    Calculate daily energy consumption.
    
    Args:
        df (pd.DataFrame): Input dataframe
        power_col (str): Column name for power usage
        
    Returns:
        pd.DataFrame: Daily energy consumption in kWh
    """
    if power_col not in df.columns:
        return pd.DataFrame()
    # Ensure we only calculate on numeric columns
    return df[[power_col]].resample('D').sum() / 60  # Convert kW to kWh

def calculate_daily_for_gen(df, power_col='gen [kW]'):
    """
    Calculate daily energy generation.
    
    Args:
        df (pd.DataFrame): Input dataframe
        power_col (str): Column name for power generation
        
    Returns:
        pd.DataFrame: Daily energy generation in kWh
    """
    if power_col not in df.columns:
        return pd.DataFrame()
    # Ensure we only calculate on numeric columns
    return df[[power_col]].resample('D').sum() / 60  # Convert kW to kWh

def calculate_hourly_for_use(df, power_col='use [kW]'):
    """
    Calculate hourly energy consumption.
    
    Args:
        df (pd.DataFrame): Input dataframe
        power_col (str): Column name for power usage
        
    Returns:
        pd.DataFrame: Hourly energy consumption in kWh
    """
    if power_col not in df.columns:
        return pd.DataFrame()
    # Sum by hour and convert from kW to kWh (power integral)
    return df[[power_col]].resample('H').sum() / 60  # kW * 1h = kWh

def calculate_hourly_for_gen(df, power_col='gen [kW]'):
    """
    Calculate hourly energy generation.
    
    Args:
        df (pd.DataFrame): Input dataframe
        power_col (str): Column name for power generation
        
    Returns:
        pd.DataFrame: Hourly energy generation in kWh
    """
    if power_col not in df.columns:
        return pd.DataFrame()
    # Sum by hour and convert from kW to kWh (power integral)
    return df[[power_col]].resample('H').sum() / 60  # kW * 1h = kWh

@st.cache_data
def process_data(_df):
    """
    Process and enrich the raw energy data with additional features.
    
    Args:
        _df (pd.DataFrame): Raw input dataframe
        
    Returns:
        pd.DataFrame: Processed dataframe with additional features
        None: If error occurs during processing
    """
    if _df is None:
        return None
        
    df = _df.copy()
    
    try:
        # Add time-based features
        df['date'] = df.index.date
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_name'] = df.index.day_name()
        df['weekend'] = df['day_of_week'].isin([5, 6])
        
        # Calculate energy metrics
        df['net_energy'] = df['use [kW]'] - df['gen [kW]']
        df['energy_ratio'] = np.where(
            df['use [kW]'] > 0,
            df['gen [kW]'] / df['use [kW]'],
            0
        )
        
        return df
        
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return None

# ====================== UI COMPONENTS ======================
def date_filter(df, key):
    """
    Display a date range filter widget.
    
    Args:
        df (pd.DataFrame): Dataframe containing date column
        key (str): Unique key for the widget
        
    Returns:
        tuple: Selected date range (start_date, end_date)
        None: If error occurs
    """
    try:
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        return st.date_input(
            "Select date range",
            [min_date, max_date],
            key=key,
            min_value=min_date,
            max_value=max_date
        )
    except Exception as e:
        st.error(f"Filter display error: {str(e)}")
        return None

def display_metrics(df):
    """
    Display key energy metrics in a 4-column layout.
    
    Args:
        df (pd.DataFrame): Processed energy data
    """
    if df is None:
        return
        
    cols = st.columns(4)
    metrics = [
        ("Total Consumption", 'use [kW]', "sum", "Total energy used"),
        ("Total Generation", 'gen [kW]', "sum", "Total energy produced"),
        ("Net Energy", 'net_energy', "sum", "Actual energy (used - produced)"),
        ("Self-sufficiency", None, "ratio", "Percentage of demand met by self-generation")
    ]
    
    for i, (name, col, metric_type, help_text) in enumerate(metrics):
        with cols[i]:
            try:
                if metric_type == "sum":
                    value = df[col].sum()
                    st.metric(name, f"{value:,.0f} kW", help=help_text)
                elif metric_type == "ratio":
                    ratio = (df['gen [kW]'].sum() / df['use [kW]'].sum() * 100 
                           if df['use [kW]'].sum() > 0 else 0)
                    st.metric(name, f"{ratio:.1f}%", help=help_text)
            except Exception as e:
                st.error(f"Error calculating {name}: {str(e)}")

# ====================== DASHBOARD PAGES ======================
def overview_page(data):
    """
    Main overview page showing energy consumption and generation metrics.
    
    Args:
        data (pd.DataFrame): Processed energy data
    """
    st.header("🏠 Energy Overview")
    
    if data is None:
        st.warning("No data available")
        return
        
    if 'use [kW]' not in data.columns:
        st.error("Column 'use [kW]' not found in data")
        st.write("Available numeric columns:", data.columns.tolist())
        return
    
    start_date = data.index.min().date()
    end_date = data.index.max().date()
    
    if start_date > end_date:
        st.error("End date must be after start date!")
        return
    
    try:
        filtered = data.loc[f"{start_date}":f"{end_date}"]
        filtered = filtered.select_dtypes(include=['number'])
        daily_energy_use = calculate_daily_for_use(filtered)
        daily_energy_gen = calculate_daily_for_gen(filtered)
        hourly_energy_use = calculate_hourly_for_use(filtered)           
        hourly_data_gen = calculate_hourly_for_gen(filtered)
        
        tab1, tab2 = st.tabs(["DAILY ENERGY CONSUMPTION & GENERATION", "ENERGY SUMMARY BY DAY"])
        
        with tab1:
            valid_dates = pd.Series(filtered.index.date).unique()
            
            if len(valid_dates) == 0:
                st.warning("No data available for selected period")
                return
                
            selected_date = st.selectbox(
                "Select date to view details",
                options=valid_dates,
                format_func=lambda x: x.strftime("%d/%m/%Y")
            )
            st.markdown("") 
            
            hourly_data = hourly_energy_use[hourly_energy_use.index.date == selected_date]

            # Calculate consumption metrics
            daily_total = hourly_data['use [kW]'].sum()
            max_hour = hourly_data['use [kW]'].idxmax()
            max_value = hourly_data['use [kW]'].max()
            avg_value = hourly_data['use [kW]'].mean()

            # Display metrics in columns
            cols = st.columns(3)
            cols[0].metric("Total Consumption", f"{daily_total:.2f} kWh")
            cols[1].metric("Peak Hour", max_hour.strftime('%H:%M'), f"{max_value:.2f} kWh")
            cols[2].metric("Hourly Average", f"{avg_value:.2f} kWh")
         
            st.markdown("---")   
            
            # Calculate generation metrics
            hourly_data_gen = hourly_data_gen[hourly_data_gen.index.date == selected_date]
            daily_total_gen = hourly_data_gen['gen [kW]'].sum()
            max_hour_gen = hourly_data_gen['gen [kW]'].idxmax()
            max_value_gen = hourly_data_gen['gen [kW]'].max()
            avg_value_gen = hourly_data_gen['gen [kW]'].mean()
            
            cols = st.columns(3)
            cols[0].metric("Total Generation", f"{daily_total_gen:.2f} kWh")
            cols[1].metric("Peak Hour", max_hour_gen.strftime('%H:%M'), f"{max_value_gen:.2f} kWh")
            cols[2].metric("Hourly Average", f"{avg_value_gen:.2f} kWh")
            
            # Plot consumption and generation together
            if not hourly_data.empty and not hourly_data_gen.empty:
                combined_data = hourly_data[['use [kW]']].join(hourly_data_gen[['gen [kW]']], how='outer').fillna(0)

                fig = px.line(
                    combined_data,
                    x=combined_data.index,
                    y=['use [kW]', 'gen [kW]'],
                    title=f"Energy Consumption & Generation - {selected_date.strftime('%d/%m/%Y')}",
                    labels={'value': 'Energy (kWh)', 'datetime': 'Time', 'variable': 'Energy Type'},
                    markers=True
                )

                fig.update_traces(line=dict(width=3), marker=dict(size=8))

                fig.update_layout(
                    xaxis_tickformat='%H:%M',
                    hovermode="x unified",
                    yaxis_title="Energy (kWh)",
                    xaxis_title="Time",
                    legend_title_text='Energy Type'
                )

                # Add annotations for peak values
                max_use_idx = combined_data['use [kW]'].idxmax()
                max_use_val = combined_data['use [kW]'].max()
                fig.add_annotation(
                    x=max_use_idx,
                    y=max_use_val,
                    text=f"Max usage: {max_use_val:.2f} kWh",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )

                max_gen_idx = combined_data['gen [kW]'].idxmax()
                max_gen_val = combined_data['gen [kW]'].max()
                fig.add_annotation(
                    x=max_gen_idx,
                    y=max_gen_val,
                    text=f"Max generation: {max_gen_val:.2f} kWh",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("No data available for selected date")

        with tab2:
            # Get available date range
            min_date = data.index.min().date()
            max_date = data.index.max().date()

            # Create date selection UI
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("From date", 
                                        min_date, 
                                        min_value=min_date, 
                                        max_value=max_date,
                                        key="start_date_selector")
            with col2:
                end_date = st.date_input("To date", 
                                        max_date, 
                                        min_value=min_date, 
                                        max_value=max_date,
                                        key="end_date_selector")

            # Validate date range
            if start_date > end_date:
                st.error("End date must be after start date!")
                st.stop()

            try:
                # Filter data by selected date range
                date_mask = (data.index.date >= start_date) & (data.index.date <= end_date)
                filtered_data = data.loc[date_mask]
                
                # Calculate daily energy
                daily_energy_use = filtered_data['use [kW]'].resample('D').sum() / 60  # Convert kW to kWh
                daily_energy_gen = filtered_data['gen [kW]'].resample('D').sum() / 60  # Convert kW to kWh
                
                # Calculate summary metrics
                total_energy_use = daily_energy_use.sum()
                total_energy_gen = daily_energy_gen.sum()
                mean_energy_use = daily_energy_use.mean()
                mean_energy_gen = daily_energy_gen.mean()
                total_days_use = len(daily_energy_use)
                total_days_gen = len(daily_energy_gen)
                
                # Display consumption metrics
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    label="TOTAL CONSUMPTION", 
                    value=f"{total_energy_use:,.2f} kWh",
                    delta=f"{total_days_use} days"
                )
                col2.metric(
                    label="DAILY AVERAGE CONSUMPTION", 
                    value=f"{mean_energy_use:,.2f} kWh"
                )
                col3.metric(
                    label="PEAK CONSUMPTION DAY", 
                    value=f"{daily_energy_use.max():.2f} kWh",
                    delta=f"Date {daily_energy_use.idxmax().strftime('%d/%m')}"
                )
                
                # Display generation metrics
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    label="TOTAL GENERATION", 
                    value=f"{total_energy_gen:,.2f} kWh",
                    delta=f"{total_days_gen} days"
                )
                col2.metric(
                    label="DAILY AVERAGE GENERATION", 
                    value=f"{mean_energy_gen:,.2f} kWh"
                )
                col3.metric(
                    label="PEAK GENERATION DAY", 
                    value=f"{daily_energy_gen.max():.2f} kWh",
                    delta=f"Date {daily_energy_gen.idxmax().strftime('%d/%m')}"
                )

                # Combine consumption and generation data
                combined_data = pd.DataFrame({
                    'use [kW]': daily_energy_use,
                    'gen [kW]': daily_energy_gen
                }).fillna(0)

                # Create grouped bar chart
                fig = px.bar(
                    combined_data,
                    x=combined_data.index,
                    y=['use [kW]', 'gen [kW]'],
                    title=f"ENERGY CONSUMPTION & GENERATION<br>From {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}",
                    labels={'value': 'kWh', 'datetime': 'Date'},
                    color_discrete_sequence=['#3498db', '#e74c3c']  # Colors for consumption and generation
                )

                # Customize chart
                fig.update_layout(
                    xaxis_tickformat='%d/%m',
                    hovermode="x unified",
                    plot_bgcolor='white',
                    height=450,
                    barmode='group'  # Bars side by side
                )

                # Format hover and bar labels
                fig.update_traces(
                    hovertemplate="<b>%{x|%d/%m/%Y}</b><br>%{y:.2f} kWh",
                    texttemplate='%{y:.1f}',
                    textposition='outside'
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
                st.stop()
                
            # Net energy section
            st.subheader("NET ENERGY (CONSUMPTION - GENERATION)")

            try:
                # Calculate daily net energy
                daily_data = filtered_data[['use [kW]', 'gen [kW]']].resample('D').sum() / 60  # Convert to kWh
                daily_data['Net [kWh]'] = daily_data['use [kW]'] - daily_data['gen [kW]']
                
                # Create line chart for net energy
                fig = px.line(
                    daily_data,
                    x=daily_data.index,
                    y='Net [kWh]',
                    title='DAILY NET ENERGY TREND',
                    labels={
                        'Net [kWh]': 'kWh',
                        'datetime': 'Date'
                    },
                    color_discrete_sequence=['#3498db'],  # Blue color
                    markers=True
                )
                
                # Customize chart
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="kWh",
                    hovermode="x unified",
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='lightgray',
                        tickformat='%d/%m'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='lightgray'
                    )
                )
                
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Data processing error: {str(e)}") 
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        st.stop()

def devices_page(df):
    """
    Page showing energy consumption analysis by device.
    
    Args:
        df (pd.DataFrame): Processed energy data
    """
    st.header("🔌 Device Analysis")
    
    if df is None:
        st.warning("No data available")
        return
        
    date_range = date_filter(df, "devices")
    filtered_df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
    
    selected_devices = st.multiselect(
        "Select devices to analyze",
        DEVICES,
        default=DEVICES[:3]
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
    
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "⏱ Trends", "🔗 Correlations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            try:
                fig = px.pie(
                    device_totals,
                    values=device_totals.values,
                    names=device_totals.index.str.replace(" [kW]", ""),
                    title='Device Consumption Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Pie chart error: {str(e)}")
        
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
                st.error(f"Bar chart error: {str(e)}")
    
    with tab2:
        try:
            fig = px.line(
                filtered_df.set_index('datetime')[selected_devices].resample('D').mean().reset_index(),
                x='datetime',
                y=selected_devices,
                title='Daily Usage Trends',
                labels={'value': 'Power (kW)', 'datetime': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Trend display error: {str(e)}")
    
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
            st.error(f"Correlation matrix error: {str(e)}")

def weather_page(df):
    """
    Page showing weather impact on energy consumption.
    
    Args:
        df (pd.DataFrame): Processed energy data with weather info
    """
    st.header("🌤️ Weather Impact")
    
    if df is None:
        st.warning("No data available")
        return
        
    date_range = date_filter(df, "weather")
    filtered_df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
    
    cols = st.columns(4)
    weather_metrics = [
        ('temperature', '🌡️ Avg Temp', '°C'),
        ('humidity', '💧 Avg Humidity', '%'),
        ('windSpeed', '🌬️ Avg Wind Speed', ' km/h'),
        ('pressure', '⏲️ Avg Pressure', ' hPa')
    ]
    
    for i, (col, name, unit) in enumerate(weather_metrics):
        with cols[i]:
            try:
                avg_value = filtered_df[col].mean()
                st.metric(name, f"{avg_value:.1f}{unit}")
            except Exception as e:
                st.error(f"Error calculating {name}: {str(e)}")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🌦 Trends", "⚡ Relationships"])
    
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
                st.error(f"Temperature chart error: {str(e)}")
        
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
                st.error(f"Humidity chart error: {str(e)}")
    
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
                st.error(f"Temperature scatter error: {str(e)}")
        
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
                st.error(f"Humidity scatter error: {str(e)}")

# ====================== MAIN APP ======================
def main():
    """
    Main application function that coordinates all components.
    """
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Process data
    with st.spinner("Processing data..."):
        processed_df = process_data(df)
    
    # Navigation sidebar
    with st.sidebar:
        st.title("🏠 Navigation")
        
        page = st.radio(
            "Select page",
            ["🏠 Overview", "🔌 Devices", "🌤️ Weather"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("**Data Summary**")
        
        if processed_df is not None:
            try:
                st.metric("Date Range", 
                         f"{processed_df['date'].min().strftime('%d/%m/%Y')} to "
                         f"{processed_df['date'].max().strftime('%d/%m/%Y')}")
            except Exception as e:
                st.error(f"Summary display error: {str(e)}")
        else:
            st.warning("No data available")
        
        st.markdown("---")

        # Data export option
        if processed_df is not None and st.button("Create Data Sample"):
            try:
                sample = processed_df.sample(min(1000, len(processed_df)))
                csv = sample.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="data_sample.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Sample creation error: {str(e)}")
    
    # Page routing
    if processed_df is not None:
        if page == "🏠 Overview":
            overview_page(processed_df)
        elif page == "🔌 Devices":
            devices_page(processed_df)
        elif page == "🌤️ Weather":
            weather_page(processed_df)
    else:
        st.error("Failed to load data. Please check data file and try again.")

if __name__ == "__main__":
    main()