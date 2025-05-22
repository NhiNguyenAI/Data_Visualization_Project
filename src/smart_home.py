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

def calculate_daily_for_device(df, device_col):
    """
    Calculate daily energy consumption for a specific device.
    
    Args:
        df (pd.DataFrame): Input dataframe with datetime index
        device_col (str): Column name for the device power usage
        
    Returns:
        pd.DataFrame: Daily energy consumption in kWh for the device
    """
    if device_col not in df.columns:
        return pd.DataFrame()
    
    # Ensure we only calculate on numeric columns
    if not np.issubdtype(df[device_col].dtype, np.number):
        return pd.DataFrame()
    
    # Resample to daily sum and convert from kW to kWh
    return df[[device_col]].resample('D').sum() / 60

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
    Display a date range filter widget with calendar.
    
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
            max_value=max_date,
            format="DD/MM/YYYY",
            help="Click to open calendar picker"
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
    st.header("🏠 Energy Overview")
    
    if data is None:
        st.warning("No data available")
        return
        
    if 'use [kW]' not in data.columns:
        st.error("Column 'use [kW]' not found in data")
        return
    
    # ========== HEATMAP SECTION ==========
    st.subheader("🌡️ Hourly Energy Consumption Patterns")
    
    # Prepare heatmap data
    heatmap_data = data.copy()
    heatmap_data['hour'] = heatmap_data.index.hour
    heatmap_data['date'] = heatmap_data.index.date
    heatmap_data['day_month'] = heatmap_data.index.strftime('%d/%m')  # Date format DD/MM
    
    # Get unique dates and limit to 30 most recent if needed
    unique_dates = sorted(heatmap_data['date'].unique(), reverse=True)
    display_dates = unique_dates[:30]  # Take most recent 30 days
    heatmap_filtered = heatmap_data[heatmap_data['date'].isin(display_dates)]
    
    # Create pivot table - using day_month for display while keeping date for sorting
    heatmap_pivot = heatmap_filtered.pivot_table(
        index=['date', 'day_month'],  # Store both for sorting and display
        columns='hour',
        values='use [kW]',
        aggfunc='mean'
    )
    
    # Sort by date (newest first) but display day_month
    heatmap_pivot = heatmap_pivot.sort_index(level='date', ascending=False)
    
    # Create heatmap with proper date display
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour of Day", y="Date", color="Energy Usage (kW)"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index.get_level_values('day_month'),  # Use formatted dates for display
        color_continuous_scale=[
        "#fff7bc",  
        "#fee391",
        "#fec44f",
        "#fe9929",
        "#ec7014",
        "#cc4c02",
        "#993404",
        "#662506"   
        ],
        aspect="auto",
        height=600 + (len(heatmap_pivot) * 3)
    )
    
    # Customize heatmap appearance
    fig_heatmap.update_layout(
        title={
            'text': "<b>Daily Energy Consumption Patterns by Hour</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18}
        },
        xaxis_title="<b>Hour of Day</b>",
        yaxis_title="<b>Date</b>",
        margin=dict(l=50, r=50, b=100, t=100),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12)
    )
    
    # Configure axes
    fig_heatmap.update_xaxes(
        tickmode='array',
        tickvals=list(range(0, 24, 2)),
        ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)',
        tickfont=dict(size=10)
    )
    
    fig_heatmap.update_yaxes(
        tickmode='array',
        tickvals=list(range(len(heatmap_pivot))),
        ticktext=heatmap_pivot.index.get_level_values('day_month'),
        tickfont=dict(size=10),
        automargin=True
    )
    
    # Display the heatmap
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ========== REST OF OVERVIEW PAGE ==========
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
            st.markdown("") 
            valid_dates = pd.Series(filtered.index.date).unique()
            
            if len(valid_dates) == 0:
                st.warning("No data available for selected period")
                return
                
            selected_date = st.date_input(
                "Select date to view details",
                valid_dates[0],
                min_value=valid_dates.min(),
                max_value=valid_dates.max(),
                format="DD/MM/YYYY",
                key="daily_detail_date"
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
         
            st.markdown("")   
            
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
            st.markdown("") 
            # Get available date range
            min_date = data.index.min().date()
            max_date = data.index.max().date()

            # Create date selection UI with calendar
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "From date", 
                    min_date, 
                    min_value=min_date, 
                    max_value=max_date,
                    key="start_date_selector_calendar",
                    format="DD/MM/YYYY"
                )
            with col2:
                end_date = st.date_input(
                    "To date", 
                    max_date, 
                    min_value=min_date, 
                    max_value=max_date,
                    key="end_date_selector_calendar",
                    format="DD/MM/YYYY"
                )

            # Validate date range
            if start_date > end_date:
                st.error("End date must be after start date!")
                st.stop()
            st.markdown("")

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
                    value=f"{total_energy_use:,.2f} kWh"
                )
                col2.metric(
                    label="PEAK CONSUMPTION DAY", 
                    value=f"{daily_energy_use.max():.2f} kWh",
                    delta=f"Date {daily_energy_use.idxmax().strftime('%d/%m')}"
                )
                col3.metric(
                    label="DAILY AVERAGE CONSUMPTION", 
                    value=f"{mean_energy_use:,.2f} kWh"
                )

                st.markdown("")
                
                # Display generation metrics
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    label="TOTAL GENERATION", 
                    value=f"{total_energy_gen:,.2f} kWh",
                )
                col2.metric(
                    label="PEAK GENERATION DAY", 
                    value=f"{daily_energy_gen.max():.2f} kWh",
                    delta=f"Date {daily_energy_gen.idxmax().strftime('%d/%m')}"
                )
                col3.metric(
                    label="DAILY AVERAGE GENERATION", 
                    value=f"{mean_energy_gen:,.2f} kWh"
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
    """
    st.header("🔌 Device Analysis")
    
    if df is None:
        st.warning("No data available")
        return
        
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "⏱ Trends", "🔗 Correlations"])
    
    with tab1:
        # Get available date range
        min_date = df.index.min().date()
        max_date = df.index.max().date()

        # Create date selection UI with calendar
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From date", 
                min_date, 
                min_value=min_date, 
                max_value=max_date,
                key="device_start_date_calendar",
                format="DD/MM/YYYY"
            )
        with col2:
            end_date = st.date_input(
                "To date", 
                max_date, 
                min_value=min_date, 
                max_value=max_date,
                key="device_end_date_calendar",
                format="DD/MM/YYYY"
            )

        # Validate date range
        if start_date > end_date:
            st.error("End date must be after start date!")
            st.stop()
        st.markdown("")

        try:
            # Filter data by selected date range
            date_mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            filtered_data = df.loc[date_mask]
            
            # Device selection
            selected_device = st.selectbox(
                "Select a device to analyze",
                DEVICES,
                index=0,
                key="device_select"
            )
            
            if not selected_device or selected_device not in filtered_data.columns:
                st.warning("Please select a valid device")
                return
            
            # Calculate daily consumption
            daily_consumption = filtered_data[[selected_device]].resample('D').sum() / 60  # Convert kW to kWh
            
            if daily_consumption.empty:
                st.warning(f"No data available for {selected_device} in selected period")
                return
                
            # Display metrics in columns
            cols = st.columns(3)
            cols[0].metric(
                "Total Consumption", 
                f"{daily_consumption[selected_device].sum():,.2f} kWh"
            )
            cols[1].metric(
                "Peak Consumption Day", 
                f"{daily_consumption[selected_device].max():,.2f} kWh",
                delta=f"Date {daily_consumption[selected_device].idxmax().strftime('%d/%m')}"
            )
            cols[2].metric(
                "Daily Average", 
                f"{daily_consumption[selected_device].mean():,.2f} kWh"
            )

            
           # Plot device consumption with peak annotations
            if not daily_consumption.empty:
                fig = px.line(
                    daily_consumption.reset_index(),
                    x='datetime',
                    y=selected_device,
                    title=f"{selected_device.replace(' [kW]', '')} Daily Consumption",
                    labels={
                        'datetime': 'Date',
                        selected_device: 'Consumption (kWh)'
                    },
                    markers=True
                )

                fig.update_traces(
                    line=dict(width=3, color='#1f77b4'),
                    marker=dict(size=8, color='#1f77b4')
                )

                fig.update_layout(
                    xaxis_tickformat='%d/%m',
                    hovermode="x unified",
                    yaxis_title="Consumption (kWh)",
                    xaxis_title="Date",
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=500
                )

                # Add annotations for peak values
                max_consump_idx = daily_consumption[selected_device].idxmax()
                max_consump_val = daily_consumption[selected_device].max()
                
                fig.add_annotation(
                    x=max_consump_idx,
                    y=max_consump_val,
                    text=f"Peak: {max_consump_val:.2f} kWh",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                    bgcolor="white"
                )

                # Add annotation for minimum value if you want
                min_consump_idx = daily_consumption[selected_device].idxmin()
                min_consump_val = daily_consumption[selected_device].min()
                
                fig.add_annotation(
                    x=min_consump_idx,
                    y=min_consump_val,
                    text=f"Min: {min_consump_val:.2f} kWh",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=40,
                    bgcolor="white"
                )

                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying device data: {str(e)}")
    with tab2:
        # Get date range from data
        min_date = df.index.min().date()
        max_date = df.index.max().date()

        # Create date selection UI with calendar
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From date", 
                min_date, 
                min_value=min_date, 
                max_value=max_date,
                key="devices_tab2_start",
                format="DD/MM/YYYY"
            )
        with col2:
            end_date = st.date_input(
                "To date", 
                max_date, 
                min_value=min_date, 
                max_value=max_date,
                key="devices_tab2_end",
                format="DD/MM/YYYY"
            )

        # Validate date range
        if start_date > end_date:
            st.error("End date must be after start date!")
            st.stop()
        
        # Filter data by selected date range
        date_mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        filtered_df = df.loc[date_mask]
        
        # Device selection
        selected_devices = st.multiselect(
            "Select devices to analyze (or leave empty for all devices)",
            DEVICES,
            default=[],
            key="devices_tab2_select"
        )
        
        # Use all devices if none selected
        devices_to_analyze = selected_devices if selected_devices else DEVICES
        
        try:
            # Calculate total consumption for each device
            device_totals = filtered_df[devices_to_analyze].sum().sort_values(ascending=False)
            
            # Calculate percentage of total consumption
            total_consumption = device_totals.sum()
            device_percentages = (device_totals / total_consumption * 100).round(1)
            
            # Get top 5 devices and group the rest into "Other"
            top_devices = device_percentages.head(5)
            other_devices = device_percentages[5:]
            
            if len(other_devices) > 0:
                other_percentage = other_devices.sum()
                top_devices['Other'] = other_percentage
            
            # Prepare data for visualization
            pie_data = top_devices.reset_index()
            pie_data.columns = ['Device', 'Percentage']
            pie_data['Device'] = pie_data['Device'].str.replace(" [kW]", "")
            
            # Display metrics in columns - top 5 devices
            st.subheader("Top 5 Energy-Consuming Devices")
            cols = st.columns(5)
            for i in range(min(5, len(device_totals))):
                device = device_totals.index[i]
                with cols[i]:
                    st.metric(
                        label=device.replace(" [kW]", ""),
                        value=f"{device_totals.iloc[i]:,.0f} kW",
                        delta=f"{device_percentages.iloc[i]:.1f}% of total"
                    )
            
            # Create two-column layout for charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart showing consumption distribution
                fig_pie = px.pie(
                    pie_data,
                    values='Percentage',
                    names='Device',
                    title='Energy Consumption Distribution (Top 5 + Other)',
                    hover_data=['Percentage'],
                    labels={'Percentage': '% of total'},
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate="<b>%{label}</b><br>%{percent:.1%}<br>%{value:.1f}%",
                    marker=dict(line=dict(color='#FFFFFF', width=1))
                )
                fig_pie.update_layout(
                    uniformtext_minsize=10,
                    uniformtext_mode='hide',
                    showlegend=False,
                    margin=dict(t=50, b=50)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar chart showing absolute consumption for all devices
                devive_totals_top_5 = device_totals.head(5).reset_index()
                devive_totals_top_5.columns = ['Device', 'Consumption']
                devive_totals_top_5['Device'] = devive_totals_top_5['Device'].str.replace(" [kW]", "")
                
                fig_bar = px.bar(
                    devive_totals_top_5,
                    x='Device',
                    y='Consumption',
                    title='Detailed Consumption by Device',
                    labels={'Consumption': 'Energy (kW)'},
                    color='Consumption',
                    color_continuous_scale='RdBu'
                )
                fig_bar.update_traces(
                    hovertemplate="<b>%{x}</b><br>%{y:,.0f} kW"
                )
                fig_bar.update_layout(
                    xaxis_title="Device",
                    yaxis_title="Total Consumption (kW)",
                    xaxis={'categoryorder':'total descending'},
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing device data: {str(e)}")

def weather_page(df):
    """
    Enhanced page showing weather impact on energy consumption with more insights.
    
    Args:
        df (pd.DataFrame): Processed energy data with weather info
    """
    st.header("🌤 Weather Impact Analysis")
    
    if df is None:
        st.warning("No data available")
        return
    
    # Ensure 'date' column exists
    if 'date' not in df.columns:
        st.error("Dataframe is missing 'date' column")
        return
    
    # Date filtering
    date_range = date_filter(df, "weather")
    if date_range is None or len(date_range) < 2:
        st.error("Invalid date range selected")
        return
        
    filtered_df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])].copy()
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    
    # Separate numeric and non-numeric columns
    numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols = [col for col in filtered_df.columns if col not in numeric_cols and col != 'date']
    
    # Create hourly aggregates - handle numeric and non-numeric separately
    if len(numeric_cols) > 0:
        # Resample numeric columns with mean
        hourly_numeric = filtered_df.set_index('date')[numeric_cols].resample('H').mean()
        
        # Resample non-numeric columns with first observation if they exist
        if non_numeric_cols:
            hourly_non_numeric = filtered_df.set_index('date')[non_numeric_cols].resample('H').first()
            hourly_df = pd.concat([hourly_numeric, hourly_non_numeric], axis=1)
        else:
            hourly_df = hourly_numeric
        
        hourly_df = hourly_df.reset_index()
    else:
        hourly_df = filtered_df.copy()
    
    # Weather summary metrics
    st.subheader("Weather Summary")
    cols = st.columns(5)
    weather_metrics = [
        ('temperature', '🌡 Avg Temp', '°C'),
        ('apparentTemperature', '🌡 Feels Like', '°C'),
        ('humidity', '💧 Humidity', '%'),
        ('windSpeed', '🌬 Wind Speed', 'km/h'),
        ('pressure', '⏲ Pressure', 'hPa')
    ]
    
    # Filter to only include valid numeric metrics
    valid_metrics = [(col, name, unit) for col, name, unit in weather_metrics 
                    if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col])]
    
    for i, (col, name, unit) in enumerate(valid_metrics):
        with cols[i]:
            try:
                avg_value = filtered_df[col].mean()
                delta = f"{filtered_df[col].max() - filtered_df[col].min():.1f}{unit} range"
                st.metric(name, f"{avg_value:.1f}{unit}", delta)
            except Exception as e:
                st.error(f"Error calculating {name}: {str(e)}")
    
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Weather Trends", "🔍 Energy Relationships", "🌦 Weather Conditions"])
    
    with tab1:
            col1, col2 = st.columns(2)
            with col1:
                # Get only numeric columns that exist in the dataframe
                available_numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
                temp_cols = [c for c in ['temperature', 'apparentTemperature', 'dewPoint'] 
                            if c in available_numeric_cols]
                
                if temp_cols:
                    hourly = filtered_df.groupby('hour')[temp_cols].mean().reset_index()
                    fig = px.bar(
                        hourly.melt(id_vars='hour'), 
                        x='hour', 
                        y='value',
                        color='variable',
                        barmode='group',
                        title='Hourly Temperature Averages',
                        labels={'hour': 'Hour of Day', 'value': 'Temperature (°C)'},
                        color_discrete_map={
                            'temperature': '#EF553B',
                            'apparentTemperature': '#AB63FA',
                            'dewPoint': '#00CC96'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric temperature data available for plotting")
                    st.write("Available numeric columns:", available_numeric_cols)
            
            with col2:
                if all(c in filtered_df.columns for c in ['hour', 'temperature', 'use [kW]']):
                    heatmap_df = filtered_df.copy()
                    # Ensure we're using numeric columns only
                    heatmap_df = heatmap_df[['hour', 'temperature', 'use [kW]']].dropna()
                    
                    fig = px.density_heatmap(
                        heatmap_df,
                        x='hour',
                        y='temperature',
                        z='use [kW]',
                        histfunc='avg',
                        title='Temperature by Hour with Energy Usage',
                        labels={'hour': 'Hour of Day', 'temperature': 'Temperature (°C)'},
                        nbinsx=24, 
                        nbinsy=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Required columns missing for heatmap")
    
    with tab2:
        st.subheader("Energy Consumption Relationships")
        
        with st.expander("Temperature Impact", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                try:
                    # Create temperature bins and calculate mean consumption
                    temp_bins = pd.cut(filtered_df['temperature'], bins=10)
                    temp_group = filtered_df.groupby(temp_bins)['use [kW]'].mean().reset_index()
                    
                    # Convert intervals to readable strings
                    temp_group['temp_range'] = temp_group['temperature'].apply(
                        lambda x: f"{x.left:.1f}°C to {x.right:.1f}°C"
                    )
                    
                    fig = px.bar(
                        temp_group,
                        x='temp_range',
                        y='use [kW]',
                        title='Average Consumption by Temperature Range',
                        labels={'use [kW]': 'Avg Power (kW)', 'temp_range': 'Temperature Range'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Could not create temperature bins chart: {str(e)}")
                
            with col2:
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
        
        # Weather condition impact
        with st.expander("Weather Condition Impact"):
            if 'icon' in filtered_df.columns:
                fig = px.box(
                    filtered_df,
                    x='icon',
                    y='use [kW]',
                    color='icon',
                    title='Energy Consumption by Weather Condition',
                    labels={'use [kW]': 'Power (kW)', 'icon': 'Weather Condition'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Weather condition data not available")
    
    with tab3:
        st.subheader("Detailed Weather Conditions")
        
        # Wind analysis
        with st.expander("Wind Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(
                    hourly_df,
                    x='date',
                    y='windSpeed',
                    title='Wind Speed Over Time',
                    labels={'windSpeed': 'Wind Speed (km/h)', 'date': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'windBearing' in filtered_df.columns:
                    fig = px.scatter_polar(
                        filtered_df,
                        r='windSpeed',
                        theta='windBearing',
                        color='temperature',
                        title='Wind Direction and Speed',
                        labels={'windSpeed': 'Wind Speed (km/h)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Precipitation analysis
        with st.expander("Precipitation Analysis"):
            if 'precipIntensity' in filtered_df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(
                        hourly_df,
                        x='date',
                        y='precipIntensity',
                        title='Precipitation Intensity',
                        labels={'precipIntensity': 'Intensity (mm/h)', 'date': 'Date'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        filtered_df[filtered_df['precipIntensity'] > 0],
                        x='precipIntensity',
                        y='use [kW]',
                        color='temperature',
                        title='Energy Use During Precipitation',
                        labels={'precipIntensity': 'Precipitation (mm/h)', 'use [kW]': 'Power (kW)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Precipitation data not available")

# ====================== MAIN APP ======================
def main():
    with st.spinner("Loading data..."):
        df = load_data()
    
    with st.spinner("Processing data..."):
        processed_df = process_data(df)
    
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