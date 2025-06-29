import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime, timedelta
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import xgboost as xgb
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
import traceback

# ====================== APP CONFIGURATION ======================
st.set_page_config(
    page_title="Smart Energy Dashboard",
    layout="wide",
    page_icon="🏠",
    initial_sidebar_state="expanded"
)
# ====================== CUSTOM CSS =============================
st.markdown("""
    <style>
    .block-container {
        padding: 2rem 3rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    @media screen and (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ====================== CONSTANTS ======================
DEVICES = [
    'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]',
    'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]',
    'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]',
    'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',
    'Microwave [kW]', 'Living room [kW]', 'Solar [kW]'
]
NON_SOLAR_DEVICES = [device for device in DEVICES if device != 'Solar [kW]']
WEATHER_COLS = [
    'temperature', 'humidity', 'windSpeed',
    'windBearing', 'pressure', 'apparentTemperature',
    'dewPoint', 'precipProbability'
]

# ====================== DATA PROCESSING ======================
@st.cache_data
def load_data():
    """
    Load and preprocess the energy data assuming 1-minute resolution
    and override incorrect timestamps with correct time index.
    """
    try:
        file_path = os.path.join("data", "HomeC.csv")
        df = pd.read_csv(file_path, low_memory=False)
        df = df[:-1]  # optional: remove last row if invalid

        # Override timestamp with correct 1-minute spacing
        df['time'] = pd.date_range(start='2016-01-01 05:00', periods=len(df), freq='min')

        # Set datetime index
        df = df.set_index('time')

        # Optional: extract time features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['weekday'] = df.index.day_name()
        df['weekofyear'] = df.index.isocalendar().week
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute

        return df

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

def calculate_hourly_for_device(df, device_col):
    """
    Calculate hourly energy consumption for a specific device.
    
    Args:
        df (pd.DataFrame): Input dataframe with datetime index
        device_col (str): Column name for the device power usage
        
    Returns:
        pd.DataFrame: Hourly energy consumption in kWh for the device
    """
    if device_col not in df.columns:
        return pd.DataFrame()
    
    # Ensure we only calculate on numeric columns
    if not np.issubdtype(df[device_col].dtype, np.number):
        return pd.DataFrame()
    
    # Resample to hourly sum and convert from kW to kWh
    return df[[device_col]].resample('H').sum() / 60  # kW * 1h = kWh

def calculate_monthly_for_device(df, device_col):
    """
    Calculate monthly energy consumption for a specific device.
    
    Args:
        df (pd.DataFrame): Input dataframe with datetime index
        device_col (str): Column name for the device power usage
        
    Returns:
        pd.DataFrame: Monthly energy consumption in kWh for the device
    """
    if device_col not in df.columns:
        return pd.DataFrame()
        
    # Ensure we only calculate on numeric columns
    if not np.issubdtype(df[device_col].dtype, np.number):
        return pd.DataFrame()
        
    # Try different frequency strings for pandas compatibility
    try:
        # For pandas 2.2+ use 'M' for month-end
        return df[[device_col]].resample('M').sum() / 60  # kW to kWh
    except ValueError:
        try:
            # For older pandas versions, try 'ME'
            return df[[device_col]].resample('ME').sum() / 60  # kW to kWh
        except ValueError:
            try:
                # Alternative: use 'MS' for month-start
                return df[[device_col]].resample('MS').sum() / 60  # kW to kWh
            except ValueError:
                # Fallback: manual monthly grouping
                monthly_data = df[[device_col]].groupby([
                    df.index.year, 
                    df.index.month
                ]).sum() / 60
                
                # Create proper datetime index for monthly data
                monthly_index = pd.to_datetime([
                    f"{year}-{month:02d}-01" 
                    for year, month in monthly_data.index
                ])
                monthly_data.index = monthly_index
                return monthly_data

def plot_temp_bins(df, power_col):
    bins = pd.cut(df['temperature'], bins=10)
    grouped = df.groupby(bins)[power_col].mean().reset_index()
    grouped['Range'] = grouped['temperature'].apply(lambda x: f"{x.left:.1f}°C to {x.right:.1f}°C")
    fig = px.bar(grouped, x='Range', y=power_col, title='Avg Consumption by Temperature Range')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def scatter_temp_vs_power(df, power_col):
    sample = df.sample(min(1000, len(df)))
    fig = px.scatter(sample, x='temperature', y=power_col, color='hour',
                     trendline='ols', title='Temperature vs Consumption')
    st.plotly_chart(fig, use_container_width=True)

def metric_box(title, value, subtitle=None, color="#1f77b4"):
    st.markdown(f"""
        <div style="padding:10px;border-radius:12px;background:{color};color:white;text-align:center;margin-bottom:10px">
            <div style="font-size:14px;">{title}</div>
            <div style="font-size:24px;font-weight:bold;">{value}</div>
            <div style="font-size:12px;">{subtitle if subtitle else ""}</div>
        </div>
    """, unsafe_allow_html=True)

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
        # Remove invalid rows
        df['cloudCover'].replace(['cloudCover'], method='bfill', inplace=True)
        df['cloudCover'] = df['cloudCover'].astype('float')

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
        
        # ========== WEEKLY HEATMAP ==========
        st.subheader("Weekly Energy Consumption Heatmap (kWh/day)")

        # Calculate daily energy consumption in kWh
        daily_kwh = data['use [kW]'].resample('D').sum() / 60  # Convert kW to kWh
        daily_kwh = daily_kwh.to_frame(name='kWh')

        # Extract time features
        daily_kwh['week'] = daily_kwh.index.isocalendar().week
        daily_kwh['weekday'] = daily_kwh.index.weekday
        daily_kwh['year'] = daily_kwh.index.year
        daily_kwh['month'] = daily_kwh.index.month
        daily_kwh['date'] = daily_kwh.index

        # Year selection
        selected_year = st.selectbox("Select Year", sorted(daily_kwh['year'].unique(), reverse=True))
        yearly_data = daily_kwh[daily_kwh['year'] == selected_year]

        # Function to determine the dominant month for a week
        def get_week_month(week, year):
            # Find the first day of the given week
            first_day = pd.to_datetime(f"{year}-W{week}-1", format="%Y-W%W-%w")
            # Get all days in the week
            week_days = pd.date_range(start=first_day, periods=7, freq='D')
            # Count days per month
            month_counts = week_days.month.value_counts()
            # Return the month with the most days
            dominant_month = month_counts.idxmax()
            month_name = pd.to_datetime(f"{year}-{dominant_month}-01").strftime("%B")
            return month_name

        # Function to get the specific date for a week and weekday
        def get_specific_date(week, weekday, year):
            try:
                # Find the first day (Monday) of the given week
                first_day = pd.to_datetime(f"{year}-W{week}-1", format="%Y-W%W-%w")
                # Add the weekday offset (0=Monday, ..., 6=Sunday)
                specific_date = first_day + pd.Timedelta(days=weekday)
                return specific_date.strftime("%d/%m/%Y")
            except ValueError:
                return "Unknown"

        # Assign month labels to weeks
        yearly_data['month_label'] = yearly_data['week'].apply(lambda w: get_week_month(w, selected_year))

        # Pivot table for heatmap: weekdays as rows, weeks as columns
        heatmap_pivot = yearly_data.pivot_table(
            index='weekday',
            columns='week',
            values='kWh',
            aggfunc='mean'
        )

        # Rename index to full English weekday names
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot.index = weekday_names

        # Ensure all weeks (1-53) are present, filling missing weeks with NaN
        all_weeks = pd.DataFrame(columns=range(1, 54))
        heatmap_pivot = heatmap_pivot.reindex(columns=all_weeks.columns, fill_value=np.nan)

        # Create custom x-axis tick labels (month names)
        week_to_month = yearly_data.groupby('week')['month_label'].first().reindex(range(1, 54), fill_value='Unknown')
        month_labels = week_to_month.values
        # Only show month names when they change for clarity
        display_labels = []
        last_month = None
        for i, month in enumerate(month_labels):
            if month != last_month and month != 'Unknown':
                display_labels.append(month)
                last_month = month
            else:
                display_labels.append('')

        # Create custom hover data with specific dates
        hover_data = np.empty_like(heatmap_pivot, dtype=object)
        for i, weekday in enumerate(range(7)):  # 0=Monday, ..., 6=Sunday
            for j, week in enumerate(range(1, 54)):
                date_str = get_specific_date(week, weekday, selected_year)
                kwh = heatmap_pivot.iloc[i, j] if not pd.isna(heatmap_pivot.iloc[i, j]) else "No data"
                hover_data[i, j] = f"Date: {date_str}<br>Week: {week}<br>Day: {weekday_names[i]}<br>Energy: {kwh:.2f} kWh" if kwh != "No data" else f"Date: {date_str}<br>Week: {week}<br>Day: {weekday_names[i]}<br>Energy: No data"

        # Energy range slider for heatmap
        kwh_min = yearly_data['kWh'].min() if not yearly_data['kWh'].empty else 0
        kwh_max = yearly_data['kWh'].max() if not yearly_data['kWh'].empty else 1
        slider_kwh_range = st.slider(
            "Select energy consumption range (kWh/day) to filter",
            min_value=float(kwh_min),
            max_value=float(kwh_max),
            value=(float(kwh_min), float(kwh_max)),
            step=0.01,
            format="%.2f",
            key="heatmap_kwh_slider"
        )

        # Extract slider min and max kWh values
        slider_kwh_min, slider_kwh_max = slider_kwh_range

        # Filter heatmap data for slider kWh range
        filtered_heatmap_pivot = heatmap_pivot.copy()
        filtered_heatmap_pivot[(filtered_heatmap_pivot < slider_kwh_min) | (filtered_heatmap_pivot > slider_kwh_max)] = np.nan

        # Create heatmap with filtered data
        fig = px.imshow(
            filtered_heatmap_pivot,
            labels=dict(x="Week (by Month)", y="Day of Week", color="Energy (kWh)"),
            color_continuous_scale=[
                [0.0, "#138413"],  # Low energy: green
                [0.3, "#F9FBF9"],  # Mid: light gray
                [0.6, "#FF9800"],  # High: orange
                [1.0, "#E53935"]   # Very high: red
            ],
            aspect="auto"
        )

        # Update layout for balanced cells and better appearance
        fig.update_layout(
            title=f"<b>Weekly Energy Consumption - {selected_year}</b>",
            xaxis_title="<b>Week (by Month)</b>",
            yaxis_title="<b>Day of Week</b>",
            height=500,  # Adjusted height for 7 rows (days)
            width=1200,  # Wider width for 53 columns (weeks)
            margin=dict(l=100, r=80, b=80, t=80),
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(53)),
                ticktext=display_labels,
                tickangle=45,
                side="top",
                showgrid=False,
                title_standoff=10
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(7)),
                ticktext=weekday_names,
                autorange="reversed",  # Monday at the top
                showgrid=False,
                title_standoff=10
            ),
            coloraxis_colorbar=dict(
                title="kWh",
                thickness=15,
                len=0.5,
                yanchor="middle",
                y=0.5
            )
        )

        # Ensure balanced cell sizes and add custom hover data
        fig.update_traces(
            xgap=2,  # Small gap between cells for clarity
            ygap=2,
            zmin=slider_kwh_min,  # Set color scale to slider range
            zmax=slider_kwh_max,
            customdata=hover_data,
            hovertemplate="%{customdata}<extra></extra>"
        )

        # Display the heatmap
        st.plotly_chart(fig, use_container_width=True)

        # Expander with instructions
        with st.expander("How to read this heatmap"):
            st.markdown("""
            - **Columns**: Weeks of the year, labeled by the dominant month (e.g., January, February).
            - **Rows**: Days of the week (Monday to Sunday).
            - **Color intensity**: Daily energy consumption in kWh, filtered by the selected energy range.
            - **Darker colors**: Higher energy consumption within the range.
            - Hover over cells to see the specific date, week number, day, and energy value.
            - Missing or filtered-out data (outside the energy range) appears as empty cells.
            - Only the first week of each month is labeled for clarity.
            - Use the slider to filter the energy consumption range displayed.
            """)

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
            
            # Filter hourly data for the selected date
            hourly_data = hourly_energy_use[hourly_energy_use.index.date == selected_date]
            hourly_data_gen = hourly_data_gen[hourly_data_gen.index.date == selected_date]

            if hourly_data.empty or hourly_data_gen.empty:
                st.warning("No data available for selected date")
                return

            # Calculate consumption metrics
            total_energy_use = hourly_data['use [kW]'].sum()
            peak_energy_use = hourly_data['use [kW]'].max()
            peak_hour_use = hourly_data['use [kW]'].idxmax().strftime('%H:%M') if not hourly_data['use [kW]'].empty else "N/A"
            mean_energy_use = hourly_data['use [kW]'].mean()

            # Display consumption metrics
            col1, col2, col3 = st.columns(3)
            col1.metric(
                label="TOTAL CONSUMPTION",
                value=f"{total_energy_use:,.2f} kWh"
            )
            col2.metric(
                label="PEAK CONSUMPTION HOUR",
                value=f"{peak_energy_use:.2f} kWh",
                delta=f"Hour {peak_hour_use}"
            )
            col3.metric(
                label="HOURLY AVERAGE CONSUMPTION",
                value=f"{mean_energy_use:,.2f} kWh"
            )

            st.markdown("")

            # Calculate generation metrics
            total_energy_gen = hourly_data_gen['gen [kW]'].sum()
            peak_energy_gen = hourly_data_gen['gen [kW]'].max()
            peak_hour_gen = hourly_data_gen['gen [kW]'].idxmax().strftime('%H:%M') if not hourly_data_gen['gen [kW]'].empty else "N/A"
            mean_energy_gen = hourly_data_gen['gen [kW]'].mean()

            # Display generation metrics
            col1, col2, col3 = st.columns(3)
            col1.metric(
                label="TOTAL GENERATION",
                value=f"{total_energy_gen:,.2f} kWh"
            )
            col2.metric(
                label="PEAK GENERATION HOUR",
                value=f"{peak_energy_gen:.2f} kWh",
                delta=f"Hour {peak_hour_gen}"
            )
            col3.metric(
                label="HOURLY AVERAGE GENERATION",
                value=f"{mean_energy_gen:,.2f} kWh"
            )

            # Combine consumption and generation data
            combined_data = pd.DataFrame({
                'use [kW]': hourly_data['use [kW]'],
                'gen [kW]': hourly_data_gen['gen [kW]']
            }).fillna(0)

            # Hourly range slider for both charts
            slider_hour_range = st.slider(
                "Select hour range to zoom",
                min_value=0,
                max_value=23,
                value=(0, 23),
                step=1,
                format="%d:00",
                key="tab1_hour_slider"
            )

            # Extract slider start and end hours
            slider_start_hour, slider_end_hour = slider_hour_range

            # Filter combined data for slider hour range
            slider_mask = (combined_data.index.hour >= slider_start_hour) & (combined_data.index.hour <= slider_end_hour)
            slider_combined_data = combined_data[slider_mask]

            # Create grouped bar chart with slider range
            if slider_combined_data.empty:
                st.warning("No data available for selected hour range for bar chart")
            else:
                fig = px.bar(
                    slider_combined_data,
                    x=slider_combined_data.index,
                    y=['use [kW]', 'gen [kW]'],
                    title=f"HOURLY ENERGY CONSUMPTION & GENERATION - {selected_date.strftime('%d/%m/%Y')} ({slider_start_hour:02d}:00 to {slider_end_hour:02d}:00)",
                    labels={'value': 'kWh', 'datetime': 'Hour'},
                    color_discrete_sequence=['#3498db', '#e74c3c']
                )

                fig.update_layout(
                    xaxis_tickformat='%H:%M',
                    hovermode="x unified",
                    plot_bgcolor='white',
                    height=450,
                    barmode='group',
                    xaxis_title="Hour",
                    yaxis_title="kWh"
                )

                fig.update_traces(
                    hovertemplate="<b>%{x|%H:%M}</b><br>%{y:.2f} kWh",
                    texttemplate='%{y:.1f}',
                    textposition='outside'
                )

                st.plotly_chart(fig, use_container_width=True)

            # Net energy section
            st.subheader("NET ENERGY (CONSUMPTION - GENERATION)")

            try:
                # Calculate hourly net energy
                hourly_net_data = pd.DataFrame({
                    'use [kW]': hourly_data['use [kW]'],
                    'gen [kW]': hourly_data_gen['gen [kW]']
                }).fillna(0)
                hourly_net_data['Net [kWh]'] = hourly_net_data['use [kW]'] - hourly_net_data['gen [kW]']
                
                # Filter net energy data for slider hour range
                slider_net_data = hourly_net_data[slider_mask]

                if slider_net_data.empty:
                    st.warning("No data available for selected hour range for line chart")
                else:
                    # Create line chart for net energy with slider range
                    fig = px.line(
                        slider_net_data,
                        x=slider_net_data.index,
                        y='Net [kWh]',
                        title=f'HOURLY NET ENERGY TREND - {selected_date.strftime("%d/%m/%Y")} ({slider_start_hour:02d}:00 to {slider_end_hour:02d}:00)',
                        labels={
                            'Net [kWh]': 'kWh',
                            'datetime': 'Hour'
                        },
                        color_discrete_sequence=['#3498db'],
                        markers=True
                    )
                    
                    fig.update_layout(
                        xaxis_title="Hour",
                        yaxis_title="kWh",
                        hovermode="x unified",
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            tickformat='%H:%M'
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Data processing error: {str(e)}")

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

                # Date range slider for both charts
                slider_date_range = st.slider(
                    "Select date range to zoom",
                    min_value=start_date,
                    max_value=end_date,
                    value=(start_date, end_date),
                    format="DD/MM/YYYY",
                    key="tab2_date_slider"
                )

                # Extract slider start and end dates
                slider_start_date, slider_end_date = slider_date_range

                # Filter combined data for slider date range
                slider_mask = (combined_data.index.date >= slider_start_date) & (combined_data.index.date <= slider_end_date)
                slider_combined_data = combined_data[slider_mask]

                # Create grouped bar chart with slider range
                if slider_combined_data.empty:
                    st.warning("No data available for selected date range for bar chart")
                else:
                    fig = px.bar(
                        slider_combined_data,
                        x=slider_combined_data.index,
                        y=['use [kW]', 'gen [kW]'],
                        title=f"ENERGY CONSUMPTION & GENERATION<br>From {slider_start_date.strftime('%d/%m/%Y')} to {slider_end_date.strftime('%d/%m/%Y')}",
                        labels={'value': 'kWh', 'datetime': 'Date'},
                        color_discrete_sequence=['#3498db', '#e74c3c']
                    )

                    fig.update_layout(
                        xaxis_tickformat='%d/%m',
                        hovermode="x unified",
                        plot_bgcolor='white',
                        height=450,
                        barmode='group',
                        xaxis_title="Date",
                        yaxis_title="kWh"
                    )

                    fig.update_traces(
                        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>%{y:.2f} kWh",
                        texttemplate='%{y:.1f}',
                        textposition='outside'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Net energy section
                st.subheader("NET ENERGY (CONSUMPTION - GENERATION)")

                try:
                    # Calculate daily net energy
                    daily_data = filtered_data[['use [kW]', 'gen [kW]']].resample('D').sum() / 60  # Convert to kWh
                    daily_data['Net [kWh]'] = daily_data['use [kW]'] - daily_data['gen [kW]']
                    
                    # Filter net energy data for slider date range
                    slider_net_data = daily_data[slider_mask]

                    if slider_net_data.empty:
                        st.warning("No data available for selected date range for line chart")
                    else:
                        # Create line chart for net energy with slider range
                        fig = px.line(
                            slider_net_data,
                            x=slider_net_data.index,
                            y='Net [kWh]',
                            title=f'DAILY NET ENERGY TREND<br>From {slider_start_date.strftime("%d/%m/%Y")} to {slider_end_date.strftime("%d/%m/%Y")}',
                            labels={
                                'Net [kWh]': 'kWh',
                                'datetime': 'Date'
                            },
                            color_discrete_sequence=['#3498db'],
                            markers=True
                        )
                        
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
                    st.stop()

            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
                st.stop()

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
        st.markdown("")
        # Get available date range
        min_date = df.index.min().date()
        max_date = df.index.max().date()

        # Create date selection UI with calendar for date range
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
            # Device selection (exclude Solar [kW])
            selected_device = st.selectbox(
                "Select a device to analyze",
                NON_SOLAR_DEVICES,
                index=0,
                key="device_select"
            )
            
            if not selected_device or selected_device not in df.columns:
                st.warning("Please select a valid device")
                return
            
            # Filter data by selected date range
            date_mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            filtered_data = df.loc[date_mask]
            
            # Calculate daily consumption for metrics (full date range)
            daily_consumption = calculate_daily_for_device(filtered_data, selected_device)
            
            if daily_consumption.empty:
                st.warning(f"No data available for {selected_device} from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
                return
                
            # Display metrics in columns (full date range)
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
            st.markdown("")

            # Single date selector for hourly chart
            valid_dates = pd.Series(filtered_data.index.date).unique()
            if len(valid_dates) == 0:
                st.warning("No valid dates available in selected range")
                return
                
            hourly_date = st.date_input(
                "Select date for hourly consumption",
                valid_dates[0],
                min_value=valid_dates.min(),
                max_value=valid_dates.max(),
                format="DD/MM/YYYY",
                key="device_hourly_date"
            )
            st.markdown("")

            # Hourly Chart
            st.subheader(f"Hourly Consumption for {selected_device.replace(' [kW]', '')}")
            hourly_date_mask = (filtered_data.index.date == hourly_date)
            hourly_filtered_data = filtered_data.loc[hourly_date_mask]
            
            hourly_consumption = calculate_hourly_for_device(hourly_filtered_data, selected_device)
            
            if hourly_consumption.empty:
                st.warning(f"No hourly data available for {selected_device} on {hourly_date.strftime('%d/%m/%Y')}")
            else:
                # Hourly range slider
                slider_hour_range = st.slider(
                    "Select hour range to zoom",
                    min_value=0,
                    max_value=23,
                    value=(0, 23),
                    step=1,
                    format="%d:00",
                    key="device_tab1_hour_slider"
                )
                
                # Extract slider start and end hours
                slider_start_hour, slider_end_hour = slider_hour_range
                
                # Filter hourly data for slider range
                slider_mask = (hourly_consumption.index.hour >= slider_start_hour) & (hourly_consumption.index.hour <= slider_end_hour)
                slider_hourly_data = hourly_consumption[slider_mask]
                
                if slider_hourly_data.empty:
                    st.warning(f"No data available for {selected_device} in selected hour range")
                else:
                    fig_hourly = px.line(
                        slider_hourly_data.reset_index(),
                        x=slider_hourly_data.index.name or 'time',
                        y=selected_device,
                        title=f"{selected_device.replace(' [kW]', '')} Hourly Consumption - {hourly_date.strftime('%d/%m/%Y')} ({slider_start_hour:02d}:00 to {slider_end_hour:02d}:00)",
                        labels={
                            slider_hourly_data.index.name or 'time': 'Hour',
                            selected_device: 'Consumption (kWh)'
                        },
                        markers=True
                    )

                    fig_hourly.update_traces(
                        line=dict(width=3, color='#1f77b4'),
                        marker=dict(size=8, color='#1f77b4')
                    )

                    fig_hourly.update_layout(
                        xaxis_tickformat='%H:%M',
                        hovermode="x unified",
                        yaxis_title="Consumption (kWh)",
                        xaxis_title="Hour",
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=500
                    )

                    # Add annotations for peak values
                    max_consump_idx = slider_hourly_data[selected_device].idxmax()
                    max_consump_val = slider_hourly_data[selected_device].max()
                    
                    fig_hourly.add_annotation(
                        x=max_consump_idx,
                        y=max_consump_val,
                        text=f"Peak: {max_consump_val:.2f} kWh",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40,
                        bgcolor="white"
                    )

                    # Add annotation for minimum value
                    min_consump_idx = slider_hourly_data[selected_device].idxmin()
                    min_consump_val = slider_hourly_data[selected_device].min()
                    
                    fig_hourly.add_annotation(
                        x=min_consump_idx,
                        y=min_consump_val,
                        text=f"Min: {min_consump_val:.2f} kWh",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=40,
                        bgcolor="white"
                    )

                    st.plotly_chart(fig_hourly, use_container_width=True)

            # Daily Chart
            st.subheader(f"Daily Consumption for {selected_device.replace(' [kW]', '')}")
            # Date range slider for daily chart
            slider_date_range = st.slider(
                "Select date range to zoom",
                min_value=start_date,
                max_value=end_date,
                value=(start_date, end_date),
                format="DD/MM/YYYY",
                key="device_tab1_date_slider"
            )
            
            # Extract slider start and end dates
            slider_start_date, slider_end_date = slider_date_range
            
            # Filter daily consumption for slider date range
            slider_date_mask = (daily_consumption.index.date >= slider_start_date) & (daily_consumption.index.date <= slider_end_date)
            slider_daily_data = daily_consumption[slider_date_mask]
            
            if slider_daily_data.empty:
                st.warning(f"No data available for {selected_device} in selected date range")
            else:
                fig_daily = px.line(
                    slider_daily_data.reset_index(),
                    x=slider_daily_data.index.name or 'time',
                    y=selected_device,
                    title=f"{selected_device.replace(' [kW]', '')} Daily Consumption - {slider_start_date.strftime('%d/%m/%Y')} to {slider_end_date.strftime('%d/%m/%Y')}",
                    labels={
                        slider_daily_data.index.name or 'time': 'Date',
                        selected_device: 'Consumption (kWh)'
                    },
                    markers=True
                )

                fig_daily.update_traces(
                    line=dict(width=3, color='#1f77b4'),
                    marker=dict(size=8, color='#1f77b4')
                )

                fig_daily.update_layout(
                    xaxis_tickformat='%d/%m',
                    hovermode="x unified",
                    yaxis_title="Consumption (kWh)",
                    xaxis_title="Date",
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=500
                )

                # Add annotations for peak values
                max_consump_idx = slider_daily_data[selected_device].idxmax()
                max_consump_val = slider_daily_data[selected_device].max()
                
                fig_daily.add_annotation(
                    x=max_consump_idx,
                    y=max_consump_val,
                    text=f"Peak: {max_consump_val:.2f} kWh",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                    bgcolor="white"
                )

                # Add annotation for minimum value
                min_consump_idx = slider_daily_data[selected_device].idxmin()
                min_consump_val = slider_daily_data[selected_device].min()
                
                fig_daily.add_annotation(
                    x=min_consump_idx,
                    y=min_consump_val,
                    text=f"Min: {min_consump_val:.2f} kWh",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=40,
                    bgcolor="white"
                )

                st.plotly_chart(fig_daily, use_container_width=True)

            # Monthly Chart
            st.subheader(f"Monthly Consumption for {selected_device.replace(' [kW]', '')}")
            # Year selector for monthly chart
            available_years = sorted(df.index.year.unique(), reverse=True)
            selected_year = st.selectbox(
                "Select Year for monthly consumption",
                available_years,
                key="device_monthly_year"
            )
            
            # Month range slider for monthly chart
            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
            slider_month_range = st.select_slider(
                "Select month range to zoom",
                options=month_names,
                value=(month_names[0], month_names[11]),  # Default to January to December
                key="device_tab1_month_slider"
            )
            
            # Convert selected month names to month numbers
            slider_start_month = month_names.index(slider_month_range[0]) + 1
            slider_end_month = month_names.index(slider_month_range[1]) + 1
            
            # Filter data by selected year
            year_mask = (df.index.year == selected_year)
            year_filtered_data = df.loc[year_mask]
            
            # Calculate monthly consumption
            monthly_consumption = calculate_monthly_for_device(year_filtered_data, selected_device)
            
            if monthly_consumption.empty:
                st.warning(f"No monthly data available for {selected_device} in {selected_year}")
            else:
                # Filter monthly consumption for slider month range
                slider_month_mask = (monthly_consumption.index.month >= slider_start_month) & (monthly_consumption.index.month <= slider_end_month)
                slider_monthly_data = monthly_consumption[slider_month_mask]
                
                if slider_monthly_data.empty:
                    st.warning(f"No data available for {selected_device} in selected month range")
                else:
                    fig_monthly = px.line(
                        slider_monthly_data.reset_index(),
                        x=slider_monthly_data.index.name or 'time',
                        y=selected_device,
                        title=f"{selected_device.replace(' [kW]', '')} Monthly Consumption - {month_names[slider_start_month-1]} to {month_names[slider_end_month-1]} {selected_year}",
                        labels={
                            slider_monthly_data.index.name or 'time': 'Month',
                            selected_device: 'Consumption (kWh)'
                        },
                        markers=True
                    )

                    fig_monthly.update_traces(
                        line=dict(width=3, color='#1f77b4'),
                        marker=dict(size=8, color='#1f77b4')
                    )

                    fig_monthly.update_layout(
                        xaxis_tickformat='%b',
                        hovermode="x unified",
                        yaxis_title="Consumption (kWh)",
                        xaxis_title="Month",
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=500
                    )

                    # Add annotations for peak values
                    max_consump_idx = slider_monthly_data[selected_device].idxmax()
                    max_consump_val = slider_monthly_data[selected_device].max()
                    
                    fig_monthly.add_annotation(
                        x=max_consump_idx,
                        y=max_consump_val,
                        text=f"Peak: {max_consump_val:.2f} kWh",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40,
                        bgcolor="white"
                    )

                    # Add annotation for minimum value
                    min_consump_idx = slider_monthly_data[selected_device].idxmin()
                    min_consump_val = slider_monthly_data[selected_device].min()
                    
                    fig_monthly.add_annotation(
                        x=min_consump_idx,
                        y=min_consump_val,
                        text=f"Min: {min_consump_val:.2f} kWh",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=40,
                        bgcolor="white"
                    )

                    st.plotly_chart(fig_monthly, use_container_width=True)
            
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
        
        # Device selection (exclude Solar [kW])
        selected_devices = st.multiselect(
            "Select devices to analyze (or leave empty for all devices)",
            NON_SOLAR_DEVICES,
            default=[],
            key="devices_tab2_select"
        )
        
        # Use all non-solar devices if none selected
        devices_to_analyze = selected_devices if selected_devices else NON_SOLAR_DEVICES
        
        try:
            # Calculate total consumption for each device
            device_totals = filtered_df[devices_to_analyze].sum().sort_values(ascending=False)
            
            # Calculate percentage of total consumption
            total_consumption = device_totals.sum()
            device_percentages = (device_totals / total_consumption * 100).round(1)
            
            # Get top 5 devices and group the rest into "Other" for pie chart
            top_devices = device_percentages.head(5)
            other_devices = device_percentages[5:]
            
            if len(other_devices) > 0:
                other_percentage = other_devices.sum()
                top_devices['Other'] = other_percentage
            
            # Prepare data for pie chart
            pie_data = top_devices.reset_index()
            pie_data.columns = ['Device', 'Percentage']
            pie_data['Device'] = pie_data['Device'].str.replace(" [kW]", "")
            
            # Prepare data for bar chart (top 5 devices only, no "Other")
            bar_data = device_totals.head(5).reset_index()
            bar_data.columns = ['Device', 'Consumption']
            bar_data['Device'] = bar_data['Device'].str.replace(" [kW]", "")
            
            # Create color mapping to ensure consistency
            # Map top 5 devices to the first 5 colors, "Other" (if present) to the 6th
            color_map = {}
            top_5_devices = bar_data['Device'].tolist()
            for i, device in enumerate(top_5_devices):
                color_map[device] = px.colors.sequential.RdBu[i]
            if 'Other' in pie_data['Device'].values:
                color_map['Other'] = px.colors.sequential.RdBu[5]  # Use 6th color for "Other"
            
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
                    color='Device',
                    color_discrete_map=color_map
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
                # Bar chart showing absolute consumption for top 5 devices
                fig_bar = px.bar(
                    bar_data,
                    x='Device',
                    y='Consumption',
                    title='Detailed Consumption by Device',
                    labels={'Consumption': 'Energy (kW)'},
                    color='Device',
                    color_discrete_map=color_map
                )
                fig_bar.update_traces(
                    hovertemplate="<b>%{x}</b><br>%{y:,.0f} kW"
                )
                fig_bar.update_layout(
                    xaxis_title="Device",
                    yaxis_title="Total Consumption (kW)",
                    xaxis={'categoryorder':'total descending'},
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing device data: {str(e)}")
    
    with tab3:
        st.subheader("Device Consumption Correlations")
        
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
                key="devices_tab3_start",
                format="DD/MM/YYYY"
            )
        with col2:
            end_date = st.date_input(
                "To date",
                max_date,
                min_value=min_date,
                max_value=max_date,
                key="devices_tab3_end",
                format="DD/MM/YYYY"
            )

        # Validate date range
        if start_date > end_date:
            st.error("End date must be after start date!")
            st.stop()
        st.markdown("")

        # Device selection (exclude Solar [kW])
        selected_devices = st.multiselect(
            "Select devices to analyze correlations (or leave empty for all devices)",
            NON_SOLAR_DEVICES,
            default=[],
            key="devices_tab3_select"
        )

        # Use all non-solar devices if none selected
        devices_to_analyze = selected_devices if selected_devices else NON_SOLAR_DEVICES

        try:
            # Filter data by selected date range
            date_mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            filtered_df = df.loc[date_mask]

            if filtered_df.empty:
                st.warning(f"No data available for the selected date range")
                return

            # Calculate daily consumption for each device
            daily_data = pd.DataFrame()
            for device in devices_to_analyze:
                device_daily = calculate_daily_for_device(filtered_df, device)
                if not device_daily.empty:
                    daily_data[device.replace(" [kW]", "")] = device_daily[device]

            if daily_data.empty or len(daily_data.columns) < 2:
                st.warning("Insufficient data or too few devices selected for correlation analysis")
                return

            # Compute correlation matrix
            corr_matrix = daily_data.corr(method='pearson').round(2)

            # Create correlation heatmap
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                title='Correlation Matrix: Device Daily Consumption',
                labels=dict(color='Correlation'),
                aspect='equal'
            )
            fig_corr.update_layout(
                height=600,
                width=600,
                margin=dict(t=80, b=80, l=80, r=80),
                xaxis=dict(
                    tickangle=45,
                    title="Device",
                    side="bottom"
                ),
                yaxis=dict(
                    title="Device"
                ),
                coloraxis_colorbar=dict(
                    title="Correlation",
                    thickness=15,
                    len=0.5,
                    yanchor="middle",
                    y=0.5
                )
            )
            fig_corr.update_traces(
                hovertemplate="Device X: %{x}<br>Device Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Find strongest positive and negative correlations
            corr_matrix_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            max_corr = corr_matrix_upper.max().max()
            min_corr = corr_matrix_upper.min().min()

            if not pd.isna(max_corr):
                max_pair = corr_matrix_upper.stack().idxmax()
                max_pair_str = f"{max_pair[0]} & {max_pair[1]}"
                max_corr_val = corr_matrix.loc[max_pair[0], max_pair[1]]
            else:
                max_pair_str = "N/A"
                max_corr_val = "N/A"

            if not pd.isna(min_corr):
                min_pair = corr_matrix_upper.stack().idxmin()
                min_pair_str = f"{min_pair[0]} & {min_pair[1]}"
                min_corr_val = corr_matrix.loc[min_pair[0], min_pair[1]]
            else:
                min_pair_str = "N/A"
                min_corr_val = "N/A"

            # Display correlation metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Strongest Positive Correlation",
                    value=f"{max_corr_val:.2f}" if max_corr_val != "N/A" else "N/A",
                    delta=max_pair_str
                )
            with col2:
                st.metric(
                    label="Strongest Negative Correlation",
                    value=f"{min_corr_val:.2f}" if min_corr_val != "N/A" else "N/A",
                    delta=min_pair_str
                )

            # Expander with instructions
            with st.expander("How to read the correlation heatmap"):
                st.markdown("""
                - **Heatmap**: Shows Pearson correlations between daily energy consumption of selected devices.
                - **Color Scale**: Red indicates negative correlations (one device’s high usage corresponds to another’s low usage), blue indicates positive correlations (devices’ usage patterns align).
                - **Values**: Range from -1 (strong negative correlation) to 1 (strong positive correlation); 0 means no correlation.
                - **Diagonal**: Always 1 (each device correlates perfectly with itself).
                - **Hover**: Displays device pairs and their correlation value.
                - **Metrics**: Highlight the strongest positive and negative correlations (excluding self-correlations).
                - Use the date range and device selection to focus the analysis.
                """)

        except Exception as e:
            st.error(f"Error processing correlation data: {str(e)}")

def forecasting_page(df):
    """
    Page showing time series forecasting for daily energy consumption and a selected device's daily energy consumption.
    Includes comparison of Prophet and ARIMA models using MAE and RMSE.
    """
    st.header("📈 Energy Consumption Forecasting")
    
    if df is None:
        st.warning("No data available")
        return
    
    tab1, tab2 = st.tabs(["📈 Energy Usage & Generation Forecasting", "📈 Device Energy Consumption Forecasting"])
    with tab1:
        st.markdown("")
        # Get available date range
        min_date = df.index.min().date()
        max_date = df.index.max().date()

        # Create date selection UI with calendar for historical data
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From date",
                min_date,
                min_value=min_date,
                max_value=max_date,
                key="energy_forecast_start_date",
                format="DD/MM/YYYY"
            )
        with col2:
            end_date = st.date_input(
                "To date",
                max_date,
                min_value=min_date,
                max_value=max_date,
                key="energy_forecast_end_date",
                format="DD/MM/YYYY"
            )

        # Validate date range
        if start_date > end_date:
            st.error("End date must be after start date!")
            st.stop()

        # Energy metric selection
        selected_metric = st.selectbox(
            "Select energy metric to forecast",
            ['use [kW]', 'gen [kW]'],
            index=0,
            key="energy_metric_select"
        )

        # Forecast horizon selection
        forecast_horizon = st.selectbox(
            "Select forecast horizon (days)",
            [7, 14, 30],
            index=0,
            key="energy_forecast_horizon"
        )

        try:
            # Filter data by date range
            date_mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            filtered_df = df.loc[date_mask]

            # Calculate daily data for the selected metric
            if selected_metric == 'use [kW]':
                daily_data = calculate_daily_for_use(filtered_df, selected_metric)
            else:  # gen [kW]
                daily_data = calculate_daily_for_gen(filtered_df, selected_metric)
            
            if daily_data.empty:
                st.warning(f"No data available for {selected_metric} from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
                return

            # Prepare data for forecasting
            daily_data = daily_data.reset_index()
            daily_data = daily_data.rename(columns={'time': 'ds', selected_metric: 'y'})
            daily_data['ds'] = pd.to_datetime(daily_data['ds'])

            # Remove duplicate dates and sort
            daily_data = daily_data.drop_duplicates(subset=['ds'], keep='last').sort_values('ds')

            # Create a complete date range and merge to fill missing dates
            full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            full_df = pd.DataFrame({'ds': full_date_range})
            daily_data = full_df.merge(daily_data, on='ds', how='left')
            
            # Check for missing data and list missing dates
            missing_dates = daily_data[daily_data['y'].isna()]['ds'].tolist()
            missing_count = len(missing_dates)
            if missing_count > 0:
                st.info(f"Missing {missing_count} days ({(missing_count/len(full_date_range)*100):.1f}%): {[d.strftime('%d/%m/%Y') for d in missing_dates]}")
                if missing_dates:
                    missing_df = pd.DataFrame({'Missing Dates': [d.strftime('%d/%m/%Y') for d in missing_dates]})
                    st.download_button(
                        "Download Missing Dates",
                        missing_df.to_csv(index=False).encode('utf-8'),
                        "missing_dates_energy.csv",
                        "text/csv"
                    )
            if missing_count > len(full_date_range) * 0.1:  # Warn if >10% missing
                st.warning(f"Warning: {missing_count} days ({(missing_count/len(full_date_range)*100):.1f}%) are missing data. Results may be unreliable.")
            
            # Fill missing values with interpolation, fallback to forward/backward fill and zero
            daily_data['y'] = daily_data['y'].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill').fillna(0)

            # Ensure sufficient data
            if len(daily_data) < 30:
                st.warning("Insufficient data for forecasting (minimum 30 days required)")
                return

            # Ensure last date matches end_date
            if daily_data['ds'].max().date() != end_date:
                st.warning(f"Data ends at {daily_data['ds'].max().date().strftime('%d/%m/%Y')}, not {end_date.strftime('%d/%m/%Y')}. Adjusting forecast start.")

            # Split data into training and validation sets (last 14 days for validation if available)
            validation_period = min(14, len(daily_data) // 2)  # Use up to 14 days or half the data
            train_data = daily_data.iloc[:-validation_period].copy()
            # Prophet Forecasting
            prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            prophet_model.fit(train_data[['ds', 'y']])

            # Create future dataframe for Prophet (including validation period + forecast horizon)
            future_dates = prophet_model.make_future_dataframe(periods=forecast_horizon + validation_period, freq='D')
            prophet_forecast = prophet_model.predict(future_dates)
            prophet_forecast = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            # ARIMA Forecasting
            arima_model = ARIMA(train_data['y'], order=(5, 1, 0))
            arima_results = arima_model.fit()
            arima_forecast = arima_results.get_forecast(steps=forecast_horizon + validation_period)
            arima_mean = arima_forecast.predicted_mean
            arima_conf_int = arima_forecast.conf_int(alpha=0.05)
            
            # Prepare ARIMA forecast dataframe
            last_date = train_data['ds'].max()
            arima_dates = pd.date_range(start=last_date, periods=forecast_horizon + validation_period + 1, freq='D')[1:]
            forecast_arima = pd.DataFrame({
                'ds': arima_dates,
                'yhat': arima_mean.values,
                'yhat_lower': arima_conf_int.iloc[:, 0].values,
                'yhat_upper': arima_conf_int.iloc[:, 1].values
            })
            
            # Set historical yhat, yhat_lower, yhat_upper to y for consistency
            historical_arima = train_data[['ds', 'y']].copy()
            historical_arima['yhat'] = historical_arima['y']
            historical_arima['yhat_lower'] = historical_arima['y']
            historical_arima['yhat_upper'] = historical_arima['y']
            
            # Concatenate historical and forecast data for ARIMA
            arima_forecast_df = pd.concat([historical_arima[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], forecast_arima], ignore_index=True)

            # Evaluate forecasts
            prophet_mae, prophet_rmse, arima_mae, arima_rmse = evaluate_forecasts(daily_data, prophet_forecast, arima_forecast_df, validation_period)

            # Combine historical and forecast data for visualization
            historical_data = daily_data[['ds', 'y']].copy()
            historical_data['type'] = 'Historical'
            
            prophet_data = prophet_forecast.copy()
            prophet_data['type'] = 'Prophet Forecast'
            prophet_data['y'] = prophet_data['yhat']
            
            arima_data = arima_forecast_df.copy()
            arima_data['type'] = 'ARIMA Forecast'
            arima_data['y'] = arima_data['yhat']

            # Create combined dataframe for plotting
            plot_data = pd.concat([
                historical_data[['ds', 'y', 'type']],
                prophet_data[['ds', 'y', 'type', 'yhat_lower', 'yhat_upper']],
                arima_data[['ds', 'y', 'type', 'yhat_lower', 'yhat_upper']]
            ], ignore_index=True)

            # Add date range slider for zooming
            plot_min_date = plot_data['ds'].min().date()
            plot_max_date = plot_data['ds'].max().date()
            slider_date_range = st.slider(
                "Select date range to zoom",
                min_value=plot_min_date,
                max_value=plot_max_date,
                value=(plot_min_date, plot_max_date),
                format="DD/MM/YYYY",
                key="energy_forecast_zoom_slider"
            )
            slider_start_date, slider_end_date = slider_date_range
            plot_data = plot_data[(plot_data['ds'].dt.date >= slider_start_date) & (plot_data['ds'].dt.date <= slider_end_date)]

            # Create line chart
            fig = go.Figure()

            # Historical data
            historical_plot = plot_data[plot_data['type'] == 'Historical']
            fig.add_trace(go.Scatter(
                x=historical_plot['ds'],
                y=historical_plot['y'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))

            # Prophet forecast
            prophet_plot = plot_data[plot_data['type'] == 'Prophet Forecast']
            fig.add_trace(go.Scatter(
                x=prophet_plot['ds'],
                y=prophet_plot['y'],
                mode='lines',
                name='Prophet Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=prophet_plot['ds'],
                y=prophet_plot['yhat_upper'],
                mode='lines',
                name='Prophet Upper CI',
                line=dict(color='rgba(255, 127, 14, 0.2)', width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=prophet_plot['ds'],
                y=prophet_plot['yhat_lower'],
                mode='lines',
                name='Prophet Lower CI',
                line=dict(color='rgba(255, 127, 14, 0.2)', width=0),
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                showlegend=False
            ))

            # ARIMA forecast
            arima_plot = plot_data[plot_data['type'] == 'ARIMA Forecast']
            fig.add_trace(go.Scatter(
                x=arima_plot['ds'],
                y=arima_plot['y'],
                mode='lines',
                name='ARIMA Forecast',
                line=dict(color='#2ca02c', width=2, dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=arima_plot['ds'],
                y=arima_plot['yhat_upper'],
                mode='lines',
                name='ARIMA Upper CI',
                line=dict(color='rgba(44, 160, 44, 0.2)', width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=arima_plot['ds'],
                y=arima_plot['yhat_lower'],
                mode='lines',
                name='ARIMA Lower CI',
                line=dict(color='rgba(44, 160, 44, 0.2)', width=0),
                fill='tonexty',
                fillcolor='rgba(44, 160, 44, 0.2)',
                showlegend=False
            ))

            # Add vertical line for forecast start
            forecast_start = daily_data['ds'].max()
            all_y_values = []
            for trace_data in [historical_plot, prophet_plot, arima_plot]:
                if not trace_data.empty:
                    all_y_values.extend(trace_data['y'].dropna().tolist())
            
            if all_y_values:
                y_min = min(all_y_values)
                y_max = max(all_y_values)
                if forecast_start.date() >= slider_start_date and forecast_start.date() <= slider_end_date:
                    fig.add_trace(go.Scatter(
                        x=[forecast_start, forecast_start],
                        y=[y_min, y_max],
                        mode='lines',
                        line=dict(color='gray', width=2, dash='dash'),
                        name='Forecast Start',
                        showlegend=False,
                        hovertemplate='Forecast Start<br>Date: %{x}<extra></extra>'
                    ))
                    fig.add_annotation(
                        x=forecast_start,
                        y=y_max * 0.9,
                        text="Forecast Start",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="gray",
                        bgcolor="white",
                        bordercolor="gray",
                        borderwidth=1
                    )

            # Update layout
            fig.update_layout(
                title=f"Daily Energy {selected_metric.replace(' [kW]', '')} Forecast",
                xaxis_title="Date",
                yaxis_title="Energy (kWh)",
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)',
                height=600,
                xaxis=dict(
                    tickformat='%d/%m/%Y',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                legend=dict(
                    x=0.01,
                    y=0.99,
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                margin=dict(t=80, b=80, l=80, r=80)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display model comparison
            display_forecast_comparison(prophet_mae, prophet_rmse, arima_mae, arima_rmse, selected_metric.replace(' [kW]', ''))

            # Display forecast summary
            st.subheader("📊 Forecast Summary")
            forecast_only_prophet = prophet_forecast[prophet_forecast['ds'] > daily_data['ds'].max()]
            forecast_only_arima = forecast_arima[forecast_arima['ds'] > daily_data['ds'].max()]
            
            if not forecast_only_prophet.empty and not forecast_only_arima.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Prophet Forecast**")
                    prophet_avg = forecast_only_prophet['yhat'].mean()
                    prophet_total = forecast_only_prophet['yhat'].sum()
                    st.metric("Average Daily", f"{prophet_avg:.2f} kWh")
                    st.metric("Total Period", f"{prophet_total:.2f} kWh")
                with col2:
                    st.markdown("**ARIMA Forecast**")
                    arima_avg = forecast_only_arima['yhat'].mean()
                    arima_total = forecast_only_arima['yhat'].sum()
                    st.metric("Average Daily", f"{arima_avg:.2f} kWh")
                    st.metric("Total Period", f"{arima_total:.2f} kWh")

            # Expander with instructions
            with st.expander("How to read the forecast chart"):
                st.markdown("""
                - **Historical Data**: Blue line shows actual daily energy for the selected metric.
                - **Prophet Forecast**: Orange dashed line shows predicted energy using Facebook Prophet, with shaded confidence intervals.
                - **ARIMA Forecast**: Green dotted line shows predicted energy using ARIMA, with shaded confidence intervals.
                - **Forecast Start**: Gray vertical line marks the start of the forecast period (if within the selected range).
                - **Hover**: View exact values and dates.
                - **Zoom Slider**: Adjust the date range to focus on specific periods, including historical or forecast data.
                - Adjust the date range, metric, and forecast horizon to explore different scenarios.
                - Note: The visualization prioritizes clarity over prediction accuracy.
                """)

        except Exception as e:
            st.error(f"Error in energy forecasting: {str(e)}")
            st.exception(e)

    with tab2:
        st.markdown("")
        # Get available date range
        min_date = df.index.min().date()
        max_date = df.index.max().date()

        # Create date selection UI with calendar for historical data
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From date",
                min_date,
                min_value=min_date,
                max_value=max_date,
                key="device_forecast_start_date",
                format="DD/MM/YYYY"
            )
        with col2:
            end_date = st.date_input(
                "To date",
                max_date,
                min_value=min_date,
                max_value=max_date,
                key="device_forecast_end_date",
                format="DD/MM/YYYY"
            )

        # Validate date range
        if start_date > end_date:
            st.error("End date must be after start date!")
            st.stop()

        # Device selection
        selected_device = st.selectbox(
            "Select a device to forecast",
            NON_SOLAR_DEVICES,
            index=0,
            key="device_forecast_select"
        )

        # Forecast horizon selection
        forecast_horizon = st.selectbox(
            "Select forecast horizon (days)",
            [7, 14, 30],
            index=0,
            key="device_forecast_horizon"
        )

        try:
            # Filter data by date range
            date_mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            filtered_df = df.loc[date_mask]

            # Calculate daily consumption for the selected device
            daily_data = calculate_daily_for_device(filtered_df, selected_device)
            
            if daily_data.empty:
                st.warning(f"No data available for {selected_device} from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
                return

            # Prepare data for forecasting
            daily_data = daily_data.reset_index()
            daily_data = daily_data.rename(columns={'time': 'ds', selected_device: 'y'})
            daily_data['ds'] = pd.to_datetime(daily_data['ds'])

            # Remove duplicate dates and sort
            daily_data = daily_data.drop_duplicates(subset=['ds'], keep='last').sort_values('ds')

            # Create a complete date range and merge to fill missing dates
            full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            full_df = pd.DataFrame({'ds': full_date_range})
            daily_data = full_df.merge(daily_data, on='ds', how='left')
            
            # Check for missing data and list missing dates
            missing_dates = daily_data[daily_data['y'].isna()]['ds'].tolist()
            missing_count = len(missing_dates)
            if missing_count > 0:
                st.info(f"Missing {missing_count} days ({(missing_count/len(full_date_range)*100):.1f}%): {[d.strftime('%d/%m/%Y') for d in missing_dates]}")
                if missing_dates:
                    missing_df = pd.DataFrame({'Missing Dates': [d.strftime('%d/%m/%Y') for d in missing_dates]})
                    st.download_button(
                        "Download Missing Dates",
                        missing_df.to_csv(index=False).encode('utf-8'),
                        "missing_dates_device.csv",
                        "text/csv"
                    )
            if missing_count > len(full_date_range) * 0.1:  # Warn if >10% missing
                st.warning(f"Warning: {missing_count} days ({(missing_count/len(full_date_range)*100):.1f}%) are missing data. Results may be unreliable.")
            
            # Fill missing values with interpolation, fallback to forward/backward fill and zero
            daily_data['y'] = daily_data['y'].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill').fillna(0)

            # Ensure sufficient data
            if len(daily_data) < 30:
                st.warning("Insufficient data for forecasting (minimum 30 days required)")
                return

            # Ensure last date matches end_date
            if daily_data['ds'].max().date() != end_date:
                st.warning(f"Data ends at {daily_data['ds'].max().date().strftime('%d/%m/%Y')}, not {end_date.strftime('%d/%m/%Y')}. Adjusting forecast start.")

            # Split data into training and validation sets (last 14 days for validation if available)
            validation_period = min(14, len(daily_data) // 2)  # Use up to 14 days or half the data
            train_data = daily_data.iloc[:-validation_period].copy()

            # Prophet Forecasting
            prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            prophet_model.fit(train_data[['ds', 'y']])

            # Create future dataframe for Prophet (including validation period + forecast horizon)
            future_dates = prophet_model.make_future_dataframe(periods=forecast_horizon + validation_period, freq='D')
            prophet_forecast = prophet_model.predict(future_dates)
            prophet_forecast = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            # ARIMA Forecasting
            arima_model = ARIMA(train_data['y'], order=(5, 1, 0))
            arima_results = arima_model.fit()
            arima_forecast = arima_results.get_forecast(steps=forecast_horizon + validation_period)
            arima_mean = arima_forecast.predicted_mean
            arima_conf_int = arima_forecast.conf_int(alpha=0.05)
            
            # Prepare ARIMA forecast dataframe
            last_date = train_data['ds'].max()
            arima_dates = pd.date_range(start=last_date, periods=forecast_horizon + validation_period + 1, freq='D')[1:]
            forecast_arima = pd.DataFrame({
                'ds': arima_dates,
                'yhat': arima_mean.values,
                'yhat_lower': arima_conf_int.iloc[:, 0].values,
                'yhat_upper': arima_conf_int.iloc[:, 1].values
            })
            
            # Set historical yhat, yhat_lower, yhat_upper to y for consistency
            historical_arima = train_data[['ds', 'y']].copy()
            historical_arima['yhat'] = historical_arima['y']
            historical_arima['yhat_lower'] = historical_arima['y']
            historical_arima['yhat_upper'] = historical_arima['y']
            
            # Concatenate historical and forecast data
            arima_forecast_df = pd.concat([historical_arima[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], forecast_arima], ignore_index=True)

            # Evaluate forecasts
            prophet_mae, prophet_rmse, arima_mae, arima_rmse = evaluate_forecasts(daily_data, prophet_forecast, arima_forecast_df, validation_period)

            # Combine historical and forecast data for visualization
            historical_data = daily_data[['ds', 'y']].copy()
            historical_data['type'] = 'Historical'
            
            prophet_data = prophet_forecast.copy()
            prophet_data['type'] = 'Prophet Forecast'
            prophet_data['y'] = prophet_data['yhat']
            
            arima_data = arima_forecast_df.copy()
            arima_data['type'] = 'ARIMA Forecast'
            arima_data['y'] = arima_data['yhat']

            # Create combined dataframe for plotting
            plot_data = pd.concat([
                historical_data[['ds', 'y', 'type']],
                prophet_data[['ds', 'y', 'type', 'yhat_lower', 'yhat_upper']],
                arima_data[['ds', 'y', 'type', 'yhat_lower', 'yhat_upper']]
            ], ignore_index=True)

            # Add date range slider for zooming
            plot_min_date = plot_data['ds'].min().date()
            plot_max_date = plot_data['ds'].max().date()
            slider_date_range = st.slider(
                "Select date range to zoom",
                min_value=plot_min_date,
                max_value=plot_max_date,
                value=(plot_min_date, plot_max_date),
                format="DD/MM/YYYY",
                key="device_forecast_zoom_slider"
            )
            slider_start_date, slider_end_date = slider_date_range
            plot_data = plot_data[(plot_data['ds'].dt.date >= slider_start_date) & (plot_data['ds'].dt.date <= slider_end_date)]

            # Create line chart
            fig = go.Figure()

            # Historical data
            historical_plot = plot_data[plot_data['type'] == 'Historical']
            fig.add_trace(go.Scatter(
                x=historical_plot['ds'],
                y=historical_plot['y'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))

            # Prophet forecast
            prophet_plot = plot_data[plot_data['type'] == 'Prophet Forecast']
            fig.add_trace(go.Scatter(
                x=prophet_plot['ds'],
                y=prophet_plot['y'],
                mode='lines',
                name='Prophet Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=prophet_plot['ds'],
                y=prophet_plot['yhat_upper'],
                mode='lines',
                name='Prophet Upper CI',
                line=dict(color='rgba(255, 127, 14, 0.2)', width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=prophet_plot['ds'],
                y=prophet_plot['yhat_lower'],
                mode='lines',
                name='Prophet Lower CI',
                line=dict(color='rgba(255, 127, 14, 0.2)', width=0),
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                showlegend=False
            ))

            # ARIMA forecast
            arima_plot = plot_data[plot_data['type'] == 'ARIMA Forecast']
            fig.add_trace(go.Scatter(
                x=arima_plot['ds'],
                y=arima_plot['y'],
                mode='lines',
                name='ARIMA Forecast',
                line=dict(color='#2ca02c', width=2, dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=arima_plot['ds'],
                y=arima_plot['yhat_upper'],
                mode='lines',
                name='ARIMA Upper CI',
                line=dict(color='rgba(44, 160, 44, 0.2)', width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=arima_plot['ds'],
                y=arima_plot['yhat_lower'],
                mode='lines',
                name='ARIMA Lower CI',
                line=dict(color='rgba(44, 160, 44, 0.2)', width=0),
                fill='tonexty',
                fillcolor='rgba(44, 160, 44, 0.2)',
                showlegend=False
            ))

            # Add vertical line for forecast start
            forecast_start = daily_data['ds'].max()
            all_y_values = []
            for trace_data in [historical_plot, prophet_plot, arima_plot]:
                if not trace_data.empty:
                    all_y_values.extend(trace_data['y'].dropna().tolist())
            
            if all_y_values:
                y_min = min(all_y_values)
                y_max = max(all_y_values)
                if forecast_start.date() >= slider_start_date and forecast_start.date() <= slider_end_date:
                    fig.add_trace(go.Scatter(
                        x=[forecast_start, forecast_start],
                        y=[y_min, y_max],
                        mode='lines',
                        line=dict(color='gray', width=2, dash='dash'),
                        name='Forecast Start',
                        showlegend=False,
                        hovertemplate='Forecast Start<br>Date: %{x}<extra></extra>'
                    ))
                    fig.add_annotation(
                        x=forecast_start,
                        y=y_max * 0.9,
                        text="Forecast Start",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="gray",
                        bgcolor="white",
                        bordercolor="gray",
                        borderwidth=1
                    )

            # Update layout
            fig.update_layout(
                title=f"Daily Energy Consumption Forecast for {selected_device.replace(' [kW]', '')}",
                xaxis_title="Date",
                yaxis_title="Consumption (kWh)",
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)',
                height=600,
                xaxis=dict(
                    tickformat='%d/%m/%Y',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                legend=dict(
                    x=0.01,
                    y=0.99,
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                margin=dict(t=80, b=80, l=80, r=80)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display model comparison
            display_forecast_comparison(prophet_mae, prophet_rmse, arima_mae, arima_rmse, selected_device.replace(' [kW]', ''))

            # Display forecast summary
            st.subheader("📊 Forecast Summary")
            forecast_only_prophet = prophet_forecast[prophet_forecast['ds'] > daily_data['ds'].max()]
            forecast_only_arima = forecast_arima[forecast_arima['ds'] > daily_data['ds'].max()]
            
            if not forecast_only_prophet.empty and not forecast_only_arima.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Prophet Forecast**")
                    prophet_avg = forecast_only_prophet['yhat'].mean()
                    prophet_total = forecast_only_prophet['yhat'].sum()
                    st.metric("Average Daily", f"{prophet_avg:.2f} kWh")
                    st.metric("Total Period", f"{prophet_total:.2f} kWh")
                with col2:
                    st.markdown("**ARIMA Forecast**")
                    arima_avg = forecast_only_arima['yhat'].mean()
                    arima_total = forecast_only_arima['yhat'].sum()
                    st.metric("Average Daily", f"{arima_avg:.2f} kWh")
                    st.metric("Total Period", f"{arima_total:.2f} kWh")

            # Expander with instructions
            with st.expander("How to read the forecast chart"):
                st.markdown("""
                - **Historical Data**: Blue line shows actual daily consumption for the selected device.
                - **Prophet Forecast**: Orange dashed line shows predicted consumption using Facebook Prophet, with shaded confidence intervals.
                - **ARIMA Forecast**: Green dotted line shows predicted consumption using ARIMA, with shaded confidence intervals.
                - **Forecast Start**: Gray vertical line marks the start of the forecast period (if within the selected range).
                - **Hover**: View exact values and dates.
                - **Zoom Slider**: Adjust the date range to focus on specific periods, including historical or forecast data.
                - Adjust the date range, device, and forecast horizon to explore different scenarios.
                - Note: The visualization prioritizes clarity over prediction accuracy.
                """)

        except Exception as e:
            st.error(f"Error in device forecasting: {str(e)}")
            st.exception(e)

def evaluate_forecasts(historical_data, prophet_forecast, arima_forecast, validation_period):
        """
        Evaluate Prophet and ARIMA forecasts using MAE and RMSE for the validation period.
        """
        # Filter historical data for the validation period
        validation_data = historical_data[historical_data['ds'].dt.date >= (historical_data['ds'].max().date() - timedelta(days=validation_period))]
        
        # Get Prophet forecast for validation period
        prophet_val = prophet_forecast[
            (prophet_forecast['ds'].dt.date >= validation_data['ds'].min().date()) &
            (prophet_forecast['ds'].dt.date <= validation_data['ds'].max().date())
        ][['ds', 'yhat']].rename(columns={'yhat': 'yhat_prophet'})
        
        # Get ARIMA forecast for validation period
        arima_val = arima_forecast[
            (arima_forecast['ds'].dt.date >= validation_data['ds'].min().date()) &
            (arima_forecast['ds'].dt.date <= validation_data['ds'].max().date())
        ][['ds', 'yhat']].rename(columns={'yhat': 'yhat_arima'})
        
        # Merge with actual validation data
        validation_data = validation_data[['ds', 'y']].merge(
            prophet_val[['ds', 'yhat_prophet']], on='ds', how='left'
        ).merge(
            arima_val[['ds', 'yhat_arima']], on='ds', how='left'
        )
        
        # Drop rows with missing forecasts
        validation_data = validation_data.dropna()
        
        if validation_data.empty:
            return None, None, None, None
        
        # Calculate MAE and RMSE
        prophet_mae = mean_absolute_error(validation_data['y'], validation_data['yhat_prophet'])
        prophet_rmse = np.sqrt(mean_squared_error(validation_data['y'], validation_data['yhat_prophet']))
        arima_mae = mean_absolute_error(validation_data['y'], validation_data['yhat_arima'])
        arima_rmse = np.sqrt(mean_squared_error(validation_data['y'], validation_data['yhat_arima']))
        
        return prophet_mae, prophet_rmse, arima_mae, arima_rmse 
def display_forecast_comparison(prophet_mae, prophet_rmse, arima_mae, arima_rmse, metric_name):
        """
        Display comparison of Prophet and ARIMA forecasts with MAE and RMSE metrics.
        """
        st.subheader("📊 Model Comparison: Prophet vs ARIMA")
        
        if prophet_mae is None or prophet_rmse is None or arima_mae is None or arima_rmse is None:
            st.warning("Insufficient data to compare models.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Prophet Model**")
            st.metric("Mean Absolute Error (MAE)", f"{prophet_mae:.2f} kWh")
            st.metric("Root Mean Squared Error (RMSE)", f"{prophet_rmse:.2f} kWh")
        with col2:
            st.markdown("**ARIMA Model**")
            st.metric("Mean Absolute Error (MAE)", f"{arima_mae:.2f} kWh")
            st.metric("Root Mean Squared Error (RMSE)", f"{arima_rmse:.2f} kWh")
        
        # Determine which model is better
        st.markdown("**Conclusion**")
        if prophet_mae < arima_mae and prophet_rmse < arima_rmse:
            st.success(f"Prophet performs better for {metric_name}, with lower MAE ({prophet_mae:.2f} vs {arima_mae:.2f}) and RMSE ({prophet_rmse:.2f} vs {arima_rmse:.2f}).")
        elif arima_mae < prophet_mae and arima_rmse < prophet_rmse:
            st.success(f"ARIMA performs better for {metric_name}, with lower MAE ({arima_mae:.2f} vs {arima_mae:.2f}) and RMSE ({arima_rmse:.2f} vs {prophet_rmse:.2f}).")
        else:
            st.info(f"Performance is mixed for {metric_name}. Prophet has {'lower MAE' if prophet_mae < arima_mae else 'higher MAE'} ({prophet_mae:.2f} vs {arima_mae:.2f}) and {'lower RMSE' if prophet_rmse < arima_rmse else 'higher RMSE'} ({prophet_rmse:.2f} vs {arima_rmse:.2f}). Consider data patterns and forecast horizon when choosing a model.")

        with st.expander("ℹ️ How to interpret MAE and RMSE"):
            st.markdown("""
            - **MAE (Mean Absolute Error)**: Measures the average absolute difference between predicted and actual values. Lower values indicate better accuracy.
            - **RMSE (Root Mean Squared Error)**: Measures the square root of the average squared differences. It penalizes larger errors more heavily than MAE.
            - **Model Selection**: The model with lower MAE and RMSE is generally better. Prophet captures seasonal patterns well, while ARIMA is suited for stationary data with short-term dependencies.
            """)
     
# ====================== WEATHER PAGE ======================
def weather_page(df):
    """
    Weather Impact Analysis: Displays how weather metrics relate to energy usage patterns.
    """
    st.header("🌤️ Weather Impact Analysis")

    if df is None:
        st.warning("No data available")
        return

    df = df.copy()

    if 'temperature' in df.columns and df['temperature'].max() > 80:
        df['temperature'] = (df['temperature'] - 32) * 5.0 / 9.0
        if 'apparentTemperature' in df.columns:
            df['apparentTemperature'] = (df['apparentTemperature'] - 32) * 5.0 / 9.0
        if 'dewPoint' in df.columns:
            df['dewPoint'] = (df['dewPoint'] - 32) * 5.0 / 9.0

    numeric_cols = ['temperature', 'apparentTemperature', 'humidity', 'pressure', 'windSpeed']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['weekday'] = df.index.day_name()
    df['month'] = df.index.month

    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df['season'] = df['month'].apply(get_season)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🌦️ Explore Weather Trends", "📉 Energy vs. Weather", "🔬 Insightful Discoveries"])

    with tab1:
        render_weather_trends_tab(df)

    with tab2:
        render_energy_relationships_tab(df)

    with tab3:
        render_advanced_insights_tab(df)

# Embedded function: Display summary weather metrics as KPIs
def show_weather_metrics(df_input, cols):
        metrics = [
            ('temperature', '🌡️ Avg Temp', '°C'),
            ('apparentTemperature', '🌡️ Feels Like', '°C'),
            ('humidity', '💧 Humidity', '%'),
            ('windSpeed', '🌬️ Wind Speed', 'km/h'),
            ('pressure', '⏲️ Pressure', 'hPa')
        ]
        valid = [(c, n, u) for c, n, u in metrics if c in df_input.columns and pd.api.types.is_numeric_dtype(df_input[c])]
        if not valid:
            st.info("No valid weather metrics found.")
            return
        for i, (col, name, unit) in enumerate(valid):
            if i >= len(cols):
                break
            try:
                avg = df_input[col].mean()
                min_val = df_input[col].min()
                max_val = df_input[col].max()
                delta = f"{min_val:.1f}{unit} → {max_val:.1f}{unit}"
                with cols[i]:
                    st.metric(name, f"{avg:.1f}{unit}", delta)
            except Exception as e:
                st.error(f"Error: {name} – {str(e)}")

def render_weather_trends_tab(df):
    st.subheader("📈 Weather & Energy Trends")

    mode = st.radio("Select Time Range", ["Full Year", "Custom Range"], horizontal=True)

    if mode == "Custom Range":
        date_range = date_filter(df, "weather_range")
        if not date_range or len(date_range) < 2:
            st.error("Please select a valid time range.")
            return
        filtered_df = df[(df.index.date >= date_range[0]) & (df.index.date <= date_range[1])].copy()
    else:
        filtered_df = df.copy()

    if filtered_df.empty:
        st.warning("No data available for the selected range.")
        return

    # Weather Summary Metrics
    st.subheader("Weather Summary")
    show_weather_metrics(filtered_df, st.columns(5))

    level = st.radio("📊 Compare by", ["Day", "Month"], horizontal=True)

    weather_options = {
        'temperature': 'Temperature (°C)',
        'apparentTemperature': 'Apparent Temperature (°C)',
        'humidity': 'Humidity (%)',
        'pressure': 'Pressure (hPa)',
        'windSpeed': 'Wind Speed (km/h)',
        'dewPoint': 'Dew Point (°C)'
    }
    available_weather_cols = [col for col in weather_options if col in filtered_df.columns]
    selected_weather = st.selectbox("🌦️ Select weather variable to compare", available_weather_cols,
                                    format_func=lambda x: weather_options[x])

    if level == "Day":
        daily = filtered_df.resample('D').mean(numeric_only=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily.index, y=daily['use [kW]'], name='Energy Usage (kWh)',
                                 mode='lines+markers', yaxis='y1'))
        if selected_weather in daily.columns:
            fig.add_trace(go.Scatter(x=daily.index, y=daily[selected_weather],
                                     name=weather_options[selected_weather],
                                     mode='lines+markers', yaxis='y2', line=dict(color='orange')))
        fig.update_layout(
            title="Daily Energy Usage and Weather Variable",
            xaxis_title="Date",
            yaxis=dict(title="Energy Usage (kWh)", title_font=dict(color="#1f77b4")),
            yaxis2=dict(title=weather_options[selected_weather], title_font=dict(color="orange"),
                        overlaying='y', side='right'),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif level == "Month":
        monthly = filtered_df.groupby('month')[['use [kW]', selected_weather]].mean(numeric_only=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly.index, y=monthly['use [kW]'], name='Energy Usage (kWh)', yaxis='y1'))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly[selected_weather],
                                 name=weather_options[selected_weather], yaxis='y2',
                                 mode='lines+markers', line=dict(color='orange')))
        fig.update_layout(
            title="Monthly Average Energy Usage and Weather Variable",
            xaxis=dict(title="Month", tickmode='array', tickvals=list(range(1, 13)),
                       ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
            yaxis=dict(title="Energy Usage (kWh)", title_font=dict(color="#1f77b4")),
            yaxis2=dict(title=weather_options[selected_weather], title_font=dict(color="orange"),
                        overlaying='y', side='right'),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_energy_relationships_tab(df):
    with st.expander("📘 Explanation: What does this tab show?"):
        st.markdown("""
        This section explores **how energy consumption relates to different weather variables**.

        - 🔢 **Correlation Matrix**: Shows how strongly energy usage is correlated with weather metrics.  
          → Useful to **quickly detect patterns** (e.g., is high temperature linked to more energy use?).  
          → 🔴 = strong negative, ⚪ = none, 🔵 = strong positive correlation.

        - 📈 **Regression Analysis**: Lets you analyze the **relationship between any weather variable** and **total or device-level energy usage**.  
          → You can choose a **trendline** (OLS or LOWESS) to visualize the correlation strength and shape.  
          → The regression summary helps interpret the **slope (coef)** and **R² score**, indicating predictive power.

        - ☁️ **Energy by Weather Group**: Groups energy usage by **weather condition labels** (e.g., Clear, Rainy).  
          → Provides both **boxplots** for distribution and **bar charts** showing the **likelihood of high usage** under each weather type.  
          → Helpful to identify which conditions often lead to **peak consumption**.

        Together, these tools help you **understand and quantify** how environmental factors influence household energy demand.
        """)


    st.subheader("🔍 Energy Consumption Relationships")
    available_cols = [col for col in WEATHER_COLS if col in df.columns]
    device_cols = ['use [kW]'] + [col for col in DEVICES if col in df.columns]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Correlation Matrix")

        with st.expander("ℹ️ How to read the correlation heatmap"):
            st.markdown("""
            - **Heatmap**: Pearson correlation between energy and weather.
            - 🔴 = strong negative | ⚪ = none | 🔵 = strong positive
            - Diagonal = always 1.
            """)

        cor_df = df[['use [kW]'] + available_cols].corr(numeric_only=True).round(2)
        mask = ~np.eye(len(cor_df), dtype=bool)
        cor_values = cor_df.where(mask)

        max_corr_val = cor_values.max().max()
        min_corr_val = cor_values.min().min()
        max_pair = cor_values.stack().idxmax()
        min_pair = cor_values.stack().idxmin()

        st.markdown(f"""
        **Strongest Positive**: `{max_pair[0]} ↔ {max_pair[1]}` = **{max_corr_val:.2f}**  
        **Strongest Negative**: `{min_pair[0]} ↔ {min_pair[1]}` = **{min_corr_val:.2f}**
        """)

        fig_cor = px.imshow(
            cor_df, text_auto=True, color_continuous_scale='RdBu_r',
            labels=dict(color='Correlation')
        )
        fig_cor.update_layout(
            title={'text': 'Correlation Matrix', 'x': 0.5},
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(size=12),
            coloraxis_colorbar=dict(tickvals=[-1, -0.5, 0, 0.5, 1])
        )
        st.plotly_chart(fig_cor, use_container_width=True)

    with col2:
        st.markdown("### Regression: Device or Total Usage vs Weather Variable")

        selected_var = st.selectbox("Select a weather variable", available_cols)
        selected_energy = st.selectbox("Select energy usage", device_cols, key="energy_vs_weather")
        trend_option = st.radio("Trendline type", ["OLS", "LOWESS", "None"], horizontal=True)

        plot_df = df[[selected_var, selected_energy]].dropna()
        if len(plot_df) > 2000:
            plot_df = plot_df.sample(2000, random_state=42)

        r2 = coef = intercept = None

        if trend_option == "OLS":
            try:
                X = sm.add_constant(plot_df[selected_var])
                y = plot_df[selected_energy]
                model = sm.OLS(y, X).fit()
                coef = model.params[selected_var]
                intercept = model.params['const']
                r2 = model.rsquared
                st.markdown(f"**Regression Summary:** R² = {r2:.2f} | Coef = {coef:.2f} | Intercept = {intercept:.2f}")
            except Exception as e:
                st.error(f"Regression failed: {e}")

        trendline_map = {"OLS": "ols", "LOWESS": "lowess", "None": None}
        fig_reg = px.scatter(
            plot_df, x=selected_var, y=selected_energy,
            trendline=trendline_map[trend_option],
            title=f'{selected_energy} vs {selected_var} (Regression)'
        )

        if trend_option == "OLS" and r2 is not None:
            fig_reg.update_layout(annotations=[{
                'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 1.1,
                'text': f"R² = {r2:.2f} | Coef = {coef:.2f} | Intercept = {intercept:.2f}",
                'showarrow': False
            }])

        st.plotly_chart(fig_reg, use_container_width=True)

    group_order = [
        '☀️ Clear', '🌤️ Partly Cloudy', '⛅️ Mostly Cloudy', '☁️ Overcast',
        '🌦️ Drizzle', '🌧️ Light Rain', '🌧️ Rain',
        '❄️ Light Snow', '❄️ Flurries',
        '💨 Breezy', '🔍 Other'
    ]

    # Group weather summary if not already grouped
    if 'summary' in df.columns and 'weather_group' not in df.columns:
        df['weather_group'] = df['summary'].map({
            'Clear': '☀️ Clear', 'Partly Cloudy': '🌤️ Partly Cloudy',
            'Mostly Cloudy': '⛅️ Mostly Cloudy', 'Overcast': '☁️ Overcast',
            'Drizzle': '🌦️ Drizzle', 'Light Rain': '🌧️ Light Rain',
            'Rain': '🌧️ Rain', 'Light Snow': '❄️ Light Snow',
            'Flurries': '❄️ Flurries', 'Breezy': '💨 Breezy'
        }).fillna('🔍 Other')

    df['weather_group'] = pd.Categorical(df['weather_group'], categories=group_order, ordered=True)

    with st.expander("🌦 Energy Usage by Weather Group"):
        st.markdown("""
        This visualization shows how **energy usage varies across different weather conditions**.  
        Each boxplot represents the distribution of energy consumption for a specific weather group.  

        The **percentage above each box** indicates the **probability that energy usage exceeds the threshold**  
        (calculated as: `mean + 1 standard deviation`), helping identify which weather types are most likely associated with high usage.

        **How to read:**
        - The wider and taller the box, the more variation in usage under that condition.
        - A high percentage label above a group suggests greater risk of high energy load.
        """)

        # Calculate threshold for "high usage": mean + 1 standard deviation
        threshold = df['use [kW]'].mean() + df['use [kW]'].std()

        # Calculate the probability (%) of high usage within each weather group
        prob_df = df.groupby('weather_group').apply(
            lambda g: (g['use [kW]'] > threshold).mean() * 100
        ).reset_index(name='High Usage Probability (%)')

        prob_df['weather_group'] = pd.Categorical(prob_df['weather_group'], categories=group_order, ordered=True)
        prob_df = prob_df.sort_values('weather_group')

        # Create the main boxplot
        fig_group = px.box(
            df, x='weather_group', y='use [kW]', color='weather_group',
            title='Energy Usage by Weather Group'
        )

        # Add percentage annotations above each box
        for i, row in prob_df.iterrows():
            fig_group.add_annotation(
                x=row['weather_group'],
                y=df['use [kW]'].max() * 1.05,
                text=f"{row['High Usage Probability (%)']:.0f}%",
                showarrow=False,
                font=dict(size=12, color='black')
            )

        fig_group.update_layout(
            yaxis_title="Energy Usage [kW]",
            xaxis_title="Weather Group",
            margin=dict(t=60),
            title_font_size=18,
            xaxis_tickangle=25
        )

        st.plotly_chart(fig_group, use_container_width=True)

def safe_sample(df, n=3000):
    return df.sample(n=n, random_state=42) if len(df) > n else df

def render_advanced_insights_tab(df):
    st.markdown("### 📊 Advanced Weather-Energy Analysis")

    available_weather = [col for col in WEATHER_COLS if col in df.columns]

    # ==== 1. Joint Distribution Plot ====
    with st.expander("🔄 Weather vs Usage Joint Distribution"):
        st.markdown("""
        This plot shows a **joint density distribution** between a selected weather variable and energy usage.

        - Darker areas = more data points.
        - Helps identify **correlated regions** or **seasonal weather-energy interactions**.
        - Combines scatter and contour (KDE) visualization.
        """)
        if not available_weather:
            st.warning("No weather columns available.")
        else:
            joint_var = st.selectbox("Select weather variable", available_weather, key="joint_var")
            joint_df = df[[joint_var, 'use [kW]']].dropna()
            joint_df = safe_sample(joint_df)

            fig_joint, ax = plt.subplots(figsize=(5, 3.5))
            sns.kdeplot(data=joint_df, x=joint_var, y='use [kW]', fill=True, cmap="mako", thresh=0.05, levels=100)
            sns.scatterplot(data=joint_df, x=joint_var, y='use [kW]', alpha=0.2, s=10, color="black")
            ax.set_title(f"Joint Distribution: {joint_var} vs Energy Usage")
            ax.set_xlabel(joint_var)
            ax.set_ylabel("Energy Usage (kW)")
            st.pyplot(fig_joint)

    st.markdown("---")

    # ==== 2. SHAP Feature Importance ====
    with st.expander("🔍 SHAP Feature Importance for Energy Prediction"):
        st.markdown("""
        This section uses **XGBoost** to model energy usage and **SHAP** to explain how each weather variable contributes to the prediction.

        **SHAP advantages**:
        - Interprets black-box models by attributing **contributions to each input**.
        - Identifies which **weather features most influence predictions**, and **how**.
        
        **Visuals**:
        - 🐝 **Beeswarm plot**: Direction & magnitude of each variable's impact
        - 🔍 **Dependence plot**: Select a feature to explore its direct effect
        """)

        shap_cols = ['use [kW]'] + available_weather
        shap_df = df[shap_cols].dropna()
        shap_df = safe_sample(shap_df)

        if shap_df.empty:
            st.warning("Not enough data for SHAP analysis.")
        else:
            try:
                
                X = shap_df.drop(columns=['use [kW]'])
                y = shap_df['use [kW]']

                model = xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
                model.fit(X, y)

                explainer = shap.Explainer(model)
                shap_values = explainer(X)

                st.markdown("#### 🐝 SHAP Beeswarm Plot")
                fig_swarm, ax = plt.subplots(figsize=(6, 4)) 
                shap.plots.beeswarm(shap_values, max_display=10, show=False)
                st.pyplot(fig_swarm)

            except Exception as e:

                st.warning(f"SHAP analysis failed: {str(e)}")
                st.error(traceback.format_exc())


    st.markdown("---")

    # ==== 3. Violin Plot Across Seasons ====
    with st.expander("🎻 Weather Variable Distribution Across Seasons"):
        st.markdown("""
        This violin plot displays **how weather metrics are distributed across seasons**.

        - Each shape = **value distribution density** for each season.
        - Use this to analyze **weather variation across the year**.
        - Boxplots and individual points overlaid for detail.
        """)
        if 'season' not in df.columns:
            st.warning("Column 'season' is missing. Cannot display violin plot.")
        elif not available_weather:
            st.warning("No weather columns available for violin plot.")
        else:
            violin_var = st.selectbox("Select weather variable", available_weather, key="violin_weather")
            sample_size = st.radio("Sample size", [500, 1000, 2000], index=1)

            violin_df = df[[violin_var, 'season']].dropna()
            violin_df = safe_sample(violin_df, sample_size)

            if not violin_df.empty:
                fig_violin = px.violin(
                    violin_df, x='season', y=violin_var, color='season',
                    box=True, points='all',
                    title=f'{violin_var} Distribution Across Seasons'
                )
                st.plotly_chart(fig_violin, use_container_width=True)
            else:
                st.info("Not enough data to display violin plot.")

    st.markdown("---")

    # ==== 4. Anomaly Detection by Month ====
    with st.expander("🚨 Monthly Anomaly Detection in Energy Usage"):
        st.markdown("""
        This section detects and visualizes **monthly anomalies in energy usage** using Isolation Forest.  
        You can explore anomalies by selecting a specific month.
        """)



        df['month'] = df.index.to_series().dt.to_period('M').astype(str)
        df['Anomaly_Month'] = 1  

        for month in df['month'].unique():
            month_df = df[df['month'] == month][['use [kW]', 'temperature', 'humidity']].dropna()
            if len(month_df) > 100:
                X_scaled = StandardScaler().fit_transform(month_df)
                preds = IsolationForest(contamination=0.03, random_state=42).fit_predict(X_scaled)
                df.loc[month_df.index, 'Anomaly_Month'] = preds  # -1 = anomaly

        available_months = sorted(df['month'].dropna().unique())
        selected_month = st.selectbox("Select a month to explore anomalies", available_months)

        month_df = df[df['month'] == selected_month]

        if month_df.empty:
            st.warning("No data available for selected month.")
        else:
            fig_month = go.Figure()
            fig_month.add_trace(go.Scatter(
                x=month_df.index, y=month_df['use [kW]'],
                mode='lines', name='Energy Usage'
            ))
            fig_month.add_trace(go.Scatter(
                x=month_df[month_df['Anomaly_Month'] == -1].index,
                y=month_df[month_df['Anomaly_Month'] == -1]['use [kW]'],
                mode='markers', name='Anomalies',
                marker=dict(color='red', size=6)
            ))
            fig_month.update_layout(
                title=f"🚨 Energy Usage with Anomalies in {selected_month}",
                xaxis_title="Time", yaxis_title="use [kW]"
            )
            st.plotly_chart(fig_month, use_container_width=True)

# ====================== MAIN APP ======================
# ====================== SIDEBAR ================================
def render_sidebar(processed_df):
    with st.sidebar:
        st.title("🏠 Navigation")

        # Custom radio button style
        st.markdown("""
            <style>
            .stRadio > div {
                flex-direction: row;
                gap: 10px;
                justify-content: center;
            }
            div[role=radiogroup] > label {
                background-color: #f0f2f6;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                border: 2px solid transparent;
                transition: 0.2s ease-in-out;
            }
            div[role=radiogroup] > label:hover {
                border-color: #1f77b4;
            }
            div[role=radiogroup] > label[data-selected="true"] {
                background-color: #1f77b4;
                color: white;
                border-color: #1f77b4;
            }
            </style>
        """, unsafe_allow_html=True)

        tabs = ["🏠 Overview", "🔌 Devices", "📈 Forecasting", "🌤️ Weather"]

        selected_tab = st.radio(
            "Select a page:",
            tabs,
            index=tabs.index(st.session_state.get("selected_tab", tabs[0])),
            label_visibility="collapsed"
        )

        page = selected_tab

        st.markdown("---")
        st.markdown("### 📅 Data Summary")

        if processed_df is not None:
            try:
                st.metric("Date Range", 
                          f"{processed_df['date'].min().strftime('%d/%m/%Y')} → "
                          f"{processed_df['date'].max().strftime('%d/%m/%Y')}")
            except Exception as e:
                st.error(f"Summary display error: {str(e)}")
        else:
            st.warning("No data available")

        st.markdown("---")

        if processed_df is not None and st.button("📥 Download Sample (CSV)"):
            try:
                sample = processed_df.sample(min(1000, len(processed_df)))
                csv = sample.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="data_sample.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Sample creation error: {str(e)}")

        return page

# ====================== MAIN ENTRY ============================
def main():
    with st.spinner("🔄 Loading data..."):
        df = load_data()

    with st.spinner("⚙️ Processing data..."):
        processed_df = process_data(df)

    if processed_df is not None:
        page = render_sidebar(processed_df)

        if page == "🏠 Overview":
            overview_page(processed_df)
        elif page == "🔌 Devices":
            devices_page(processed_df)
        elif page == "📈 Forecasting":
            forecasting_page(processed_df)
        elif page == "🌤️ Weather":
            weather_page(processed_df)
    else:
        st.error("❌ Failed to load data. Please check data file and try again.")

if __name__ == "__main__":
    main()
