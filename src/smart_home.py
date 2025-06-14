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

def show_weather_metrics(df, cols):
    metrics = [
        ('temperature', '🌡️ Avg Temp', '°C'),
        ('apparentTemperature', '🌡️ Feels Like', '°C'),
        ('humidity', '💧 Humidity', '%'),
        ('windSpeed', '🌬️ Wind Speed', 'km/h'),
        ('pressure', '⏲️ Pressure', 'hPa')
    ]
    valid = [(c, n, u) for c, n, u in metrics if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    for i, (col, name, unit) in enumerate(valid[:len(cols)]):
        try:
            avg = df[col].mean()
            min_val = df[col].min()
            max_val = df[col].max()
            delta = f"{min_val:.1f}{unit} → {max_val:.1f}{unit}"  
            with cols[i]:
                st.metric(name, f"{avg:.1f}{unit}", delta)
        except Exception as e:
            st.error(f"Error: {name} – {str(e)}")


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
                     trendline='lowess', title='Temperature vs Consumption')
    st.plotly_chart(fig, use_container_width=True)

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
                    x=daily_consumption.index.name or 'index',  # fallback if unnamed
                    y=selected_device,
                    title=f"{selected_device.replace(' [kW]', '')} Daily Consumption",
                    labels={
                        daily_consumption.index.name or 'index': 'Date',
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
    Weather Impact Analysis:
    Displays how weather metrics relate to energy usage patterns.
    """
    st.header("🌤️ Weather Impact Analysis")

    if df is None:
        st.warning("No data available")
        return

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Weather Trends", "🔍 Energy Relationships", "📊 Advanced Insights"])

    # ========== TAB 1: WEATHER TRENDS ==========
    with tab1:
        st.subheader("📈 Weather & Energy Trends")

        # Ensure datetime info is available
        df['date'] = df.index.date
        df['hour'] = df.index.hour
        df['weekday'] = df.index.day_name()

        mode = st.radio("Select Time Range", ["Full Year", "Custom Range"], horizontal=True)

        if mode == "Custom Range":
            date_range = date_filter(df, "weather_range")
            if not date_range or len(date_range) < 2:
                st.error("Please select a valid time range.")
                return
            filtered_df = df[(df.index.date >= date_range[0]) & (df.index.date <= date_range[1])].copy()
        else:
            filtered_df = df.copy()

        # General Weather Summary (keep this section)
        st.subheader("Weather Summary")
        show_weather_metrics(filtered_df, st.columns(5))

        # Aggregation level
        level = st.radio("📊 Compare by", ["Day", "Month"], horizontal=True)

        # Popular weather variables
        weather_options = {
            'temperature': 'Temperature (°C)',
            'apparentTemperature': 'Apparent Temperature (°C)',
            'humidity': 'Humidity (%)',
            'pressure': 'Pressure (hPa)',
            'windSpeed': 'Wind Speed (km/h)',
            'dewPoint': 'Dew Point (°C)'
        }

        # Select weather variable
        available_weather_cols = [col for col in weather_options if col in filtered_df.columns]
        selected_weather = st.selectbox(
            "🌦️ Select weather variable to compare",
            available_weather_cols,
            format_func=lambda x: weather_options[x]
        )

        if level == "Day":
            daily = filtered_df.resample('D').mean(numeric_only=True)

            import plotly.graph_objects as go
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=daily.index,
                y=daily['use [kW]'],
                name='Energy Usage (kWh)',
                mode='lines+markers',
                yaxis='y1'
            ))

            if selected_weather in daily.columns:
                fig.add_trace(go.Scatter(
                    x=daily.index,
                    y=daily[selected_weather],
                    name=weather_options[selected_weather],
                    mode='lines+markers',
                    yaxis='y2',
                    line=dict(color='orange')
                ))

            fig.update_layout(
                title="Daily Energy Usage and Weather Variable",
                xaxis_title="Date",
                yaxis=dict(
                    title="Energy Usage (kWh)",
                    titlefont=dict(color="#1f77b4"),
                    tickfont=dict(color="#1f77b4")
                ),
                yaxis2=dict(
                    title=weather_options[selected_weather],
                    titlefont=dict(color="orange"),
                    tickfont=dict(color="orange"),
                    overlaying='y',
                    side='right'
                ),
                hovermode="x unified",
                legend=dict(x=0.01, y=0.99),
                margin=dict(t=60, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

        elif level == "Month":
            filtered_df['month'] = filtered_df.index.month
            monthly = filtered_df.groupby('month')[['use [kW]', selected_weather]].mean(numeric_only=True)

            import plotly.graph_objects as go
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=monthly.index,
                y=monthly['use [kW]'],
                name='Energy Usage (kWh)',
                yaxis='y1'
            ))

            if selected_weather in monthly.columns:
                fig.add_trace(go.Scatter(
                    x=monthly.index,
                    y=monthly[selected_weather],
                    name=weather_options[selected_weather],
                    yaxis='y2',
                    mode='lines+markers',
                    line=dict(color='orange')
                ))

            fig.update_layout(
                title="Monthly Average Energy Usage and Weather Variable",
                xaxis=dict(
                    title="Month",
                    tickmode='array',
                    tickvals=list(range(1, 13)),
                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ),
                yaxis=dict(
                    title="Energy Usage (kWh)",
                    titlefont=dict(color="#1f77b4"),
                    tickfont=dict(color="#1f77b4")
                ),
                yaxis2=dict(
                    title=weather_options[selected_weather],
                    titlefont=dict(color="orange"),
                    tickfont=dict(color="orange"),
                    overlaying='y',
                    side='right'
                ),
                barmode='group',
                hovermode="x unified",
                legend=dict(x=0.01, y=0.99),
                margin=dict(t=60, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)


    # ========== TAB 2: Placeholder (unchanged) ==========
    with tab2:
        st.subheader("🔍 Energy Consumption Relationships")

        weather_cols = [
            'temperature', 'humidity', 'windSpeed', 
            'windBearing', 'pressure', 'apparentTemperature',
            'dewPoint', 'precipProbability'
        ]
        available_cols = [col for col in weather_cols if col in filtered_df.columns]
        device_cols = ['use [kW]'] + [col for col in DEVICES if col in filtered_df.columns]

        col1, col2 = st.columns(2)

        with col1:
            # Correlation Matrix
            st.markdown("### Correlation Matrix")
            import plotly.express as px
            cor_df = filtered_df[['use [kW]'] + available_cols].corr().round(2)
            fig_cor = px.imshow(
                cor_df,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title='Correlation Matrix: Energy Usage and Weather Metrics',
                aspect='auto',
                height=500,
                labels=dict(color='Correlation')
            )
            fig_cor.update_layout(
                margin=dict(l=40, r=40, t=60, b=40),
                font=dict(size=12),
                xaxis=dict(side='bottom', tickangle=45),
                yaxis=dict(tickmode='linear')
            )
            st.plotly_chart(fig_cor, use_container_width=True)

        with col2:
            # Regression line: Any Energy Variable vs Weather Variable
            st.markdown("### Regression: Device or Total Usage vs Weather Variable")
            selected_var = st.selectbox("Select a weather variable", available_cols, index=0)
            selected_energy = st.selectbox("Select energy usage", device_cols, index=0, key="energy_vs_weather")

            if selected_var in filtered_df.columns and selected_energy in filtered_df.columns:
                fig_reg = px.scatter(
                    filtered_df, x=selected_var, y=selected_energy,
                    trendline='ols',
                    title=f'{selected_energy} vs {selected_var} (with Regression Line)',
                    labels={selected_var: selected_var, selected_energy: 'Energy Usage (kW)'}
                )
                st.plotly_chart(fig_reg, use_container_width=True)
            else:
                st.info("Selected variables not available for regression analysis.")

        if 'summary' in filtered_df.columns:
            filtered_df['weather_group'] = filtered_df['summary'].apply(
                lambda s: (
                    '☀️ Clear' if s == 'Clear'
                    else '⛅️ Mostly Cloudy' if s == 'Mostly Cloudy'
                    else '☁️ Overcast' if s == 'Overcast'
                    else '🌤️ Partly Cloudy' if s == 'Partly Cloudy'
                    else '🌦️ Drizzle' if s == 'Drizzle'
                    else '🌧️ Light Rain' if s == 'Light Rain'
                    else '🌧️ Rain' if s == 'Rain'
                    else '❄️ Light Snow' if s == 'Light Snow'
                    else '❄️ Flurries' if s == 'Flurries'
                    else '💨 Breezy' if s == 'Breezy'
                    else '🔍 Other'
                )
            )

            st.markdown("### 🌦️ Energy Usage by Weather Group (Based on Summary)")
            fig_group = px.box(
                filtered_df,
                x='weather_group',
                y='use [kW]',
                color='weather_group',
                title='Energy Usage Distribution by Weather Summary Group',
                labels={'weather_group': 'Weather Group', 'use [kW]': 'Energy Usage (kW)'}
            )
            st.plotly_chart(fig_group, use_container_width=True)
        else:
            st.info("Column 'summary' not found.")


    # ========== TAB 3: Placeholder (unchanged) ==========
    with tab3:
        # 1. Weather-Based Device Usage Clustering
        st.markdown("### 🔀 Weather-Based Device Usage Clustering")
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans

        cluster_cols = ['use [kW]'] + [col for col in weather_cols if col in filtered_df.columns]
        cluster_df = filtered_df[cluster_cols].dropna().copy()

        if len(cluster_df) > 1000:
            cluster_df_sample = cluster_df.sample(1000, random_state=42)
        else:
            cluster_df_sample = cluster_df

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_df_sample)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)

        cluster_df_sample['Cluster'] = labels
        cluster_df_sample['PC1'] = X_pca[:, 0]
        cluster_df_sample['PC2'] = X_pca[:, 1]

        fig_cluster = px.scatter(
            cluster_df_sample,
            x='PC1', y='PC2', color='Cluster',
            title="K-Means Clustering of Weather & Energy Patterns",
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        # 2. Seasonal Comparison (Bar + Violin Combined)
        st.markdown("### 🌱 Seasonal Weather and Energy Analysis")

        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Autumn"

        full_df = df.copy()
        full_df['month'] = full_df.index.month
        full_df['season'] = full_df['month'].apply(get_season)

        # Bar plot: Energy usage by season
        seasonal_usage = full_df.groupby('season')['use [kW]'].mean().reindex(['Winter', 'Spring', 'Summer', 'Autumn'])
        fig_season_usage = px.bar(
            seasonal_usage,
            x=seasonal_usage.index,
            y=seasonal_usage.values,
            labels={'x': 'Season', 'y': 'Average Energy Usage (kW)'},
            title='Average Energy Usage by Season'
        )
        st.plotly_chart(fig_season_usage, use_container_width=True)

        # Violin plot: Weather variable by season
        st.markdown("### 🎻 Weather Variable Distribution by Season")
        violin_var = st.selectbox("Select weather variable", weather_cols, key="violin_weather")
        if violin_var in full_df.columns:
            fig_violin = px.violin(
                full_df,
                x='season', y=violin_var, color='season',
                box=True, points='all',
                title=f'{violin_var} Distribution Across Seasons'
            )
            st.plotly_chart(fig_violin, use_container_width=True)


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