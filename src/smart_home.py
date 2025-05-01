import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px


st.set_page_config(page_title="Smart Home Energy Dashboard", layout="wide", page_icon="🏠")

# Hàm tải dữ liệu
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("../data/HomeC.csv")
        # Đổi tên cột time thành datetime để rõ ràng hơn
        df = df.rename(columns={'time': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except:
        st.error("Không tìm thấy file dữ liệu 'smart_home_data.csv'")
        return None

# Tiền xử lý dữ liệu
def preprocess_data(df):
    # Tạo các cột thời gian
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    return df

# Dashboard Tổng quan
def overview_dashboard(df):
    st.header("🏠 Tổng quan Tiêu thụ Năng lượng")
    
    # Lựa chọn ngày
    date_range = st.date_input("Chọn khoảng thời gian", 
                              [df['date'].min(), df['date'].max()])
    
    if len(date_range) == 2:
        mask = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
        filtered_df = df[mask]
    else:
        filtered_df = df
    
    # Tính toán các chỉ số tổng quan
    total_use = filtered_df['use [kW]'].sum()
    total_gen = filtered_df['gen [kW]'].sum()
    net_energy = total_use - total_gen
    avg_temp = filtered_df['temperature'].mean()
    
    # Hiển thị các metric
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng tiêu thụ (kW)", f"{total_use:,.2f}")
    col2.metric("Tổng sản xuất (kW)", f"{total_gen:,.2f}")
    col3.metric("Năng lượng ròng (kW)", f"{net_energy:,.2f}", 
                "Tiết kiệm" if net_energy < 0 else "Tiêu thụ thêm")
    col4.metric("Nhiệt độ trung bình (°C)", f"{avg_temp:.1f}")
    
    # Biểu đồ tổng quan
    tab1, tab2, tab3 = st.tabs(["Tiêu thụ & Sản xuất", "Xu hướng theo giờ", "Phân bố nhiệt độ"])
    
    with tab1:
        fig = px.line(filtered_df, x='datetime', y=['use [kW]', 'gen [kW]', 'House overall [kW]'], 
                      title='Tiêu thụ và Sản xuất Năng lượng theo thời gian')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        hourly_avg = filtered_df.groupby('hour')[['use [kW]', 'gen [kW]']].mean().reset_index()
        fig = px.bar(hourly_avg, x='hour', y=['use [kW]', 'gen [kW]'], 
                     barmode='group', title='Tiêu thụ trung bình theo giờ')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.density_heatmap(filtered_df, x='datetime', y='temperature', 
                                title='Phân bố nhiệt độ theo thời gian')
        st.plotly_chart(fig, use_container_width=True)

# Dashboard Chi tiết thiết bị
def devices_dashboard(df):
    st.header("🔌 Chi tiết Thiết bị")
    
    # Danh sách thiết bị
    devices = ['Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]', 
               'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]', 
               'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]', 
               'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]', 
               'Microwave [kW]', 'Living room [kW]', 'Solar [kW]']
    
    # Lựa chọn thiết bị
    selected_devices = st.multiselect("Chọn thiết bị để phân tích", devices, default=devices[:5])
    
    if selected_devices:
        # Lựa chọn ngày
        date_range = st.date_input("Chọn khoảng thời gian (cho thiết bị)", 
                                  [df['date'].min(), df['date'].max()], key='devices_date')
        
        if len(date_range) == 2:
            mask = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
            filtered_df = df[mask]
        else:
            filtered_df = df
        
        # Tính tổng năng lượng theo thiết bị
        device_totals = filtered_df[selected_devices].sum().sort_values(ascending=False)
        
        # Hiển thị biểu đồ
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tổng tiêu thụ theo thiết bị")
            fig = px.pie(device_totals, values=device_totals.values, names=device_totals.index,
                         title='Phân bổ năng lượng theo thiết bị')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("So sánh thiết bị")
            fig = px.bar(device_totals, x=device_totals.index, y=device_totals.values,
                         title='Tổng năng lượng tiêu thụ theo thiết bị')
            st.plotly_chart(fig, use_container_width=True)
        
        # Biểu đồ chi tiết theo thời gian
        st.subheader("Biểu đồ tiêu thụ theo thời gian")
        fig = px.line(filtered_df, x='datetime', y=selected_devices,
                     title='Tiêu thụ năng lượng theo thời gian')
        st.plotly_chart(fig, use_container_width=True)
        
        # Phân tích tương quan
        st.subheader("Phân tích tương quan giữa các thiết bị")
        corr_matrix = filtered_df[selected_devices].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title='Ma trận tương quan giữa các thiết bị')
        st.plotly_chart(fig, use_container_width=True)

# Dashboard Thông tin thời tiết
def weather_dashboard(df):
    st.header("🌤️ Thông tin Thời tiết")
    
    # Lựa chọn ngày
    date_range = st.date_input("Chọn khoảng thời gian (cho thời tiết)", 
                              [df['date'].min(), df['date'].max()], key='weather_date')
    
    if len(date_range) == 2:
        mask = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
        filtered_df = df[mask]
    else:
        filtered_df = df
    
    # Hiển thị các chỉ số thời tiết
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nhiệt độ trung bình", f"{filtered_df['temperature'].mean():.1f}°C")
    col2.metric("Độ ẩm trung bình", f"{filtered_df['humidity'].mean():.1f}%")
    col3.metric("Tốc độ gió trung bình", f"{filtered_df['windSpeed'].mean():.1f} km/h")
    col4.metric("Áp suất trung bình", f"{filtered_df['pressure'].mean():.1f} hPa")
    
    # Biểu đồ thời tiết
    tab1, tab2, tab3 = st.tabs(["Nhiệt độ & Độ ẩm", "Gió & Áp suất", "Thông tin khác"])
    
    with tab1:
        fig = px.line(filtered_df, x='datetime', y=['temperature', 'apparentTemperature', 'dewPoint'],
                     title='Nhiệt độ theo thời gian')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.line(filtered_df, x='datetime', y='humidity',
                     title='Độ ẩm theo thời gian')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.line(filtered_df, x='datetime', y=['windSpeed', 'windBearing'],
                     title='Tốc độ và hướng gió')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.line(filtered_df, x='datetime', y='pressure',
                     title='Áp suất theo thời gian')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.line(filtered_df, x='datetime', y=['visibility', 'precipIntensity'],
                     title='Tầm nhìn và cường độ mưa')
        st.plotly_chart(fig, use_container_width=True)
        
        # Phân bổ điều kiện thời tiết
        weather_counts = filtered_df['summary'].value_counts().reset_index()
        weather_counts.columns = ['summary', 'count']
        fig = px.bar(weather_counts, x='summary', y='count',
                    title='Phân bổ điều kiện thời tiết')
        st.plotly_chart(fig, use_container_width=True)

# Sidebar điều hướng
def main():
    st.sidebar.title("Điều hướng")
    page = st.sidebar.radio("Chọn dashboard", 
                           ["🏠 Tổng quan", "🔌 Thiết bị", "🌤️ Thời tiết"])
    
    # Tải dữ liệu
    df = load_data()
    if df is not None:
        df = preprocess_data(df)
        
        if page == "🏠 Tổng quan":
            overview_dashboard(df)
        elif page == "🔌 Thiết bị":
            devices_dashboard(df)
        elif page == "🌤️ Thời tiết":
            weather_dashboard(df)
    else:
        st.warning("Vui lòng tải lên file dữ liệu 'smart_home_data.csv'")

if __name__ == "__main__":
    main()